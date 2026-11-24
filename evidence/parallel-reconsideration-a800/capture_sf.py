import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from specforge.data.preprocessing import build_eagle3_dataset, generate_vocab_mapping_file

root = Path("/dev/shm/prc-pilot")
model_path = root / "models/qwen25"
out = root / "cache/sf"
out.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
with (root / "data/ultrachat_2048.jsonl").open() as stream:
    source = [{"conversations": json.loads(line)["conversations"]} for line in stream]
data = build_eagle3_dataset(
    dataset=Dataset.from_list(source), tokenizer=tokenizer, chat_template="qwen",
    max_length=512, cache_dir=str(root / "data/tokenized"), cache_key="qwen25_prc",
    num_proc=4,
)
assert len(data) > 0
generate_vocab_mapping_file(
    dataset=data, target_vocab_size=151936, draft_vocab_size=16000,
    cache_dir=str(out / "vocab_mapping"),
)
print("Captured inputs ready", len(data), flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, local_files_only=True,
    attn_implementation="sdpa",
).eval().to("cuda")
start = time.monotonic()
with torch.inference_mode():
    for offset in range(0, len(data), 8):
        rows = [data[i] for i in range(offset, min(len(data), offset + 8))]
        ids = [row["input_ids"].reshape(-1).long() for row in rows]
        lengths = [x.numel() for x in ids]
        batch = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
        mask = torch.arange(batch.shape[1], device=batch.device)[None] < torch.tensor(lengths, device=batch.device)[:, None]
        features = model.model(input_ids=batch, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
        # HF hidden_states[0] is the input embedding. The selected values are
        # post-block outputs for the capture layers in the recipe.
        aux = torch.cat([features[i + 1] for i in (1, 11, 20)], dim=-1).cpu()
        final = features[-1].cpu()
        for j, (row, tokens, n) in enumerate(zip(rows, ids, lengths)):
            record = {
                "input_ids": tokens.clone(),
                "loss_mask": row["loss_mask"].reshape(-1).clone(),
                "hidden_states": aux[j, :n].contiguous().clone(),
                "target_last_hidden_states": final[j, :n].contiguous().clone(),
            }
            assert record["hidden_states"].shape == (n, 2688)
            assert record["target_last_hidden_states"].shape == (n, 896)
            p = out / f"row_{offset + j:06d}.ckpt"
            tmp = p.with_suffix(".tmp")
            torch.save(record, tmp)
            tmp.replace(p)
        if offset % 256 == 0:
            print(f"capture rows={offset+len(rows)} elapsed={time.monotonic()-start:.2f}", flush=True)
print(f"captured rows={len(data)} elapsed={time.monotonic()-start:.2f} peak_allocated={torch.cuda.max_memory_allocated()}", flush=True)
