"""Freeze 32,768 distinct UltraChat train_sft user/assistant pairs for online distillation."""
import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
import re
from pathlib import Path
import unicodedata

import pyarrow.parquet as pq
import torch
from specforge.utils import load_tokenizer

REPO = "HuggingFaceH4/ultrachat_200k"
REV = "8049631c405ae6576f93f445c6b8166f76f5505a"
TARGET_TOKENIZER = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
COUNT = 32768


def digest(value):
    return sha256(value.encode()).hexdigest()


def file_sha(path):
    with path.open("rb") as data:
        return file_digest(data, "sha256").hexdigest()


def prompt_hashes(question):
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    near = " ".join(re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE).split())
    return digest(normalized), digest(near)


def iterate(shards, exclusions):
    seen, families = set(), set()
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=512, columns=["prompt_id", "messages"]):
            for row in batch.to_pylist():
                messages = row["messages"]
                if not messages or len(messages) < 2 or [v["role"] for v in messages[:2]] != ["user", "assistant"]:
                    continue
                question, answer = (v["content"] for v in messages[:2])
                if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
                    continue
                uid = "ultrachat/" + row["prompt_id"]
                if uid in families:
                    continue
                families.add(uid)
                exact, near = prompt_hashes(question)
                if int(near[:8], 16) % 100 >= 50 or near in seen or exact in exclusions or near in exclusions:
                    continue
                seen.add(near)
                yield digest("online-ultrachat-original-v1/" + near), uid, exact, near, question, answer


def write_frozen(args):
    pinned = json.loads((args.source / "manifest.json").read_text())
    assert (pinned["repo"], pinned["revision"], pinned["split"]) == (REPO, REV, "train_sft")
    assert file_sha(args.model / "tokenizer.json") == TARGET_TOKENIZER
    shards = [args.source / f["path"] for f in pinned["files"]]
    for path, f in zip(shards, pinned["files"]):
        assert path.stat().st_size == f["bytes"] and file_sha(path) == f["sha256"], path.name
    blocked = set(args.blocklist.read_text().splitlines())
    assert len(blocked) > 8192 and all(len(x) == 64 for x in blocked)
    deny = set(args.denylist.read_text().splitlines())
    blocked.update(deny)
    candidates = sorted(iterate(shards, blocked))
    print(f"Eligible pre-tokenization candidates: {len(candidates)}", flush=True)
    tok = load_tokenizer(str(args.model))
    mapping = torch.load(args.vocab, map_location="cpu", weights_only=True)
    t2d = mapping["t2d"].tolist()
    assert len(t2d) == 151936 and sum(t2d) == 32000
    selected, audit, tokens, supervised, mapped = [], [], 0, 0, 0
    lengths = Counter()
    for offset in range(0, len(candidates), 128):
        batch = candidates[offset: offset + 128]
        pairs = [[{"role":"user","content":r[4]}, {"role":"assistant","content":r[5]}] for r in batch]
        prefill = [tok.apply_chat_template(p[:1], tokenize=False, add_generation_prompt=True) for p in pairs]
        conversation = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=False) for p in pairs]
        options = dict(add_special_tokens=False, return_attention_mask=False, truncation=True)
        prompts = tok(prefill, max_length=897, **options)["input_ids"]
        full = tok(conversation, max_length=1536, **options)["input_ids"]
        for (_, uid, exact, near, _, _), prompt, inputs in zip(batch, prompts, full):
            n = len(prompt)
            m = len(inputs) - n
            if n > 896 or m < 8 or inputs[:n] != prompt:
                continue
            selected.append({"id": uid, "prompt_len": n, "input_ids": inputs, "loss_mask": [0]*n + [1]*m})
            audit.append({"id": uid, "family": uid, "normalized_sha256": exact, "near_sha256": near, "prompt_len": n, "answer_len": m})
            tokens += len(inputs)
            supervised += m
            mapped += sum(t2d[i] for i in inputs[n:])
            lengths[len(inputs)] += 1
            if len(selected) == COUNT:
                break
        if len(selected) == COUNT:
            break
    assert len({x["id"] for x in selected}) == len({x["near_sha256"] for x in audit}) == COUNT
    assert not ({x["normalized_sha256"] for x in audit} | {x["near_sha256"] for x in audit}) & blocked
    args.out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("train.jsonl", selected), ("families.jsonl", audit)):
        with (args.out / name).open("x") as file:
            for row in rows:
                file.write(json.dumps(row, separators=(",", ":")) + "\n")
    files = {x.name: {"bytes": x.stat().st_size, "sha256": file_sha(x)} for x in sorted(args.out.iterdir())}
    manifest = dict(source=pinned, assistant_source="original UltraChat SFT assistant text", rows=COUNT, unique_prompts=COUNT,
                    input_tokens=tokens, supervised_tokens=supervised, vocab_sha256=file_sha(args.vocab),
                    mapped_supervised_fraction=mapped/supervised, token_min=min(lengths), token_max=max(lengths),
                    tokens_above_640=sum(n for size,n in lengths.items() if size>640), tokenizer_sha256=TARGET_TOKENIZER,
                    sharegpt_blocklist_sha256=file_sha(args.blocklist), eval_denylist_sha256=file_sha(args.denylist),
                    selection="first valid rows in ascending SHA256(online-ultrachat-original-v1/ + punctuation-insensitive hash); bucket 0-49",
                    files=files)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "vocab", "blocklist", "denylist", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    write_frozen(parser.parse_args())
