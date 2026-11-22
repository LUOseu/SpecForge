"""Build the original-ShareGPT training partition from a pinned source and frozen exclusions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import torch
from specforge.data.preprocessing import process_token_dict_to_mappings
from specforge.utils import load_tokenizer

SOURCE_BYTES = 464322921
SOURCE_SHA = "dd5828821b7e707ca3dc4d0de07e2502c3ce278fcf1a74b81a3464f26006371e"
TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def question_hashes(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    near = " ".join(re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE).split())
    return digest(normalized), digest(near)


def source_rows(path):
    decoder = json.JSONDecoder()
    skip = re.compile(r"[\s,]*")
    with path.open(encoding="utf8") as stream:
        block = stream.read(1 << 20)
        pos = block.index("[") + 1
        while True:
            pos = skip.match(block, pos).end()
            if pos < len(block) and block[pos] == "]":
                return
            try:
                row, pos = decoder.raw_decode(block, pos)
            except json.JSONDecodeError:
                extra = stream.read(1 << 20)
                if not extra:
                    raise
                block = block[pos:] + extra
                pos = 0
                continue
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "denylist", "smoke-manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=8192)
    args = parser.parse_args()
    if (
        args.source.stat().st_size != SOURCE_BYTES
        or file_sha(args.source) != SOURCE_SHA
    ):
        raise ValueError("Source differs from pinned ShareGPT SHA-256")
    if file_sha(args.model / "tokenizer.json") != TOKENIZER_SHA:
        raise ValueError("Target tokenizer SHA-256 mismatch")
    external_deny = set(args.denylist.read_text().splitlines())
    smoke = json.loads(args.smoke_manifest.read_text())
    prior_ids = set(smoke["train_ids"] + smoke["held_out_ids"])
    prior_families = {re.sub(r"_\d+$", "", i) for i in prior_ids}
    smoke_hashes = set()
    source = []
    seen_family = set()
    seen_near = set()
    for row in source_rows(args.source):
        row_id = row["id"]
        messages = row["conversations"]
        if not messages or messages[0].get("from") != "human":
            continue
        question = messages[0].get("value", "")
        if not isinstance(question, str) or not question.strip():
            continue
        exact, near = question_hashes(question)
        if row_id in prior_ids:
            smoke_hashes.update((exact, near))
        if (
            not row_id.endswith("_0")
            or len(messages) < 2
            or messages[1].get("from") != "gpt"
        ):
            continue
        family = row_id[:-2]
        answer = messages[1].get("value", "")
        if not isinstance(answer, str) or not answer.strip():
            continue
        if (
            family in prior_families
            or family in seen_family
            or near in seen_near
            or exact in external_deny
            or int(near[:8], 16) % 100 >= 30
        ):
            continue
        seen_family.add(family)
        seen_near.add(near)
        source.append(
            (
                digest("original-training-v1/" + family),
                family,
                exact,
                near,
                question,
                answer,
            )
        )
    source = [
        x
        for x in sorted(source)
        if x[2] not in smoke_hashes and x[3] not in smoke_hashes
    ]
    tokenizer = load_tokenizer(str(args.model))
    counts = Counter()
    selected = []
    audit = []
    for start in range(0, len(source), 64):
        batch = source[start : start + 64]
        pairs = [
            [{"role": "user", "content": r[4]}, {"role": "assistant", "content": r[5]}]
            for r in batch
        ]
        prompt_text = [
            tokenizer.apply_chat_template(
                pair[:1], tokenize=False, add_generation_prompt=True
            )
            for pair in pairs
        ]
        complete_text = [
            tokenizer.apply_chat_template(
                pair, tokenize=False, add_generation_prompt=False
            )
            for pair in pairs
        ]
        prompts = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=True,
            max_length=897,
        )["input_ids"]
        completes = tokenizer(
            complete_text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=True,
            max_length=1536,
        )["input_ids"]
        for (_, family, exact, near, _, _), prompt, full in zip(
            batch, prompts, completes
        ):
            n = len(prompt)
            m = len(full) - n
            if n > 896 or m < 8 or full[:n] != prompt:
                continue
            counts.update(full[n:])
            selected.append(
                {
                    "id": family + "_0",
                    "prompt_len": n,
                    "input_ids": full,
                    "loss_mask": [0] * n + [1] * m,
                }
            )
            audit.append(
                {
                    "id": family + "_0",
                    "family": family,
                    "normalized_sha256": exact,
                    "near_sha256": near,
                    "prompt_len": n,
                    "answer_len": m,
                }
            )
            if len(selected) >= args.max_samples:
                break
        if len(selected) >= args.max_samples:
            break
    usable = len(selected) // 8 * 8
    if usable < 512:
        raise ValueError(f"Insufficient eligible samples: {usable}")
    selected = selected[:usable]
    audit = audit[:usable]
    counts = Counter(
        token for row in selected for token in row["input_ids"][row["prompt_len"] :]
    )
    observed = len(counts)
    answer_tokens = sum(counts.values())
    d2t, t2d = process_token_dict_to_mappings(counts.copy(), 32000, 151936)
    assert (
        d2t.shape == (32000,) and t2d.shape == (151936,) and t2d.sum().item() == 32000
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    with (output / "train.jsonl").open("w") as f:
        for row in selected:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (output / "families.jsonl").open("w") as f:
        for row in audit:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    torch.save({"d2t": d2t, "t2d": t2d}, output / "vocab.pt")
    manifest = {
        "input_source": "Aeala/ShareGPT_Vicuna_unfiltered@8b0048ad6ae8c22f46a78c15559dec98feef5539/ShareGPT_V4.3_unfiltered_cleaned_split.json",
        "input_sha256": SOURCE_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "external_denylist_sha256": file_sha(args.denylist),
        "prior_smoke_ids": sorted(prior_ids),
        "partition": "original first user and assistant turns, original row suffix _0, normalized near-prompt SHA buckets 0..29",
        "usable_samples": len(selected),
        "supervised_tokens": answer_tokens,
        "input_tokens": sum(len(r["input_ids"]) for r in selected),
        "longer_than_640": sum(len(r["input_ids"]) > 640 for r in selected),
        "full_length_1536": sum(len(r["input_ids"]) == 1536 for r in selected),
        "observed_answer_vocab": observed,
        "mapped_token_coverage": sum(
            count for key, count in counts.items() if t2d[key].item()
        )
        / answer_tokens,
        "mapping_sha256": file_sha(output / "vocab.pt"),
        "train_sha256": file_sha(output / "train.jsonl"),
        "family_manifest_sha256": file_sha(output / "families.jsonl"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
