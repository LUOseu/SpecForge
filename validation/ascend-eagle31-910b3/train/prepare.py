"""Prepare the pinned ShareGPT smoke split and upstream EAGLE vocabulary map."""

import argparse
import hashlib
import json
import re
import runpy
import urllib.request
from collections import Counter
from pathlib import Path

import torch

from specforge.data.preprocessing import (
    preprocess_conversations,
    process_token_dict_to_mappings,
)
from specforge.data.template import TEMPLATE_REGISTRY
from specforge.utils import load_tokenizer

DATA_ID = "Aeala/ShareGPT_Vicuna_unfiltered"
DATA_REV = "8b0048ad6ae8c22f46a78c15559dec98feef5539"
DATA_FILE = "ShareGPT_V4.3_unfiltered_cleaned_split.json"
PREFIX_SIZE = 8 << 20
PREFIX_SHA = "ce09ad630e0d54d87a7df3d4cf66815d5184c1ddbea9827b4304ff7392641bbb"
TARGET_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TARGET_REV = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
MAX_LENGTH = 512
TOKENIZER_FILES = {
    "config.json": "a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows(local_prefix):
    if local_prefix:
        with local_prefix.open("rb") as stream:
            data = stream.read(PREFIX_SIZE)
    else:
        url = f"https://huggingface.co/datasets/{DATA_ID}/resolve/{DATA_REV}/{DATA_FILE}?bytes={PREFIX_SIZE}"
        request = urllib.request.Request(
            url, headers={"Range": f"bytes=0-{PREFIX_SIZE - 1}"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            data = response.read(PREFIX_SIZE)
    if hashlib.sha256(data).hexdigest() != PREFIX_SHA:
        raise ValueError("ShareGPT source prefix does not match the pinned SHA-256")
    # Only complete JSON objects in the verified prefix are candidates. A final
    # UTF-8 sequence or JSON object can legitimately straddle the byte cutoff.
    text = data.decode("utf-8", errors="ignore")
    pos = text.index("[") + 1
    decoder = json.JSONDecoder()
    skip = re.compile(r"[\s,]*")
    while pos < len(text):
        pos = skip.match(text, pos).end()
        try:
            row, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        yield row


def save_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument(
        "--model",
        required=True,
        help="Local pinned target snapshot or tokenizer directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-prefix",
        type=Path,
        help="Optional local copy of the pinned source prefix",
    )
    args = parser.parse_args()
    convert = runpy.run_path(str(args.repo / "scripts/prepare_data.py"))[
        "process_sharegpt_row"
    ]
    model_dir = Path(args.model)
    for name, expected in TOKENIZER_FILES.items():
        if sha256(model_dir / name) != expected:
            raise ValueError(
                f"{model_dir / name} differs from {TARGET_ID}@{TARGET_REV}"
            )
    tokenizer = load_tokenizer(args.model)
    template = TEMPLATE_REGISTRY.get("qwen")
    train, held_out, seen = [], [], set()
    counts = Counter()
    for row in source_rows(args.source_prefix):
        converted, _ = convert(row)
        identifier = converted["id"]
        parent_id = re.sub(r"_\d+$", "", identifier)
        pair = converted["conversations"][:2]
        if parent_id in seen or [m["role"] for m in pair] != ["user", "assistant"]:
            continue
        parsed = preprocess_conversations(
            tokenizer, [pair], template, max_length=MAX_LENGTH + 1
        )
        if not parsed["input_ids"]:
            continue
        ids = parsed["input_ids"][0].flatten().tolist()
        mask = parsed["loss_mask"][0].flatten().tolist()
        if len(ids) > MAX_LENGTH or sum(mask) < 8:
            continue
        seen.add(parent_id)
        if len(train) < 8:
            train.append({"id": identifier, "input_ids": ids, "loss_mask": mask})
            counts.update(token for token, supervised in zip(ids, mask) if supervised)
        else:
            held_out.append({"id": identifier, "messages": [pair[0]]})
        if len(held_out) == 8:
            break
    if len(held_out) != 8:
        raise ValueError("Verified source prefix did not provide the requested split")

    d2t, t2d = process_token_dict_to_mappings(counts, 32000, 151936)
    if d2t.shape != (32000,) or t2d.shape != (151936,) or t2d.sum().item() != 32000:
        raise ValueError("Unexpected EAGLE vocabulary shape or target membership")
    args.output.mkdir(parents=True, exist_ok=False)
    save_jsonl(args.output / "train.jsonl", train)
    save_jsonl(args.output / "held-out.jsonl", held_out)
    torch.save({"d2t": d2t, "t2d": t2d}, args.output / "vocab.pt")
    manifest = {
        "source": DATA_ID,
        "source_revision": DATA_REV,
        "file": DATA_FILE,
        "source_prefix_bytes": PREFIX_SIZE,
        "source_prefix_sha256": PREFIX_SHA,
        "target": TARGET_ID,
        "target_revision": TARGET_REV,
        "max_length": MAX_LENGTH,
        "template": "qwen",
        "train_ids": [r["id"] for r in train],
        "held_out_ids": [r["id"] for r in held_out],
        "train_samples": len(train),
        "train_input_tokens": sum(len(r["input_ids"]) for r in train),
        "train_supervised_tokens": sum(sum(r["loss_mask"]) for r in train),
        "tokenizer_json_sha256": sha256(model_dir / "tokenizer.json"),
        "sha256": {f.name: sha256(f) for f in sorted(args.output.iterdir())},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
