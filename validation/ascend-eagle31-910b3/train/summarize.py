"""Summarize observed online runs and verify their capture and serving artifacts."""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from plot_loss import read_loss
from safetensors import safe_open


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(base, method, data):
    run_id = "qwen30b-910b3-" + method
    run = base / run_id
    logs = run / "train.log"
    metrics = read_loss(logs)
    refs = [
        json.loads(line)
        for line in (run / "control/refs.jsonl").read_text().splitlines()
    ]
    sizes = [row["num_tokens"] for row in refs]
    if len(refs) != 8 or len({x["sample_id"] for x in refs}) != 8:
        raise ValueError(f"{run_id}: expected eight unique captured samples")
    if Counter(sizes) != Counter(len(row["input_ids"]) for row in data):
        raise ValueError(
            f"{run_id}: captured token counts differ from the prepared inputs"
        )
    for ref, length in zip(refs, sizes):
        features = ref["feature_specs"]
        if features["hidden_states"]["shape"] != [1, length, 6144] or features[
            "target_last_hidden_states"
        ]["shape"] != [1, length, 2048]:
            raise ValueError(f"{run_id}: unexpected target feature widths")
    if not re.search(
        r"drive_producer returning produced=8 prompts_failed=0 pending=0 leased=0",
        logs.read_text(),
    ):
        raise ValueError(f"{run_id}: producer success marker missing")
    checkpoint = run / f"{run_id}-step8"
    export = run / "sglang-export"
    config = json.loads((export / "config.json").read_text())
    enabled = method == "eagle3.1"
    if config.get("fc_norm") != enabled or config.get("norm_output") != enabled:
        raise ValueError(f"{run_id}: unexpected export flags")
    with safe_open(export / "model.safetensors", framework="numpy") as weights:
        norm_keys = sorted(k for k in weights.keys() if k.startswith("fc_norm."))
        expected = [f"fc_norm.{i}.weight" for i in range(3)] if enabled else []
        if norm_keys != expected or "norm.weight" not in weights.keys():
            raise ValueError(f"{run_id}: missing or incorrect exported norm tensors")
    components = [
        checkpoint / "config.json",
        checkpoint / "training_state.pt",
        export / "config.json",
        export / "model.safetensors",
    ]
    paths = {
        str(p.relative_to(run)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in components
    }
    return {
        "method": method,
        "run_id": run_id,
        "checkpoint_id": checkpoint.name,
        "optimizer_updates": len(metrics),
        "loss_by_optimizer_update": metrics,
        "training_command_wall_seconds_including_initialization_and_checkpoint": int(
            (run / "wall-time-seconds.txt").read_text()
        ),
        "producer": {
            "captured": len(refs),
            "failed": 0,
            "pending": 0,
            "leased": 0,
            "capture_sequence_lengths": sizes,
            "auxiliary_hidden_width": 6144,
            "last_hidden_width": 2048,
        },
        "sglang_export": {
            "fc_norm": config["fc_norm"],
            "norm_output": config["norm_output"],
            "auxiliary_layers": config["eagle_config"][
                "eagle_aux_hidden_state_layer_ids"
            ],
            "per_feature_norm_tensors": norm_keys,
        },
        "artifacts": paths,
        "optimizer_and_rng_rank_0_bytes": (checkpoint / "training_state_rank0.pt")
        .stat()
        .st_size,
    }


def main():
    base, data_dir = map(Path, sys.argv[1:])
    data = [
        json.loads(line) for line in (data_dir / "train.jsonl").read_text().splitlines()
    ]
    evidence = {m: summarize(base, m, data) for m in ("eagle3", "eagle3.1")}
    (base / "train-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
