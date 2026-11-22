"""Independently read optimizer checkpoint, sample order and rank-local traces."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit(root, name):
    arm = root / name
    outcome = json.loads((arm / "outcome.json").read_text())
    assert outcome["returncode"] == 0
    rows = read_rows(arm / "steps.csv")
    baseline = [(r["update"], r["sample_indices"], r["loss"], r["grad_norm"], r["adam_step"]) for r in rows]
    assert len(rows) == 100
    assert all(int(r["update"]) == int(r["adam_step"]) == i for i, r in enumerate(rows, 1))
    assert all(math.isfinite(float(r["loss"])) and 0 < float(r["grad_norm"]) < math.inf and float(r["lr_used"]) == 1e-4 for r in rows)
    for rank in (1, 2, 3):
        local = read_rows(arm / f"rank{rank}.csv")
        assert [(r["update"], r["sample_indices"], r["loss"], r["grad_norm"], r["adam_step"]) for r in local] == baseline
    ids = [int(s) for r in rows for s in r["sample_indices"].split()]
    assert len(set(ids)) == len(ids) == 400
    assert ids == torch.randperm(512, generator=torch.Generator().manual_seed(42)).tolist()[:400]
    holdout = read_rows(arm / "validation.csv")
    assert [int(r["update"]) for r in holdout] == [0, 25, 50, 75, 100]
    assert all(int(r["n_samples"]) == 16 and math.isfinite(float(r["loss"])) for r in holdout)
    ck = arm / "train" / f"cp100-{name}-step100"
    state = torch.load(ck / "training_state.pt", mmap=True, map_location="cpu", weights_only=True)
    assert state["global_step"] == 100 and state["world_size"] == state["accumulation_steps"] == 4
    assert state["epoch_samples"] == state["epoch_batch"] == 400
    optimizer = state["replicated_optimizer_state"]
    moments = optimizer["optimizer_state_dict"]["state"].values()
    steps = [int(s["step"]) for s in moments]
    assert steps and set(steps) == {100}
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in optimizer["fp32_params"])
    assert all(torch.isfinite(s[k]).all() for s in moments for k in ("exp_avg", "exp_avg_sq"))
    assert all(float(s["exp_avg_sq"].max()) > 0 for s in moments)
    trainable = json.loads((arm / "init_parameters.json").read_text())
    changed = []
    for full_name, p in trainable.items():
        if not p["trainable"]:
            continue
        key = full_name.split("draft_model.", 1)[1]
        final = state["draft_state_dict"][key].detach().contiguous().view(torch.uint8).numpy()
        assert hashlib.sha256(memoryview(final)).hexdigest() != p["sha256"]
        changed.append(key)
    assert len(changed) == len(steps)
    for i in range(4):
        shard = torch.load(ck / f"training_state_rank{i}.pt", mmap=True, map_location="cpu", weights_only=True)
        assert shard["rng"]["device_type"] == "npu" and shard["rng"]["npu"].numel() > 0
    summary = dict(method=outcome["method"], checkpoint=str(ck), global_step=state["global_step"],
                   microbatches=state["epoch_batch"], unique_training_rows=len(set(ids)),
                   actual_order_verified_against_seed42=True, rank_traces_identical=True,
                   sample_order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                   checkpoint_adam_parameter_step_range=[min(steps), max(steps)],
                   all_adam_moments_and_fp32_masters_finite=True,
                   final_parameters_changed_from_initial=len(changed), npu_rng_ranks=4,
                   heldout_evaluations=holdout, scheduler=optimizer["lr_scheduler_type"],
                   training=true_metrics(rows))
    (arm / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")


def true_metrics(rows):
    losses = [float(row["loss"]) for row in rows]
    grads = [float(row["grad_norm"]) for row in rows]
    return dict(first_10_mean_loss=math.fsum(losses[:10]) / 10,
                last_10_mean_loss=math.fsum(losses[-10:]) / 10,
                gradient_norm_range=[min(grads), max(grads)])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("arms", nargs="+", choices=["eagle3", "eagle31"])
    args = p.parse_args()
    torch.set_num_threads(3)
    for arm in args.arms:
        audit(args.root, arm)
