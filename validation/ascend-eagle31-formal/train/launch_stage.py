"""Launch a finite, reproducible online EAGLE3.1 stage with the upstream CLI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def sha(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("repo", "target", "prompts", "vocab", "base"):
        p.add_argument("--" + opt, required=True, type=Path)
    p.add_argument("--run", required=True)
    p.add_argument(
        "--data-kind", required=True, choices=["original-answers", "target-generated"]
    )
    p.add_argument("--init", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--save-interval", type=int, default=64)
    p.add_argument("--log-interval", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--length", type=int, default=1536)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--master", default="127.0.0.1:35551")
    p.add_argument("--metadata", default="http://127.0.0.1:35880/metadata")
    p.add_argument(
        "--sharding",
        default="NO_SHARD",
        choices=["NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD"],
    )
    p.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--trim", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    out = a.base.resolve() / a.run
    out.mkdir(parents=True, exist_ok=True)
    if not a.resume and (out / "control").exists():
        raise SystemExit(
            "Use a new run ID for a fresh attempt, or --resume an eligible consumer"
        )
    if a.init and a.resume:
        raise SystemExit("Choose either --init or --resume")
    with a.prompts.open("rb") as fp:
        total = sum(1 for l in fp if l.strip())
    total = min(total, a.max_samples or total)
    natural = (total * a.epochs) // (a.world * a.batch)
    steps = min(a.max_steps or natural, natural)
    if steps < 1:
        raise SystemExit("Insufficient rows to train a complete optimizer window")
    config = {
        "model": {
            "target_model_path": str(a.target.resolve()),
            "vocab_mapping_path": str(a.vocab.resolve()),
            "target_backend": "sglang",
            "embedding_key": "model.embed_tokens.weight",
            "torch_dtype": "bfloat16",
        },
        "draft": str(
            a.repo.resolve()
            / "specforge/config/draft/presets/qwen3-30B-A3B-eagle3.1.yaml"
        ),
        "data": {
            "prompts_path": str(a.prompts.resolve()),
            "max_length": a.length,
            "chat_template": "qwen",
            "build_dataset_num_proc": 1,
            "dataloader_num_workers": 0,
            "cache_dir": str(out / "cache"),
        },
        "training": {
            "num_epochs": a.epochs,
            "max_steps": steps,
            "batch_size": a.batch,
            "accumulation_steps": 1,
            "learning_rate": a.lr,
            "warmup_ratio": a.warmup,
            "lr_scheduler": a.schedule,
            "max_grad_norm": 0.5,
            "objective_options": {"ttt_length": 7},
            "attention_backend": "sdpa",
            "tp_size": 1,
            "fsdp_sharding": a.sharding,
            "save_interval": min(a.save_interval, steps),
            "max_checkpoints": 4,
            "log_interval": a.log_interval,
            "dist_timeout": 30,
            "seed": a.seed,
        },
        "runtime": {
            "producer_lease": 2,
            "producer_concurrency": 1,
            "in_flight_high_watermark": max(32, 4 * a.world * a.batch),
            "in_flight_low_watermark": max(16, 2 * a.world * a.batch),
        },
        "run_id": a.run,
        "output_dir": str(out),
        "tracking": {"report_to": "none"},
        "deployment": {
            "mode": "disaggregated",
            "trainer": {"nnodes": 1, "nproc_per_node": a.world},
            "disaggregated": {
                "control_dir": str(out / "control"),
                "consumer_state_dir": str(out / "consumer"),
                "backend": "mooncake",
                "server_urls": [a.url] * a.workers,
                "mooncake_metadata_server": a.metadata,
                "mooncake_master_server_addr": a.master,
                "mooncake_protocol": "tcp",
            },
        },
    }
    if a.max_samples:
        config["data"]["max_prompts"] = a.max_samples
    if a.init:
        config["initialization"] = {"checkpoint": str(a.init.resolve())}
    if a.trim:
        config["training"]["loss"] = {"execution": {"trim": True}}
    if a.resume:
        config["training"]["resume_from"] = str(a.resume.resolve())
    filename = out / ("resume.yaml" if a.resume else "run.yaml")
    filename.write_text(yaml.safe_dump(config, sort_keys=False))
    args = [sys.executable, "-m", "specforge.cli", "train", "-c", str(filename)]
    artifact = {
        "run_id": a.run,
        "data_kind": a.data_kind,
        "source_samples_in_input": total,
        "optimizer_steps_planned": steps,
        "global_batch": a.world * a.batch,
        "init_checkpoint": str(a.init.resolve()) if a.init else None,
        "ordinary_optimizer_state": "fresh for stage" if not a.resume else "restored",
        "resume_checkpoint": str(a.resume.resolve()) if a.resume else None,
        "prompt_sha256": sha(a.prompts),
        "vocab_sha256": sha(a.vocab),
        "runtime_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "training_config_sha256": sha(filename),
    }
    journal = out.parent.parent / "campaign/chain.json"
    if journal.is_file():
        for entry in json.loads(journal.read_text()):
            if entry["run"] == a.run and (
                entry["data_sha256"] != artifact["prompt_sha256"]
                or entry["vocab_sha256"] != artifact["vocab_sha256"]
            ):
                raise ValueError(
                    "Input or vocab differs from the persisted campaign plan"
                )
    (out / ("resume-manifest.json" if a.resume else "manifest.json")).write_text(
        json.dumps(artifact, indent=2) + "\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(a.repo.resolve())
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
        "SPECFORGE_DEVICE": "npu",
        "MOONCAKE_PROTOCOL": "tcp",
        "MOONCAKE_LOCAL_HOSTNAME": "127.0.0.1",
        "MOONCAKE_METADATA_SERVER": a.metadata,
        "MOONCAKE_MASTER_SERVER_ADDR": a.master,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HCCL_CONNECT_TIMEOUT": "600",
        "HCCL_EXEC_TIMEOUT": "600",
    }
    with (out / "plan.json").open("w") as fp:
        subprocess.run(args + ["--plan"], check=True, cwd=a.repo, env=env, stdout=fp)
    if a.plan_only:
        return
    (out / "launcher.pid").write_text(str(os.getpid()) + "\n")
    start = time.time()
    log = out / ("resume.log" if a.resume else "train.log")
    with log.open("x") as fp:
        rc = subprocess.run(
            args, cwd=a.repo, env=env, stdout=fp, stderr=subprocess.STDOUT
        ).returncode
    ckpt = out / f"{a.run}-step{steps}"
    healthy = (ckpt / "training_state.pt").is_file() and all(
        (ckpt / f"training_state_rank{i}.pt").is_file() for i in range(a.world)
    )
    outcome_tmp = out / "outcome.json.tmp"
    outcome_tmp.write_text(
        json.dumps(
            {
                "exit_code": rc,
                "wall_seconds": time.time() - start,
                "full_final_checkpoint": healthy,
                "final_checkpoint": ckpt.name,
            },
            indent=2,
        )
        + "\n"
    )
    outcome_tmp.replace(out / "outcome.json")
    if rc or not healthy:
        raise SystemExit(rc or 1)


if __name__ == "__main__":
    main()
