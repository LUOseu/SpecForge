"""Launch the two recipes sequentially through SpecForge on the same 910B3s."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, arm):
    source = Path(os.environ["SPECFORGE_REPO"]).resolve()
    tools = root / "tools"
    directory = root / arm
    directory.mkdir(exist_ok=False)
    feature_dir, holdout_dir = root / "features", root / "heldout/features"
    assert len(list(feature_dir.glob("*.ckpt"))) == 512
    assert len(list(holdout_dir.glob("*.ckpt"))) == 16
    version = "3" if arm == "eagle3" else "3.1"
    method = "EAGLE" + version
    config = {
        "model": {
            "target_model_path": os.environ["TARGET_MODEL"], "target_backend": "sglang",
            "torch_dtype": "bfloat16", "embedding_key": "model.embed_tokens.weight",
            "vocab_mapping_path": os.environ["VOCAB_MAP"],
        },
        "draft": str(source / f"specforge/config/draft/presets/qwen3-30B-A3B-eagle{version}.yaml"),
        "data": {"hidden_states_path": str(feature_dir), "eval_hidden_states_path": str(holdout_dir),
                 "max_length": 1536, "chat_template": "qwen", "dataloader_num_workers": 0,
                 "cache_dir": str(directory / "cache")},
        "training": {
            "num_epochs": 1, "max_steps": 100, "batch_size": 1, "accumulation_steps": 1,
            "loss": {"execution": {"chunk_size": 512}},
            "learning_rate": 1e-4, "lr_scheduler": "constant", "warmup_ratio": 0,
            "max_grad_norm": 0.5, "objective_options": {"ttt_length": 3, "depth_decay": 0.8},
            "attention_backend": "usp", "sp_ulysses_size": 2, "sp_ring_size": 2,
            "tp_size": 1, "fsdp_sharding": "NO_SHARD", "seed": 42, "dist_timeout": 5,
            "save_interval": 100, "log_interval": 1, "eval_interval": 25,
        },
        "deployment": {"mode": "local_colocated", "trainer": {"nproc_per_node": 4, "nnodes": 1}},
        "tracking": {"report_to": "none"}, "run_id": f"cp100-{arm}", "output_dir": str(directory / "train"),
    }
    config_file = directory / "train.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    env = {**os.environ, "PYTHONPATH": str(source)+os.pathsep+os.environ.get("PYTHONPATH", ""),
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "SPECFORGE_DEVICE": "npu",
           "PYTHONUNBUFFERED": "1", "ASCEND_RT_VISIBLE_DEVICES": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
           "HCCL_CONNECT_TIMEOUT": "120", "HCCL_EXEC_TIMEOUT": "120"}
    args = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
            str(tools / "train_with_trace.py"), str(config_file), method, "16"]
    start = time.time()
    with (directory / "train.log").open("x") as output:
        code = subprocess.run(args, cwd=source, env=env, stdout=output, stderr=subprocess.STDOUT).returncode
    outcome = {"returncode": code, "method": method, "start_unix": start, "end_unix": time.time(),
               "command": args, "config_sha256": digest(config_file)}
    (directory / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
    if code:
        raise RuntimeError(f"{arm} exited with status {code}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("arms", choices=["eagle3", "eagle31"], nargs="+")
    options = parser.parse_args()
    hashes = json.loads((options.root / "tools/source_manifest.json").read_text())
    source = Path(os.environ["SPECFORGE_REPO"]).resolve()
    mismatches = [name for name, want in hashes.items() if not (source / name).is_file() or digest(source / name) != want]
    if mismatches:
        raise ValueError(f"Source differs from c0725d44: {mismatches}")
    for arm in options.arms:
        run(options.root, arm)
