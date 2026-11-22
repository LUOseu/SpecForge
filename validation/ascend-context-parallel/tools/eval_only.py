"""Evaluate a training recipe once at the restored checkpoint or fresh seed."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml


def worker(config, reference_csv, expected_update):
    import torch.distributed as dist
    import specforge.eval.evaluator as implementation
    from specforge.cli import main
    from specforge.training.controller import TrainerController

    def evaluate_only(controller, data):
        assert controller.global_step == expected_update
        actual = controller.evaluate_configured()
        with reference_csv.open() as stream:
            [reference] = [float(r["loss"]) for r in csv.DictReader(stream)
                           if int(r["update"]) == expected_update]
        delta = actual["eval/avg_loss"] - reference
        if dist.get_rank() == 0:
            result = dict(global_step=controller.global_step, additional_optimizer_steps=0,
                          reference_holdout_loss=reference, native_minus_reference=delta,
                          evaluator_sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
                          world_size=dist.get_world_size(), metrics=actual)
            config.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        assert math.isclose(delta, 0, abs_tol=1e-5), (actual["eval/avg_loss"], reference)
        return controller.global_step

    TrainerController.fit = evaluate_only
    return main(["train", "-c", str(config)])


def replay(root, source, arm, update):
    cfg = yaml.safe_load((root / arm / "train.yaml").read_text())
    target = root / "native-validation"
    target.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(target / f"{arm}-step{update}-outputs")
    cfg["data"]["cache_dir"] = str(target / f"{arm}-step{update}-cache")
    cfg["draft"] = str(source / "specforge/config/draft/presets" / Path(cfg["draft"]).name)
    if update:
        cfg["resume"] = str(root / arm / "train" / f"cp100-{arm}-step{update}")
    config_path = target / f"{arm}-step{update}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = {**os.environ, "PYTHONPATH": str(source)+os.pathsep+os.environ.get("PYTHONPATH", ""),
           "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3"), "SPECFORGE_DEVICE": "npu", "OMP_NUM_THREADS": "2",
           "HCCL_CONNECT_TIMEOUT": "120", "HCCL_EXEC_TIMEOUT": "120", "PYTHONUNBUFFERED": "1"}
    with config_path.with_suffix(".log").open("w") as output:
        return subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                               "--nproc-per-node=4", str(Path(__file__).resolve()), "worker", str(config_path),
                               str(root / arm / "validation.csv"), str(update)],
                              env=env, cwd=source, stdout=output, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        sys.exit(worker(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])))
    root = Path(sys.argv[1])
    source = Path(sys.argv[2])
    for update in map(int, sys.argv[3:]):
        for arm in ("eagle3", "eagle31"):
            code = replay(root, source, arm, update)
            if code:
                sys.exit(code)
