"""Continue from a completed original-answer stage into immutable teacher-generated stages."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_state(path, **data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**data, "updated_unix": time.time()}, indent=2) + "\n")
    tmp.replace(path)


def verify_data(data, n, previous, work):
    key = f"train-first{n:04d}"
    manifest = json.loads((data / (key + ".manifest.json")).read_text())
    full = data / (key + ".jsonl")
    if (
        manifest["rows"] != n
        or manifest["answer_source"] != "target-only greedy SGLang output_ids"
        or manifest["temperature"] != 0
        or sha(full) != manifest["files"][full.name]
    ):
        raise ValueError("Teacher data provenance mismatch")
    raw = full.read_bytes().splitlines(keepends=True)
    if len(raw) != n:
        raise ValueError("Incorrect teacher row count")
    rows = list(map(json.loads, raw))
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate teacher ID")
        ids.add(row["id"])
        tok = row["input_ids"]
        mask = row["loss_mask"]
        n_prompt = mask.index(1)
        if not (
            len(tok) == len(mask) <= 1536
            and len(mask) - n_prompt >= 8
            and mask == ([0] * n_prompt + [1] * (len(mask) - n_prompt))
        ):
            raise ValueError("Bad teacher mask")
    if not previous:
        return full
    old = data / f"train-first{previous:04d}.jsonl"
    if old.read_bytes() != b"".join(raw[:previous]):
        raise ValueError("Published teacher prefix changed")
    piece = work / f"teacher-rows-{previous:04d}-{n:04d}.jsonl"
    text = b"".join(raw[previous:])
    if piece.exists() and piece.read_bytes() != text:
        raise ValueError("Delta fingerprint changed")
    if not piece.exists():
        tmp = piece.with_suffix(".tmp")
        tmp.write_bytes(text)
        tmp.replace(piece)
    return piece


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "repo", "target", "vocab"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--original", default="original8k-ddp4-b2-v1")
    p.add_argument("--original-step", type=int, default=1024)
    a = p.parse_args()
    train = a.root / "train"
    runs = train / "runs"
    data = a.root / "data"
    exports = train / "exports"
    work = train / "follow-on"
    work.mkdir(exist_ok=True)
    status = work / "state.json"
    lock = (work / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prev_id = a.original
    prev_step = a.original_step
    pos = 0
    try:
        while pos < 8192:
            prior = runs / prev_id
            outcome = prior / "outcome.json"
            while not outcome.is_file():
                write_state(status, phase="waiting-for-stage", run=prev_id)
                time.sleep(20)
            result = json.loads(outcome.read_text())
            ckpt = prior / f"{prev_id}-step{prev_step}"
            if (
                result["exit_code"]
                or not result["full_final_checkpoint"]
                or not (ckpt / "training_state.pt").is_file()
            ):
                raise RuntimeError(
                    f"Previous stage {prev_id} failed or final checkpoint is absent, inspect its logs"
                )
            while True:
                # Only complete, digest-sealed releases are eligible. Larger
                # prefixes take priority when multiple stages are already ready.
                stage_sizes = (8192, 2048, 512) if pos == 0 else (8192, 2048)
                ready = [
                    n
                    for n in stage_sizes
                    if n > pos
                    and (data / f"train-first{n:04d}.manifest.json").is_file()
                ]
                prior_attempts = [
                    n
                    for n in ready
                    if (
                        runs / f"teacher{pos:04d}-{n:04d}-ddp4-b2-v1" / "manifest.json"
                    ).exists()
                ]
                if len(prior_attempts) > 1:
                    raise RuntimeError(
                        "Conflicting prior target stages at the same data offset"
                    )
                if prior_attempts:
                    ready = prior_attempts
                if ready:
                    break
                write_state(
                    status,
                    phase="waiting-for-teacher-release",
                    prior=prev_id,
                    trained_teacher_rows=pos,
                )
                time.sleep(15)
            end = ready[0]
            selected = verify_data(data, end, pos, work)
            name = f"teacher{pos:04d}-{end:04d}-ddp4-b2-v1"
            steps = (end - pos) // 8
            run = runs / name
            previous_success = run / "outcome.json"
            if not previous_success.exists():
                if (run / "train.log").exists():
                    raise RuntimeError(
                        f"Unfinished stage {name}: preserve attempt state and investigate"
                    )
                export_steps = sorted(
                    set(i for i in [64, 256, 512, steps] if i <= steps)
                )
                script = train / "export_watch.py"
                with (work / (name + "-export.log")).open("a") as log:
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(script),
                            "--repo",
                            str(a.repo),
                            "--run",
                            str(run),
                            "--target",
                            str(a.target),
                            "--exports",
                            str(exports),
                            "--steps",
                            *map(str, export_steps),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                write_state(
                    status,
                    phase="training",
                    run=name,
                    previous_checkpoint=ckpt.name,
                    input_selection=[pos, end],
                    data_sha256=sha(selected),
                    vocab_sha256=sha(a.vocab),
                    optimizer="fresh Adam, weights only from previous stage",
                    lr=5e-5,
                    steps=steps,
                )
                cmd = [
                    sys.executable,
                    "-u",
                    str(train / "launch_stage.py"),
                    "--repo",
                    str(a.repo),
                    "--target",
                    str(a.target),
                    "--prompts",
                    str(selected),
                    "--vocab",
                    str(a.vocab),
                    "--base",
                    str(runs),
                    "--init",
                    str(ckpt),
                    "--run",
                    name,
                    "--data-kind",
                    "target-generated",
                    "--lr",
                    "0.00005",
                    "--warmup",
                    "0.02",
                    "--save-interval",
                    "64",
                    "--log-interval",
                    "4",
                ]
                with (work / (name + "-process.log")).open("a") as log:
                    subprocess.run(
                        cmd, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
            result = json.loads(previous_success.read_text())
            if result["exit_code"] or not result["full_final_checkpoint"]:
                raise RuntimeError(f"Target stage {name} failed")
            prev_id = name
            prev_step = steps
            pos = end
        write_state(
            status,
            phase="complete",
            run=prev_id,
            final_step=prev_step,
            unique_teacher_rows=pos,
        )
    except BaseException as exc:
        write_state(
            status,
            phase="stopped-on-error",
            prior=prev_id,
            trained_teacher_rows=pos,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
