"""SGLang/Ascend serving helpers shared by the scale evaluator. Derived from the verified 910B3 smoke runner."""
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
PIN = "c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0"
SEED = 42
STEPS = 3


def check(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_runtime():
    spec = importlib.util.find_spec("sglang")
    if spec is None:
        raise RuntimeError("SGLang is missing from the serving interpreter")
    code = Path(spec.origin).resolve().parent
    dist = importlib.metadata.distribution("sglang")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    installed_dir = Path(dist.locate_file("sglang")).resolve()
    revision = direct.get("vcs_info", {}).get("commit_id") if installed_dir == code else None
    # Editable installs usually point directly to a checkout.
    for ancestor in code.parents:
        if (ancestor / ".git").exists() and (ancestor / "python/sglang").resolve() == code:
            revision = subprocess.check_output(["git", "-C", str(ancestor), "rev-parse", "HEAD"], text=True).strip()
            break
    # The pinned source tar is also used in offline containers and carries no .git directory.
    # Compare the whole SGLang runtime Python tree in that case.
    source_proof = None
    if revision is None:
        tree = hashlib.sha256()
        for path in sorted((code / "srt").rglob("*.py"), key=lambda p: p.relative_to(code).as_posix()):
            tree.update(f"{path.relative_to(code).as_posix()} {file_sha(path)}\n".encode())
        source_proof = tree.hexdigest()
        check(source_proof == "b8bcb479e0dadadb1b997e79a39bd558dfe5e95e0e757ee50a482236382fe6d8",
              f"unidentified installed SGLang source; srt Python tree digest {source_proof}")
    else:
        check(revision == PIN, f"runtime revision {revision!r}; expected {PIN}")
    critical = {
        "srt/models/llama_eagle3.py": "b25e5f839b088ddf50138bce4a3c4569acc31f8203727b9a7e2a53c337287dac",
        "srt/speculative/spec_decoding_metrics.py": "693489e43f995486ca92b00426bdfcb10753b95f0a296ca0dd5e2ac578073fab",
    }
    for name, expected in critical.items():
        check(file_sha(code / name) == expected, f"installed SGLang file differs from pinned source: {name}")
    return {"vcs_revision": revision, "expected_revision": PIN, "srt_tree_sha256": source_proof,
            "version": dist.version, "module": str(code), "source_sha256": critical}


def launch_args(target, draft, tp, port, fraction, context, deterministic=False):
    args = [sys.executable, "-m", "sglang.launch_server", "--model-path", str(target),
            "--tokenizer-path", str(target), "--host", "127.0.0.1", "--port", str(port),
            "--device", "npu", "--attention-backend", "ascend", "--tp-size", str(tp),
            "--dtype", "bfloat16", "--random-seed", str(SEED),
            "--mem-fraction-static", str(fraction), "--context-length", str(context),
            "--max-running-requests", "4", "--chunked-prefill-size", "-1", "--disable-radix-cache"]
    if deterministic:
        args.append("--enable-deterministic-inference")
    if draft:
        args.extend(["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", str(draft),
                     "--speculative-num-steps", str(STEPS), "--speculative-eagle-topk", "1",
                     "--speculative-num-draft-tokens", str(STEPS + 1)])
    return args


def check_server(info, target, draft, tp, fraction, context, deterministic=None):
    expected = {"model_path": str(target), "tokenizer_path": str(target), "device": "npu",
                "attention_backend": "ascend", "dtype": "bfloat16", "tp_size": tp,
                "random_seed": SEED, "context_length": context, "mem_fraction_static": fraction,
                "disable_radix_cache": True, "max_running_requests": 4,
                "speculative_algorithm": "EAGLE3" if draft else None,
                "speculative_draft_model_path": str(draft) if draft else None}
    if deterministic is not None:
        expected["enable_deterministic_inference"] = deterministic
    if draft:
        expected.update(speculative_num_steps=STEPS, speculative_eagle_topk=1,
                        speculative_num_draft_tokens=STEPS + 1)
    for field, value in expected.items():
        check(field in info and info[field] == value, f"/server_info: {field}={info.get(field)!r}, expected {value!r}")
    extra = ("version", "disable_overlap_schedule", "cuda_graph_backend_decode",
             "disable_decode_cuda_graph", "prefill_attention_backend", "decode_attention_backend",
             "quantization", "speculative_draft_model_quantization", "load_format", "dp_size", "ep_size",
             "chunked_prefill_size", "enable_deterministic_inference")
    return {k: info[k] for k in [*expected, *extra] if k in info}


@contextlib.contextmanager
def server(cmd, port, log_path, session):
    # Refuse an occupied endpoint before spawning. Only ever stop our own child group.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    with log_path.open("wb") as logs:
        proc = subprocess.Popen(cmd, stdout=logs, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"SGLang exited with {proc.returncode}; see {log_path}")
                try:
                    res = session.get(f"http://127.0.0.1:{port}/server_info", timeout=10)
                    if res.ok:
                        yield_info = res.json()
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(f"SGLang startup timed out; see {log_path}")
            yield yield_info
        finally:
            # A failed launcher may leave worker processes behind in the same group.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pass
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def resolve_devices(requested, mapping_file):
    if mapping_file:
        listing = mapping_file.read_text()
    else:
        listing = subprocess.check_output(["npu-smi", "info", "-m"], text=True, timeout=30)
    devices = {}
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0].isdigit() and fields[2].isdigit() and "910b3" in "".join(fields[3:]).lower():
            check(fields[0] not in devices, f"ambiguous device mapping: {line}")
            devices[fields[0]] = fields[2]
    check(all(x in devices for x in requested), f"requested physical 910B3 IDs {requested} absent from npu-smi info -m")
    return ",".join(devices[x] for x in requested), listing
