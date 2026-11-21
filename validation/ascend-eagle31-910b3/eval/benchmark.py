#!/usr/bin/env python3
"""Measure target-only, EAGLE3 and EAGLE3.1 on the same local Ascend deployment."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

PIN = "c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0"
DATA_REV = "3101c7d5072418e28b9008a6636bde82a006892c"
DATA_URL = f"https://raw.githubusercontent.com/openai/grade-school-math/{DATA_REV}/grade_school_math/data/test.jsonl"
DATA_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
ARMS = ("target", "eagle3", "eagle31")
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


def ordered_questions(raw, count):
    check(hashlib.sha256(raw).hexdigest() == DATA_SHA, "GSM8K source SHA256 mismatch")
    data = [json.loads(line)["question"] for line in raw.splitlines()]
    check(1 <= count <= len(data) - 1, "select 1..1318 questions; one more is reserved for warmup")
    indices = sorted(range(len(data)), key=lambda i: hashlib.sha256(f"{SEED}:{i}".encode()).digest())
    return [(i, data[i]) for i in indices[:count + 1]]


def workload(args, target):
    if args.held_out:
        raw = args.held_out.read_bytes()
        manifest = json.loads(args.held_out.with_name("manifest.json").read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        check(digest == manifest["sha256"][args.held_out.name], "held-out file checksum mismatch")
        check(file_sha(target / "tokenizer.json") == manifest["tokenizer_json_sha256"],
              "held-out data and serving tokenizer do not match")
        rows = [json.loads(line) for line in raw.splitlines()]
        ids = [row["id"] for row in rows]
        check(ids and ids == manifest["held_out_ids"] and len(set(ids)) == len(ids)
              and set(ids).isdisjoint(manifest["train_ids"]), "wrong or overlapping held-out IDs")
        check(args.count is None or 1 <= args.count <= len(rows), "held-out sample count exceeds split")
        source = {"dataset": manifest["source"], "revision": manifest["source_revision"],
                  "file": args.held_out.name, "sha256": digest}
        messages = [("warmup", [{"role": "user", "content": "Explain what a prime number is."}])]
        messages.extend((r["id"], r["messages"]) for r in rows[:args.count])
    else:
        raw = args.gsm8k.read_bytes() if args.gsm8k else urlopen(DATA_URL, timeout=90).read()
        source = {"dataset": "GSM8K test", "source": DATA_URL, "sha256": DATA_SHA}
        messages = [(f"gsm8k/test/{index}", [{"role": "user", "content": question}])
                    for index, question in ordered_questions(raw, 64 if args.count is None else args.count)]
    return source, tokenize(target, messages)


def tokenize(target, messages):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    rows = []
    for identifier, turns in messages:
        ids = tokenizer.apply_chat_template(
            turns, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False,
        )
        check(isinstance(ids, list) and ids and type(ids[0]) is int, "unexpected chat template result")
        rows.append({"id": identifier, "input_ids": ids})
    return rows


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


def artifacts(path, *, weights):
    masks = ["*.json", "*.jinja", "*.model", "merges.txt", "vocab.txt"]
    if weights:
        masks.append("*.safetensors")
    return {p.name: file_sha(p) for mask in masks for p in sorted(path.glob(mask)) if p.is_file()}


def draft_configs(paths):
    from safetensors import safe_open

    configs = []
    mappings = []
    for arm, path in paths.items():
        cfg = json.loads((path / "config.json").read_bytes())
        metadata = cfg["specforge_artifact"]
        check(metadata["format"] == "sglang" and metadata["sglang_commit"] == PIN,
              f"{path} must be a serving export for {PIN}")
        expected = arm == "eagle31"
        check(cfg["architectures"] == ["LlamaForCausalLMEagle3"] and cfg["fc_norm"] is expected
              and cfg["norm_output"] is expected, f"{arm}: wrong draft architecture or normalization flags")
        map_hashes = {}
        learned_norms = set()
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(shard, framework="numpy") as weights:
                for name in weights.keys():
                    if name in ("d2t", "t2d"):
                        check(name not in map_hashes, f"duplicate {name} in {path}")
                        values = weights.get_tensor(name)
                        map_hashes[name] = (values.dtype.str, values.shape,
                                            hashlib.sha256(values.tobytes()).hexdigest())
                    elif name.startswith("fc_norm."):
                        learned_norms.add(name)
        check(set(map_hashes) == {"d2t", "t2d"}, f"missing exported vocabulary buffers at {path}")
        expected_norms = {f"fc_norm.{i}.weight" for i in range(cfg["num_aux_hidden_states"])} if expected else set()
        check(learned_norms == expected_norms, f"{arm}: wrong learned feature-normalization weights")
        mappings.append(map_hashes)
        configs.append(cfg)
    # Other config fields (including layer taps, all width fields and vocab size) must match.
    ignored = {"fc_norm", "norm_output", "_name_or_path"}
    check({k: v for k, v in configs[0].items() if k not in ignored} ==
          {k: v for k, v in configs[1].items() if k not in ignored},
          "the two exported draft configs differ beyond the EAGLE3.1 switches")
    check(mappings[0] == mappings[1], "the two drafts have different exported vocabularies")


def launch_args(target, draft, tp, port, fraction, context):
    args = [sys.executable, "-m", "sglang.launch_server", "--model-path", str(target),
            "--tokenizer-path", str(target), "--host", "127.0.0.1", "--port", str(port),
            "--device", "npu", "--attention-backend", "ascend", "--tp-size", str(tp),
            "--dtype", "bfloat16", "--random-seed", str(SEED),
            "--mem-fraction-static", str(fraction), "--context-length", str(context),
            "--max-running-requests", "1", "--chunked-prefill-size", "-1", "--disable-radix-cache"]
    if draft:
        args.extend(["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", str(draft),
                     "--speculative-num-steps", str(STEPS), "--speculative-eagle-topk", "1",
                     "--speculative-num-draft-tokens", str(STEPS + 1)])
    return args


def check_server(info, target, draft, tp, fraction, context):
    expected = {"model_path": str(target), "tokenizer_path": str(target), "device": "npu",
                "attention_backend": "ascend", "dtype": "bfloat16", "tp_size": tp,
                "random_seed": SEED, "context_length": context, "mem_fraction_static": fraction,
                "disable_radix_cache": True, "max_running_requests": 1,
                "speculative_algorithm": "EAGLE3" if draft else None,
                "speculative_draft_model_path": str(draft) if draft else None}
    if draft:
        expected.update(speculative_num_steps=STEPS, speculative_eagle_topk=1,
                        speculative_num_draft_tokens=STEPS + 1)
    for field, value in expected.items():
        check(field in info and info[field] == value, f"/server_info: {field}={info.get(field)!r}, expected {value!r}")
    extra = ("version", "disable_overlap_schedule", "cuda_graph_backend_decode",
             "disable_decode_cuda_graph", "prefill_attention_backend", "decode_attention_backend",
             "quantization", "speculative_draft_model_quantization", "load_format", "dp_size", "ep_size",
             "chunked_prefill_size")
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


def parse_result(row, response, elapsed, speculative, limit):
    meta = response["meta_info"]
    tokens = meta["completion_tokens"]
    ids = response["output_ids"]
    check(meta["finish_reason"]["type"] in ("stop", "length"), f"{row['id']} failed: {meta.get('finish_reason')}")
    check(type(tokens) is int and 0 < tokens <= limit and len(ids) == tokens,
          f"{row['id']} invalid completion token count")
    check(isinstance(ids, list) and all(type(token) is int for token in ids), "bad output token IDs")
    check(meta["prompt_tokens"] == len(row["input_ids"]), f"{row['id']} prompt token count mismatch")
    result = {"id": row["id"], "output_ids": ids, "prompt_tokens": meta["prompt_tokens"],
              "completion_tokens": tokens, "seconds": elapsed, "finish_reason": meta["finish_reason"]}
    if not speculative:
        check(not meta.get("spec_verify_ct"), "target-only server reported speculative verification")
        return result
    n, good, proposed = (meta[k] for k in ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts"))
    check(all(type(x) is int for x in (n, good, proposed)) and n > 0 and
          proposed == n * STEPS and 0 <= good <= proposed, "missing or inconsistent speculative counters")
    check(math.isclose(meta["spec_accept_length"], tokens / n) and
          math.isclose(meta["spec_accept_rate"], good / proposed), "backend metric definition mismatch")
    result.update({key: meta[key] for key in ("spec_verify_ct", "spec_num_correct_drafts",
                  "spec_num_proposed_drafts", "spec_accept_length", "spec_accept_rate")})
    return result


def summarize(results, elapsed, speculative):
    tokens = sum(r["completion_tokens"] for r in results)
    stats = {"requests": len(results), "output_tokens": tokens, "measurement_seconds": elapsed,
             "wall_output_tok_s": tokens / elapsed, "median_latency_s": statistics.median(r["seconds"] for r in results),
             "verify_rounds": None, "pooled_accept_length": None, "pooled_draft_accept_rate": None}
    if speculative:
        rounds = sum(r["spec_verify_ct"] for r in results)
        accepted = sum(r["spec_num_correct_drafts"] for r in results)
        proposed = sum(r["spec_num_proposed_drafts"] for r in results)
        stats.update(verify_rounds=rounds, pooled_accept_length=tokens / rounds,
                     accepted_draft_tokens=accepted, proposed_draft_tokens=proposed,
                     pooled_draft_accept_rate=accepted / proposed,
                     mean_request_accept_length=statistics.fmean(r["spec_accept_length"] for r in results))
    return stats


def compare(runs):
    target_rows, target_stats = runs["target"]
    report = {"arms": {arm: stats for arm, (_, stats) in runs.items()}, "output_mismatches": {},
              "speedup_vs_target": None, "eagle31_over_eagle3_throughput": None,
              "eagle31_minus_eagle3_accept_length": None}
    for arm in ARMS[1:]:
        rows, _ = runs[arm]
        check([r["id"] for r in rows] == [r["id"] for r in target_rows], "mismatched requests")
        report["output_mismatches"][arm] = [r["id"] for r, base in zip(rows, target_rows) if r["output_ids"] != base["output_ids"]]
    valid = {k: runs[k][1]["wall_output_tok_s"] / target_stats["wall_output_tok_s"]
             for k in ARMS[1:] if not report["output_mismatches"][k]}
    report["speedup_vs_target"] = valid or None
    report["eagle31_vs_eagle3_mismatches"] = [
        a["id"] for a, b in zip(runs["eagle3"][0], runs["eagle31"][0]) if a["output_ids"] != b["output_ids"]
    ]
    if not report["eagle31_vs_eagle3_mismatches"]:
        report["eagle31_over_eagle3_throughput"] = runs["eagle31"][1]["wall_output_tok_s"] / runs["eagle3"][1]["wall_output_tok_s"]
        report["eagle31_minus_eagle3_accept_length"] = runs["eagle31"][1]["pooled_accept_length"] - runs["eagle3"][1]["pooled_accept_length"]
    return report


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


def run(args):
    import requests

    check(args.devices and all(x.isdigit() for x in args.devices.split(",")) and
          len(set(args.devices.split(","))) == len(args.devices.split(",")), "pass unique physical device IDs")
    tp = len(args.devices.split(","))
    check(0 < args.fraction < 1 and args.max_new_tokens >= 2, "invalid memory fraction or generation cap")
    check(float(os.environ.get("SGLANG_SIMULATE_ACC_LEN", "-1")) <= 0, "disable simulated speculative acceptance")
    visible, device_map = resolve_devices(args.devices.split(","), args.device_map)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible
    os.environ["SPECFORGE_SPEC_CAPTURE"] = "0"
    target = args.target.resolve(strict=True)
    drafts = {}
    if not args.target_only:
        check(args.eagle3 is not None and args.eagle31 is not None, "matched run requires both exported drafts")
        drafts = {"eagle3": args.eagle3.resolve(strict=True), "eagle31": args.eagle31.resolve(strict=True)}
        draft_configs(drafts)
    runtime = installed_runtime()
    source, prompt_rows = workload(args, target)
    check(max(len(r["input_ids"]) for r in prompt_rows) + args.max_new_tokens <= args.context, "insufficient context length")
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    inputs = {"dataset": source, "seed": SEED, "warmup": prompt_rows[0], "measured": prompt_rows[1:]}
    if args.resume:
        previous = json.loads((out / "manifest.json").read_bytes())
        check(json.loads((out / "prompts.json").read_bytes()) == inputs, "resume prompt mismatch")
        check((out / "npu-device-map.txt").read_text() == device_map, "resume device map mismatch")
    else:
        out.mkdir(exist_ok=False)
        (out / "npu-device-map.txt").write_text(device_map)
        save(out / "prompts.json", inputs)
    sampling = {"temperature": 0, "max_new_tokens": args.max_new_tokens}
    probe_code = ("import json,torch,torch_npu; print(json.dumps([torch.npu.get_device_name(i) "
                  "for i in range(torch.npu.device_count())]))")
    detected = json.loads(subprocess.check_output([sys.executable, "-c", probe_code], text=True).splitlines()[-1])
    check(len(detected) == tp and all("910b3" in x.lower().replace(" ", "") for x in detected),
          f"expected {tp} visible Ascend 910B3 processors, torch_npu reported {detected}")
    if shutil.which("npu-smi"):
        inventory = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
        (out / "npu-smi.txt").write_text(inventory.stdout + inventory.stderr)
    versions = {}
    for name in ("torch", "torch-npu", "transformers", "sglang", "sgl-kernel-npu", "triton-ascend", "specforge"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {"mode": "target-only diagnostic" if args.target_only else "matched three-way",
                "runtime": runtime, "packages": versions, "host": platform.node(), "platform": platform.platform(),
                "python": sys.version, "physical_devices": args.devices, "runtime_visible_devices": visible,
                "npu_names": detected, "tp": tp, "concurrency": 1,
                "seed": SEED, "sampling": sampling, "prompts_sha256": file_sha(out / "prompts.json"),
                "target": str(target), "target_files_sha256": artifacts(target, weights=True),
                "drafts": {arm: {"path": str(p), "files_sha256": artifacts(p, weights=True)} for arm, p in drafts.items()},
                "env": {k: os.environ[k] for k in ("ASCEND_RT_VISIBLE_DEVICES", "HCCL_CONNECT_TIMEOUT",
                        "HCCL_EXEC_TIMEOUT", "PYTORCH_NPU_ALLOC_CONF", "SGLANG_SIMULATE_ACC_LEN",
                        "SPECFORGE_SPEC_CAPTURE") if k in os.environ},
                "commands": {}}
    if previous is not None:
        check({k: v for k, v in previous.items() if k != "commands"} ==
              {k: v for k, v in manifest.items() if k != "commands"}, "resume run identity mismatch")
        manifest["commands"] = previous["commands"]
    save(out / "manifest.json", manifest)
    runs = {}
    for arm in (("target",) if args.target_only else ARMS):
        draft = drafts.get(arm)
        cmd = launch_args(target, draft, tp, args.port, args.fraction, args.context)
        if arm in manifest["commands"]:
            check(manifest["commands"][arm] == cmd, f"resume command mismatch for {arm}")
        if previous is not None and (out / f"{arm}.json").exists():
            saved = json.loads((out / f"{arm}.json").read_bytes())
            outputs, stats = saved["requests"], saved["summary"]
            check([r["id"] for r in outputs] == [r["id"] for r in prompt_rows[1:]], f"resume IDs differ: {arm}")
            check(summarize(outputs, stats["measurement_seconds"], bool(draft)) == stats,
                  f"resume aggregate mismatch: {arm}")
            check_server(json.loads((out / f"{arm}.server.json").read_bytes()),
                         target, draft, tp, args.fraction, args.context)
            runs[arm] = outputs, stats
            continue
        manifest["commands"][arm] = cmd
        save(out / "manifest.json", manifest)
        log = out / f"{arm}.log"
        if log.exists():
            log = out / f"{arm}.retry-{time.time_ns()}.log"
        with requests.Session() as session:
            session.trust_env = False
            with server(cmd, args.port, log, session) as info:
                save(out / f"{arm}.server.json", check_server(info, target, draft, tp, args.fraction, args.context))
                outputs = []
                for i, row in enumerate(prompt_rows):
                    if i == 1:
                        start = time.perf_counter()
                    start_req = time.perf_counter()
                    res = session.post(f"http://127.0.0.1:{args.port}/generate",
                                       json={"input_ids": row["input_ids"], "sampling_params": sampling, "stream": False},
                                       timeout=(15, 600))
                    res.raise_for_status()
                    stop_req = time.perf_counter()
                    response = res.json()
                    try:
                        result = parse_result(row, response, stop_req - start_req, bool(draft), args.max_new_tokens)
                    except (ValueError, KeyError, TypeError):
                        save(out / f"{arm}.failed-request.json",
                             {"id": row["id"], "response": response, "completed_requests": outputs})
                        raise
                    if i:
                        outputs.append(result)
                elapsed = stop_req - start
                stats = summarize(outputs, elapsed, bool(draft))
                save(out / f"{arm}.json", {"summary": stats, "requests": outputs})
                runs[arm] = outputs, stats
    if args.target_only:
        print(json.dumps({"mode": manifest["mode"], "target": runs["target"][1]}, indent=2))
        return
    report = compare(runs)
    save(out / "comparison.json", report)
    print(json.dumps(report, indent=2))
    if any(report["output_mismatches"].values()):
        raise SystemExit("greedy token IDs diverged; prompt IDs and raw responses are saved for diagnosis")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--eagle3", type=Path)
    parser.add_argument("--eagle31", type=Path)
    parser.add_argument("--target-only", action="store_true", help="standalone diagnostic, requires no draft exports")
    parser.add_argument("--resume", action="store_true", help="verify an interrupted run and finish the missing arms")
    parser.add_argument("--devices", default="0,1", help="comma-separated physical Ascend device IDs, TP equals their number")
    parser.add_argument("--device-map", type=Path, help="host npu-smi info -m output if unavailable in container")
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("--held-out", type=Path, help="train/data/held-out.jsonl and its adjacent manifest.json")
    dataset.add_argument("--gsm8k", type=Path, help="local pinned GSM8K test.jsonl; auto-fetch if no dataset is given")
    parser.add_argument("--count", type=int, help="measured requests, default: all held-out prompts or 64 GSM8K")
    parser.add_argument("--port", type=int, default=30231)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--fraction", type=float, default=.85)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
