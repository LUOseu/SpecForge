"""Replay differing requests and measure the target's reported token scores on 910B3."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import requests

import benchmark as b


def hash_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def observe(session, port, ids, n_tokens, index, expected_ids, ref_a, ref_b):
    data = {"input_ids": ids, "stream": False, "return_logprob": True, "top_logprobs_num": 5,
            "logprob_start_len": -1, "sampling_params": {"temperature": 0, "max_new_tokens": n_tokens}}
    response = session.post(f"http://127.0.0.1:{port}/generate", json=data, timeout=180)
    if not response.ok:
        return {"http_status": response.status_code, "error": response.text[:350]}
    payload = response.json()
    outputs, meta = payload["output_ids"], payload["meta_info"]
    record = {"completion_count": len(outputs),
              "common_prefix_reproduced": outputs[:index] == expected_ids[:index],
              "selected_matches_ar": len(outputs) > index and outputs[index] == ref_a,
              "selected_matches_eagle31": len(outputs) > index and outputs[index] == ref_b,
              "spec_verify_ct": meta.get("spec_verify_ct")}
    top = meta.get("output_top_logprobs") or []
    selected = meta.get("output_token_logprobs") or []
    if index >= len(top) or len(selected) != len(outputs):
        record["score_status"] = "response has no aligned top-k output scores"
        return record
    if any(row[1] != emitted for row, emitted in zip(selected, outputs)):
        record["score_status"] = "returned score token IDs do not align with output IDs"
        return record
    dist = sorted(top[index], key=lambda entry: entry[0], reverse=True)
    record["score_status"] = "reported"
    record["top1_minus_top2_logprob"] = dist[0][0] - dist[1][0] if len(dist) >= 2 else None
    for label, token in (("ar_candidate", ref_a), ("eagle31_candidate", ref_b)):
        record[label] = next(({"rank_in_top5": j + 1, "logprob": v[0]} for j, v in enumerate(dist) if v[1] == token), None)
    return record


def run(args):
    inp = args.input
    target, draft = [json.loads((inp / f"{n}.json").read_text())["requests"] for n in ("target", "eagle31")]
    m = json.loads((inp / "manifest.json").read_text())
    logdir = args.out
    logdir.mkdir(parents=True, exist_ok=False)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = m["runtime_visible_devices"]
    os.environ["SPECFORGE_SPEC_CAPTURE"] = "0"
    b.installed_runtime()
    inputs = json.loads((inp / "prompts.json").read_text())["measured"]
    b.check(len(target) == len(draft) == len(inputs), "request count mismatch")
    records = []
    for t, s, q in zip(target, draft, inputs):
        b.check(t["id"] == s["id"] == q["id"], "request order mismatch")
        difference = next((i for i, (u, v) in enumerate(zip(t["output_ids"], s["output_ids"])) if u != v), None)
        if difference is not None:
            records.append((q["input_ids"], t["output_ids"], s["output_ids"], difference))
    report = {"mode": "logprob diagnostic, no timing", "top_k": 5, "query_types": ["original replay", "prefill common prefix, one output token"],
              "zero_based_first_difference": True, "arms": {}}
    for arm in ("target", "eagle31"):
        cmd = m["commands"][arm].copy()
        cmd[cmd.index("--port") + 1] = str(args.port)
        report["arms"][arm] = []
        with requests.Session() as session:
            session.trust_env = False
            with b.server(cmd, args.port, logdir / f"{arm}.log", session) as info:
                b.check_server(info, Path(m["target"]), None if arm == "target" else Path(m["drafts"][arm]["path"]),
                               m["tp"], .85, 2048)
                for ids, ar, sd, k in records:
                    expected = ar if arm == "target" else sd
                    row = {"input_sha256": hash_ids(ids), "first_difference_index": k}
                    row["replay"] = observe(session, args.port, ids, m["sampling"]["max_new_tokens"], k, expected, ar[k], sd[k])
                    row["common_prefix_prefill"] = observe(session, args.port, ids + ar[:k], 1, 0, [], ar[k], sd[k])
                    report["arms"][arm].append(row)
                    b.save(logdir / "greedy-scores.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30677)
    run(parser.parse_args())
