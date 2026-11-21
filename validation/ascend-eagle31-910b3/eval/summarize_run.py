"""Compute a shareable three-arm summary from the private per-request benchmark evidence."""
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path

from benchmark import ARMS, PIN, check, compare, file_sha, save, summarize


def hash_ids(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def summarize_run(root, diagnostic=None):
    def load(name):
        return json.loads((root / name).read_bytes())

    manifest, prompts = load("manifest.json"), load("prompts.json")["measured"]
    data = {arm: load(f"{arm}.json") for arm in ARMS}
    requests = {arm: record["requests"] for arm, record in data.items()}
    summaries = {arm: record["summary"] for arm, record in data.items()}
    for arm in ARMS:
        rows, summary = requests[arm], summaries[arm]
        check([r["id"] for r in rows] == [p["id"] for p in prompts], "request alignment failed")
        check(summarize(rows, summary["measurement_seconds"], arm != "target") == summary, "aggregate mismatch")
        for r, p in zip(rows, prompts):
            check(len(p["input_ids"]) == r["prompt_tokens"] and len(r["output_ids"]) == r["completion_tokens"],
                  "token count error")
            if arm != "target":
                check(math.isclose(r["spec_accept_length"], r["completion_tokens"] / r["spec_verify_ct"])
                      and math.isclose(r["spec_accept_rate"], r["spec_num_correct_drafts"] / r["spec_num_proposed_drafts"])
                      and r["spec_num_proposed_drafts"] == 3 * r["spec_verify_ct"], "invalid request counters")
    pairs = list(itertools.combinations(ARMS, 2))
    report = {
        "dataset": "Aeala/ShareGPT_Vicuna_unfiltered",
        "dataset_revision": "8b0048ad6ae8c22f46a78c15559dec98feef5539",
        "target": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "target_revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "sglang_source_revision": PIN,
        "runtime_srt_python_tree_sha256": manifest["runtime"]["srt_tree_sha256"],
        "packages": manifest["packages"], "hardware": manifest["npu_names"],
        "physical_npu_ids": manifest["physical_devices"], "runtime_visible_ids": manifest["runtime_visible_devices"],
        "tp_size": manifest["tp"], "concurrency": manifest["concurrency"], "seed": manifest["seed"],
        "sampling": manifest["sampling"], "steps": 3, "eagle_topk": 1, "num_draft_tokens": 4,
        "measurement": "one separate warmup, end-to-end client wall time over all measured requests, non-streaming",
        "pooled_accept_length_definition": "sum(completion_tokens) / sum(spec_verify_ct), including bonus tokens",
        "strict_draft_rate_definition": "sum(spec_num_correct_drafts) / sum(spec_num_proposed_drafts), excluding bonus tokens",
        "draft_weights_sha256": {k:v["files_sha256"]["model.safetensors"] for k,v in manifest["drafts"].items()},
        "source_results_sha256": {arm:file_sha(root / f"{arm}.json") for arm in ARMS},
        "prompt_hash_algorithm": "SHA-256 of the UTF-8 compact JSON array of input token IDs, decimal integers and comma separators",
        "zero_based_first_difference": True, "summary": summaries, "pairs": {}, "requests": [],
    }
    for i, prompt in enumerate(prompts):
        entry = {"input_sha256": hash_ids(prompt["input_ids"]), "prompt_tokens": len(prompt["input_ids"]), "arms": {},
                 "first_output_difference": {}}
        for arm in ARMS:
            row = requests[arm][i]
            allowed = ("completion_tokens", "seconds", "spec_verify_ct", "spec_num_correct_drafts",
                       "spec_num_proposed_drafts", "spec_accept_length", "spec_accept_rate")
            entry["arms"][arm] = {k: row[k] for k in allowed if k in row}
            entry["arms"][arm].update(output_sha256=hash_ids(row["output_ids"]), finish=row["finish_reason"]["type"])
        for a,b in pairs:
            left,right = requests[a][i]["output_ids"],requests[b][i]["output_ids"]
            k = next((j for j,(u,v) in enumerate(zip(left,right)) if u != v), None)
            if k is None and len(left)!=len(right):
                k=min(len(left),len(right))
            entry["first_output_difference"][f"{a}__{b}"] = k
        report["requests"].append(entry)
    calc = compare({a:(requests[a],summaries[a]) for a in ARMS})
    for a,b in pairs:
        key=f"{a}__{b}"
        exact = sum(row["first_output_difference"][key] is None for row in report["requests"])
        report["pairs"][key]={"identical_requests":exact,"requests":len(prompts),
                              "throughput_ratio_second_over_first_when_outputs_match":
                              summaries[b]["wall_output_tok_s"]/summaries[a]["wall_output_tok_s"] if exact==len(prompts) else None}
    report["eagle31_minus_eagle3_pooled_accept_length"] = calc["eagle31_minus_eagle3_accept_length"]
    safe_fields = ("device", "attention_backend", "sampling_backend", "disable_custom_all_reduce",
                   "dtype", "context_length", "max_running_requests",
                   "chunked_prefill_size", "mem_fraction_static", "disable_radix_cache", "disable_overlap_schedule",
                   "speculative_algorithm", "speculative_eagle_topk", "speculative_num_steps", "speculative_num_draft_tokens",
                   "enable_deterministic_inference", "speculative_use_rejection_sampling",
                   "speculative_accept_threshold_single", "speculative_accept_threshold_acc")
    report["server_settings"] = {}
    for arm in ARMS:
        info = load(f"{arm}.server.json")
        for line in (root/f"{arm}.log").read_text(errors="replace").splitlines():
            if "server_args=" in line:
                info = ast.literal_eval(line.split("server_args=",1)[1]) | info
                break
        report["server_settings"][arm] = {k:info.get(k) for k in safe_fields}
    if diagnostic:
        d = json.loads((diagnostic/"target.json").read_bytes())
        check([r["id"] for r in d["requests"]] == [p["id"] for p in prompts], "diagnostic prompt IDs differ")
        report["independent_target_only_repeat"] = {
            "identical_to_formal_baseline":sum(x["output_ids"]==y["output_ids"] for x,y in zip(d["requests"], requests["target"])),
            "requests":len(prompts), "wall_output_tok_s":d["summary"]["wall_output_tok_s"],
            "used_in_comparisons":False,
        }
    return report


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input",required=True,type=Path)
    cli.add_argument("--diagnostic",type=Path)
    cli.add_argument("--out",required=True,type=Path)
    args=cli.parse_args()
    save(args.out,summarize_run(args.input,args.diagnostic))
