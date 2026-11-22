"""Recompute matched serving and correctness evidence for any completed cells; no accelerator needed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from evaluate import cell_matrix, metrics, read
from npu_runtime import check, save

def wilson(k,n,z=1.95996398454):
    if not n:return None
    p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d
    h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [max(0,m-h),min(1,m+h)]

def paired_ci(diffs):
    if not diffs:return None
    rng=random.Random(42);n=len(diffs)
    reps=sorted(sum(rng.choice(diffs) for _ in range(n))/n for _ in range(2000))
    return [reps[49],reps[1949]]

def cell(path,c):
    d=read(path/(c["key"]+".json"))
    check(d["cell"]==c and metrics(d["requests"],d["summary"]["wall_seconds"])==d["summary"],"raw metric mismatch")
    return d

def compare_rows(ar, draft):
    base={r["key"]:r for r in ar}
    check(len(base)==len(ar)==len(draft),"duplicate or missing request")
    exact=0;lcp_sum=0;delta=[];first=[]
    for s in draft:
        b=base.pop(s["key"])
        check(s["prompt_tokens"]==b["prompt_tokens"],"different prompts")
        x,y=b["output_ids"],s["output_ids"]
        k=next((j for j,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)))
        lcp_sum+=k
        if x==y: exact+=1
        else:first.append({"key_sha256":s["key"],"first_difference_zero_based":k})
        if "correct" in b:delta.append(int(s["correct"])-int(b["correct"]))
    check(not base,"unmatched IDs")
    return dict(exact_requests=exact,requests=len(ar),mean_longest_common_prefix_tokens=lcp_sum/len(ar),first_differences=first,
                paired_accuracy_delta=sum(delta)/len(delta) if delta else None,
                paired_accuracy_delta_bootstrap95=paired_ci(delta))

def load_runs(paths):
    runs=[];shared=None
    for p in paths:
        m=read(p/"manifest.json")
        identity={k:m[k] for k in ("stage","hostname","physical_devices","runtime_devices","cases_sha256","gold_sha256",
                                   "model_sha_manifest","temperature","profile","max_running_requests","seed")}
        identity["sglang"]=m["runtime"]["expected_revision"]
        if shared is None:shared=identity
        check(identity==shared,"run-level hardware, target, protocol or source mismatch")
        runs.append((m["name"],m,p))
    check(len(set(n for n,_,_ in runs))==len(runs),"duplicate human-readable names")
    return shared,runs

def make(paths):
    identity,runs=load_runs(paths)
    bases=[(n,m,p) for n,m,p in runs if m["arm"]=="target"]
    check(len(bases)==1,"exactly one target-only run is required")
    bn,_,bp=bases[0]
    report={"stage":identity["stage"],"base":bn,"dataset_sha256":identity["cases_sha256"],
            "target_file_manifest_sha256":identity["model_sha_manifest"],"sglang":identity["sglang"],
            "runs":{},"cells":{},"missing":{},"repeat_stability":{},"repeated_throughput":{}}
    for n,m,p in runs:
        did=m["draft_identity"];report["runs"][n]={"arm":m["arm"],"draft_sha256":did["sha256"] if did else None}
    for c in identity["profile"]:
        key=c["key"];prior=None;current={};raw_cells={}
        if (bp/(key+".json")).exists():prior=cell(bp,c)
        for name,m,p in runs:
            path=p/(key+".json")
            if not path.exists():
                report["missing"].setdefault(name,[]).append(key);continue
            d=cell(p,c);s=d["summary"];record=dict(summary=s)
            raw_cells[name]=d
            if s["correct"] is not None:record["gsm8k_accuracy_wilson95"]=wilson(s["correct"],s["n"])
            if c.get("stream"):
                nonstream=p/(key[:-len("-stream")]+".json")
                if nonstream.exists():
                    public=compare_rows(read(nonstream)["requests"],d["requests"])
                    record["stream_vs_same_arm_nonstream_exact"]={"identical_requests":public["exact_requests"],"n":s["n"]}
            if prior and name!=bn:
                matched=compare_rows(prior["requests"],d["requests"])
                ratio=s["throughput_tok_s"]/prior["summary"]["throughput_tok_s"]
                record["vs_target"]=dict(**matched,wall_tok_s_ratio=ratio,
                                          ratio_numerator=s["throughput_tok_s"],
                                          ratio_denominator=prior["summary"]["throughput_tok_s"],
                                          equal_output_speedup=ratio if matched["exact_requests"]==s["n"] else None)
            current[name]=record
            if c["repeat"]==2:
                ck=key.rsplit("-r",1)[0];first=p/(ck+"-r1.json")
                if first.exists():
                    d1=read(first)
                    repeats=compare_rows(d1["requests"],d["requests"])
                    rates=[d1["summary"]["throughput_tok_s"],s["throughput_tok_s"]]
                    report["repeat_stability"].setdefault(name,{})[ck]={"exact_requests":repeats["exact_requests"],"n":s["n"]}
                    report["repeated_throughput"].setdefault(name,{})[ck]={"replicate_tok_s":rates,"median_tok_s":statistics.median(rates)}
        refnames=[n for n,m,_ in runs if m["arm"]=="pretrained" and n in raw_cells]
        if len(refnames)==1:
            refname=refnames[0];ref=raw_cells[refname]
            for n,m,_ in runs:
                if n not in raw_cells or m["arm"] in ("target","pretrained"):continue
                cand=raw_cells[n];match=compare_rows(ref["requests"],cand["requests"])
                numerator=cand["summary"]["throughput_tok_s"];denominator=ref["summary"]["throughput_tok_s"]
                current[n]["vs_pretrained_eagle3"]=dict(**match,reference=refname,ratio_numerator=numerator,
                    ratio_denominator=denominator,wall_tok_s_ratio=numerator/denominator,
                    equal_output_speedup=numerator/denominator if match["exact_requests"]==len(ref["requests"]) else None)
        if current:report["cells"][key]=current
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",action="append",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args();save(a.out,make(a.run))
