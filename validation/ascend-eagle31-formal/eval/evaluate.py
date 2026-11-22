"""Run a fresh, verifiable SGLang checkpoint evaluation on a reserved Ascend TP2 pair."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
from decimal import InvalidOperation
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import threading
import time

from fetch_reference import FILES, REV
import npu_runtime as rt

HERE=Path(__file__).resolve().parent
CAP_CONTEXT=2048
FRACTION=.85
FINAL_NUMBER=re.compile(r"^\s*####\s*\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?)\s*[.]?\s*$")
PROTOCOL_SHA="6b37729faf9ca3634742719d3c043392a308077e2aed13b7ac9a464c54bdfce0"
GOLD_SHA="263ae7d12b76b2d033bbd0ab19197c2af9de4a7ba4291e8292f967fdd67eb865"

def hash_ids(ids):
    return hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def score(text, gold):
    lines=text.strip().splitlines()
    match=FINAL_NUMBER.fullmatch(lines[-1]) if lines else None
    if not match: return None,False
    found=match.group(1).replace(",","")
    try: return found, Fraction(found)==Fraction(gold)
    except (ValueError,ZeroDivisionError,InvalidOperation): return found,False

def cell_matrix(stage):
    if stage=="screen": base=[("dev",16,256,(1,),1)]
    elif stage=="dev": base=[("dev",32,256,(1,4),1)]
    elif stage=="final":
        base=[("gsm8k-final",128,512,(1,4),2),("mtbench-first-turn",16,256,(1,4),2),
              ("gsm8k-final",32,256,(1,4),1)]
    else: raise ValueError(stage)
    out=[]
    for group,count,limit,concurrencies,repeats in base:
        for repeat in range(repeats):
            for c in (concurrencies if repeat%2==0 else tuple(reversed(concurrencies))):
                out.append(dict(key=f"{group}-n{count}-o{limit}-c{c}-r{repeat+1}",group=group,count=count,
                                cap=limit,concurrency=c,repeat=repeat+1))
    if stage=="final":
        for c in (1,4):
            out.append(dict(key=f"mtbench-first-turn-n16-o256-c{c}-r1-stream",group="mtbench-first-turn",
                            count=16,cap=256,concurrency=c,repeat=1,stream=True))
    return out

def selected_rows(cases, cell):
    group=cases[cell["group"]][:cell["count"]]
    return sorted(group,key=lambda r:hashlib.sha256(f"42/{cell['repeat']}/{r['hash']}".encode()).digest())

def target_identity(target,cache_dir):
    expected=read(HERE/"target-sha256.json")
    cache_dir.mkdir(parents=True,exist_ok=True)
    fp={}
    for name in expected:
        t=(target/name).stat()
        fp[name]=[t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns]
    machine=hashlib.sha256((socket.gethostname()+str(target)).encode()).hexdigest()[:18]
    cached=cache_dir/(machine+".json")
    with (cache_dir/(machine+".lock")).open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        prev=read(cached) if cached.exists() else {}
        if prev.get("file_stats")!=fp or prev.get("sha256")!=expected:
            for name,value in expected.items():
                rt.check(rt.file_sha(target/name)==value,f"target snapshot hash mismatch: {name}")
            rt.save(cached,dict(file_stats=fp,sha256=expected))
    return rt.file_sha(HERE/"target-sha256.json")

def draft_identity(arm,draft,require_ready):
    if arm=="target":
        rt.check(draft is None,"target arm takes no draft")
        return None
    rt.check(draft and draft.is_dir(),"draft arm requires its serving export")
    cfg=read(draft/"config.json")
    rt.check(cfg["architectures"]==["LlamaForCausalLMEagle3"],"unexpected draft architecture")
    variant=arm=="eagle31"
    rt.check(bool(cfg.get("fc_norm",False))==variant and bool(cfg.get("norm_output",False))==variant,"wrong draft normalization switches")
    weights=sorted(draft.glob("*.safetensors"))
    rt.check(weights,"no draft weight shards")
    sha={p.name:rt.file_sha(p) for p in [draft/"config.json",*weights]}
    ready=read(draft/"ready.json") if (draft/"ready.json").exists() else None
    if arm=="pretrained":
        rt.check(sha["config.json"]==FILES["config.json"] and sha["model.safetensors"]==FILES["model.safetensors"],"public pretrained reference mismatch")
    else:
        rt.check(cfg.get("specforge_artifact",{}).get("sglang_commit")==rt.PIN,"wrong/missing SpecForge serving-export pin")
        rt.check(not require_ready or ready is not None,"await the training publisher's ready.json")
        if ready is not None:
            rt.check(ready["config_sha256"]==sha["config.json"] and ready["model_sha256"]==sha["model.safetensors"]
                     and ready["weight_bytes"]==(draft/"model.safetensors").stat().st_size,
                     "training publisher's ready.json does not match the serving export")
    return dict(sha256=sha,ready=ready,pretrained_revision=REV if arm=="pretrained" else None)

def parse_response(item,res,seconds,cap,speculative,gold):
    meta=res["meta_info"]
    tokens=res["output_ids"]
    count=meta["completion_tokens"]
    finish=meta["finish_reason"]["type"]
    rt.check(finish in ("stop","length") and count==len(tokens) and 0<count<=cap, "invalid generation result")
    rt.check(meta["prompt_tokens"]==len(item["input_ids"]),"prompt length mismatch")
    row=dict(key=item["hash"],category=item["category"],prompt_tokens=meta["prompt_tokens"],output_tokens=count,
             output_sha256=hash_ids(tokens),output_ids=tokens,finish=finish,latency_seconds=seconds)
    if item["category"]=="math":
        answer,correct=score(res["text"],gold[item["hash"]])
        row.update(parsed_answer=answer,correct=correct)
    rounds=meta.get("spec_verify_ct",0)
    if speculative and not rounds:
        rt.check(count==1,"missing speculative counters on multi-token reply")
    if not speculative:
        rt.check(not rounds,"target-only endpoint reported speculative rounds")
    if rounds:
        good,proposed=meta["spec_num_correct_drafts"],meta["spec_num_proposed_drafts"]
        rt.check(type(rounds) is int and rounds>0 and proposed==3*rounds and 0<=good<=proposed,
                 "invalid speculative counters")
        rt.check(math.isclose(count/rounds,meta["spec_accept_length"]) and
                 math.isclose(good/proposed,meta["spec_accept_rate"]),"runtime acceptance definition mismatch")
        row.update(verify_rounds=rounds,accepted_draft_tokens=good,proposed_draft_tokens=proposed)
    else: row.update(verify_rounds=0,accepted_draft_tokens=0,proposed_draft_tokens=0)
    return row

def read_stream(response, start, incremental):
    first=last=None;first_count=previous=0;ids=[];text="";meta=None;done=False
    for line in response.iter_lines(chunk_size=None):
        received=time.perf_counter()
        if not line.startswith(b"data:"):continue
        packet=line[5:].strip()
        if packet==b"[DONE]":done=True;continue
        rt.check(not done,"SSE emitted data after the terminal marker")
        body=json.loads(packet);meta=body["meta_info"]
        count=meta["completion_tokens"]
        rt.check(count>=previous,"nonmonotonic streaming token counter")
        if count>previous:
            if first is None:first,first_count=received,count
            last=received
        previous=count
        if incremental:
            ids.extend(body.get("output_ids",[]));text+=body.get("text") or ""
        else:
            ids=body.get("output_ids",ids);text=body.get("text",text)
    rt.check(done and meta and first is not None and previous==len(ids),"incomplete SSE result")
    interval=last-first
    timing={"ttft_ms":1000*(first-start),"first_chunk_tokens":first_count,"decode_observed_seconds":interval,
            "tokens_after_first_chunk":previous-first_count,
            "tpot_ms":1000*interval/(previous-1) if previous>1 and interval>0 else None}
    return {"meta_info":meta,"output_ids":ids,"text":text},timing


class Client:
    def __init__(self, port, speculative, gold, stream=False, incremental=False):
        self.url=f"http://127.0.0.1:{port}/generate";self.speculative=speculative;self.gold=gold
        self.stream=stream;self.incremental=incremental
        self.local=threading.local();self.sessions=[];self.lock=threading.Lock()
    def request(self,item,cap):
        import requests
        if not hasattr(self.local,"session"):
            session=requests.Session();session.trust_env=False
            self.local.session=session
            with self.lock:self.sessions.append(session)
        payload={"input_ids":item["input_ids"],"sampling_params":{"temperature":0,"max_new_tokens":cap},"stream":self.stream}
        start=time.perf_counter()
        with self.local.session.post(self.url,json=payload,stream=self.stream,timeout=(15,600)) as response:
            response.raise_for_status()
            if self.stream:
                body,timing=read_stream(response,start,self.incremental)
                elapsed=time.perf_counter()-start
            else:
                elapsed=time.perf_counter()-start;body=response.json();timing={}
        row=parse_response(item,body,elapsed,cap,self.speculative,self.gold)
        row.update(timing)
        return row
    def close(self):
        for s in self.sessions:s.close()

def metrics(rows,seconds):
    total=sum(x["output_tokens"] for x in rows);n=len(rows)
    rounds=sum(x["verify_rounds"] for x in rows)
    accepted=sum(x["accepted_draft_tokens"] for x in rows);proposed=sum(x["proposed_draft_tokens"] for x in rows)
    mathrows=[r for r in rows if r["category"]=="math"]
    times=sorted(x["latency_seconds"] for x in rows)
    summary=dict(n=n,wall_seconds=seconds,tokens=total,throughput_tok_s=total/seconds,latency_median_s=statistics.median(times),
                latency_p95_s=times[math.ceil(.95*n)-1],finished_at_cap=sum(r["finish"]=="length" for r in rows),
                verify_rounds=rounds,verified_output_tokens=sum(r["output_tokens"] for r in rows if r["verify_rounds"]),
                accepted_draft_tokens=accepted,proposed_draft_tokens=proposed,
                pooled_accept_length=sum(r["output_tokens"] for r in rows if r["verify_rounds"])/rounds if rounds else None,
                strict_draft_accept_rate=accepted/proposed if proposed else None,
                correct=sum(r["correct"] for r in mathrows) if mathrows else None,
                unparsed=sum(r["parsed_answer"] is None for r in mathrows) if mathrows else None,
                gsm8k_accuracy=sum(r["correct"] for r in mathrows)/len(mathrows) if mathrows else None)
    if rows and "ttft_ms" in rows[0]:
        tpot=[r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]
        denom=sum(r["decode_observed_seconds"] for r in rows)
        summary.update(mean_ttft_ms=statistics.mean(r["ttft_ms"] for r in rows),
                       mean_tpot_ms=statistics.mean(tpot) if tpot else None,tpot_observed_requests=len(tpot),
                       mean_first_chunk_tokens=statistics.mean(r["first_chunk_tokens"] for r in rows),
                       per_request_decode_tok_s_pooled=sum(r["tokens_after_first_chunk"] for r in rows)/denom if denom else None)
    return summary

def measure_cell(client,cell,cases):
    rows=selected_rows(cases,cell)
    # Measured rows enter an empty request queue after all warmups complete; one persistent client/session per worker.
    with ThreadPoolExecutor(cell["concurrency"]) as pool:
        list(pool.map(lambda r:client.request(r,cell["cap"]),cases["warmup"]))
        start=time.perf_counter()
        outputs=list(pool.map(lambda r:client.request(r,cell["cap"]),rows))
        elapsed=time.perf_counter()-start
    return dict(cell=cell,summary=metrics(outputs,elapsed),requests=outputs)

def evaluate(a):
    import requests
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    with (out/".measure.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cells=cell_matrix(a.stage)
        if a.cells:
            want=set(a.cells.split(","));cells=[c for c in cells if c["key"] in want]
            rt.check({c["key"] for c in cells}==want,"unknown requested cell")
        rt.check(len(a.devices.split(","))==2 and len(set(a.devices.split(",")))==2,"locked run requires two distinct physical 910B3 devices")
        visible,listing=rt.resolve_devices(a.devices.split(","),a.device_map)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"]=visible;os.environ["SPECFORGE_SPEC_CAPTURE"]="0"
        rt.check(float(os.environ.get("SGLANG_SIMULATE_ACC_LEN","-1"))<=0,"simulated acceptance is enabled")
        target=a.target.resolve(strict=True);draft=a.draft.resolve(strict=True) if a.draft else None
        runtime=rt.installed_runtime()
        cases_path=a.frozen/"cases.json";gold_path=a.frozen/"gold.json"
        rt.check(rt.file_sha(cases_path)==PROTOCOL_SHA and rt.file_sha(gold_path)==GOLD_SHA,"locked questions or gold have changed")
        cases=read(cases_path);gold=read(gold_path)
        modelsha=target_identity(target,a.cache)
        draftsha=draft_identity(a.arm,draft,a.require_ready)
        cmd=rt.launch_args(target,draft,2,a.port,FRACTION,CAP_CONTEXT,False)
        manifest=dict(stage=a.stage,arm=a.arm,name=a.name,model_sha_manifest=modelsha,target=str(target),
                      draft=str(draft) if draft else None,draft_identity=draftsha,physical_devices=a.devices,runtime_devices=visible,
                      hostname=socket.gethostname(),runtime=runtime,command=cmd,cases_sha256=PROTOCOL_SHA,gold_sha256=GOLD_SHA,
                      python=sys.version,profile=cell_matrix(a.stage),max_running_requests=4,temperature=0,seed=42)
        mpath=out/"manifest.json"
        if mpath.exists(): rt.check(read(mpath)==manifest,"different run identity at output path")
        else: rt.save(mpath,manifest);(out/"device-map.txt").write_text(listing)
        pending=[]
        for cell in cells:
            path=out/(cell["key"]+".json")
            if not path.exists(): pending.append(cell);continue
            saved=read(path)
            rt.check(saved["cell"]==cell and [r["key"] for r in saved["requests"]]==[r["hash"] for r in selected_rows(cases,cell)],
                     "completed cell has wrong rows")
            rt.check(metrics(saved["requests"],saved["summary"]["wall_seconds"])==saved["summary"],"completed-cell counters differ")
        if not pending:return
        with requests.Session() as ctl:
            ctl.trust_env=False;log=out/f"server-{time.time_ns()}.log"
            with rt.server(cmd,a.port,log,ctl) as info:
                rt.save(out/"server.json",rt.check_server(info,target,draft,2,FRACTION,CAP_CONTEXT))
                for cell in pending:
                    client=Client(a.port,bool(draft),gold,stream=bool(cell.get("stream")),
                                  incremental=bool(info.get("incremental_streaming_output",False)))
                    try: result=measure_cell(client,cell,cases)
                    finally:client.close()
                    rt.save(out/(cell["key"]+".json"),result)
                    print(json.dumps({"arm":a.arm,"cell":cell["key"],**result["summary"]}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("screen","dev","final"),required=True)
    p.add_argument("--arm",choices=("target","pretrained","eagle3","eagle31"),required=True)
    p.add_argument("--name",required=True,help="stable checkpoint/source label")
    p.add_argument("--target",type=Path,required=True);p.add_argument("--draft",type=Path)
    p.add_argument("--devices",required=True,help="authorized physical IDs in rank order")
    p.add_argument("--device-map",type=Path);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True,help="local target snapshot verification cache")
    p.add_argument("--frozen",type=Path,default=HERE/"frozen");p.add_argument("--port",type=int,default=30681)
    p.add_argument("--require-ready",action="store_true");p.add_argument("--cells",help="comma-separated locked cell keys")
    evaluate(p.parse_args())
