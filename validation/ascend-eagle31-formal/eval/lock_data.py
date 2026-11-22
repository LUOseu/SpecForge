"""Freeze external evaluation questions independently of training. Uses CPU only."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.request import urlopen

GSM = "740312add88f781978c0658806c59bc2815b9866"
MT = "e3a795c5e9a82ee40611c416b8a7786c73198991"
URLS = {
    "train.parquet": (f"https://huggingface.co/datasets/openai/gsm8k/resolve/{GSM}/main/train-00000-of-00001.parquet", "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"),
    "test.parquet": (f"https://huggingface.co/datasets/openai/gsm8k/resolve/{GSM}/main/test-00000-of-00001.parquet", "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"),
    "mtbench.jsonl": (f"https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/resolve/{MT}/raw/question.jsonl", "420ae1d63cbf0f499d8afc3b28d86714e77f13e44c133eedf942ed16c544dd34"),
}
SEED = "scale-20261121/external-eval-v1"
MATH_INSTRUCTION = "Solve the problem. Show your calculation briefly and finish with a separate line in the form #### <number>."
TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def norm_hash(question):
    return sha(" ".join(unicodedata.normalize("NFKC", question).casefold().split()).encode("utf-8"))

def stable_order(rows):
    return sorted(rows, key=lambda r: sha((SEED + "/" + r["hash"]).encode()))

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def freeze(src, out, tokenizer_dir):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    src.mkdir(exist_ok=True, parents=True)
    out.mkdir(exist_ok=True, parents=True)
    for file,(url,digest) in URLS.items():
        dest=src/file
        if not dest.exists():
            with urlopen(url,timeout=90) as response:
                dest.write_bytes(response.read())
        if sha(dest.read_bytes())!=digest: raise ValueError(f"source checksum mismatch: {file}")
    if sha((tokenizer_dir/"tokenizer.json").read_bytes())!=TOKENIZER_SHA:
        raise ValueError("unexpected target tokenizer")
    tok=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    seen=set(); bysplit={}
    for split in ["test","train"]:
        rows=[]
        for i,rec in enumerate(pq.read_table(src/(split+".parquet")).to_pylist()):
            question=rec["question"]; key=norm_hash(question)
            if key in seen: continue
            seen.add(key)
            m=re.search(r"(?m)^####\s*([+-]?[\d,.]+)\s*$",rec["answer"])
            if not m: raise ValueError(f"invalid GSM8K gold on {split} row {i}")
            rows.append(dict(hash=key,item=i,question=question,answer=m[1].replace(",",""),category="math"))
        bysplit[split]=stable_order(rows)
    train_select=bysplit["train"][:36]
    groups={"warmup":train_select[:4],"dev":train_select[4:36],"gsm8k-final":bysplit["test"][:128]}
    mtrows=[json.loads(line) for line in (src/"mtbench.jsonl").read_text().splitlines() if line.strip()]
    mt=[dict(hash=norm_hash(r["prompt"][0]),item=r["question_id"],question=r["prompt"][0],category=r["category"]) for r in mtrows]
    groups["mtbench-first-turn"]=[]
    for i in range(8):
        for category in ["writing","extraction"]:
            groups["mtbench-first-turn"].append(stable_order([r for r in mt if r["category"]==category])[i])
    # Reserve the entire external test split and all MT-Bench first turns against training contamination.
    deny={r["hash"] for r in bysplit["test"]+train_select+mt}
    (out/"denylist.sha256").write_text("".join(v+"\n" for v in sorted(deny)))
    cases={}; labels={}
    for group,rs in groups.items():
        cases[group]=[]
        for row in rs:
            q=row["question"]
            if row["category"]=="math":
                q=q+"\n\n"+MATH_INSTRUCTION
                labels[row["hash"]]=row["answer"]
            ids=tok.apply_chat_template([{"role":"user","content":q}],tokenize=True,return_dict=False,enable_thinking=False,add_generation_prompt=True)
            cases[group].append(dict(hash=row["hash"], item=row["item"],category=row["category"],input_ids=ids))
    dump(out/"cases.json",cases); dump(out/"gold.json",labels)
    meta={"seed":SEED,"sources":URLS,"gsm8k_repo":"openai/gsm8k","gsm8k_revision":GSM,
          "mt_bench_repo":"HuggingFaceH4/mt_bench_prompts","mt_bench_revision":MT,"tokenizer_sha256":TOKENIZER_SHA,
          "math_instruction":MATH_INSTRUCTION,"selection":"ascending SHA256(seed + '/' + normalized_question_sha256), UTF-8",
          "deny_normalization":"UTF8(' '.join(NFKC(text).casefold().split())) then SHA-256",
          "unique_official_test_questions":len(bysplit["test"]),"deny_count":len(deny),
          "counts":{k:len(v) for k,v in cases.items()},"max_input_tokens":{k:max(len(r["input_ids"]) for r in v) for k,v in cases.items()},
          "digests":{n:sha((out/n).read_bytes()) for n in ["denylist.sha256","cases.json","gold.json"]}}
    dump(out/"protocol.json",meta)
    return meta

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("source","output","tokenizer"): p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args(); print(json.dumps(freeze(a.source,a.output,a.tokenizer),indent=2))
