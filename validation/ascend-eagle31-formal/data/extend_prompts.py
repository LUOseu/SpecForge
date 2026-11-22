"""Append fully checked ShareGPT and UltraChat teacher prompts; preserve the exact released base prefix."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import runpy
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from prepare_prompts import DENY_SHA,SEED,TOKENIZER_SHA,digest,interleave,rowhash,sha

ULTRA_REV='8049631c405ae6576f93f445c6b8166f76f5505a'
SHARDS={'train_sft-00000-of-00003-a3ecf92756993583.parquet':'afa8fa7426081b2a0e732fb50dbb5cd402a28ad5f0dbe66c0d996d63e7220727',
        'train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet':'c8230190bc8b29084c974ff32f05589b458fad1908fc4b2fb2b9e1e9f7921f03',
        'train_sft-00002-of-00003-ee46ed25cfae92c6.parquet':'6fe7d2a5e95cf660f972ecaf304aa5632d7f53384e4b0e0d0f44d9c96733c03e'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','eligible','ultra','block','deny','model','repo','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    b=json.loads((a.base/'source-manifest.json').read_text());assert sha(a.model/'tokenizer.json')==TOKENIZER_SHA and sha(a.deny)==DENY_SHA
    ban=set(a.deny.read_text().splitlines());crossexact=set((a.block/'normalized.sha256').read_text().splitlines());crossnear=set((a.block/'near.sha256').read_text().splitlines())
    counts=Counter();existing=[];base_sha=hashlib.sha256()
    with (a.eligible/'train-prompts.jsonl').open('rb') as f:
        for n,line in enumerate(f):
            if n<b['selections']['train']['count']:base_sha.update(line)
            else:existing.append(json.loads(line))
    assert base_sha.hexdigest()==b['selections']['train']['sha256'],'base prompt-byte prefix changed'
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);adapter=runpy.run_path(str(a.repo/'scripts/prepare_data.py'))['process_ultrachat_row']
    ids=set();near=set();eligible=[]
    reject=re.compile(r'(?i)(?:\b(?:ransomware|bioweapon|exploit payload|reverse shell|shellcode|pipe bomb|child porn|chemical weapon|make a bomb|build a bomb|malware|how to kill)\b|<\|(?:im_start|im_end|endoftext)\|>)')
    for fname,expected in SHARDS.items():
        path=a.ultra/fname;assert sha(path)==expected
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=['messages','prompt_id','prompt']):
            for source in batch.to_pylist():
                counts['ultra_source_rows']+=1;rid='ultrachat/'+str(source['prompt_id'])
                if rid in ids:counts['duplicate_family']+=1;continue
                ids.add(rid)
                try:converted,_=adapter(source);first=converted['conversations'][0];text=first['content']
                except (ValueError,KeyError,IndexError,TypeError):continue
                if first['role']!='user' or not isinstance(text,str):continue
                h,nh=rowhash(text)
                if int(nh[:8],16)%100<50:counts['reserved_original_half']+=1;continue
                if h in ban or (isinstance(source.get('prompt'),str) and rowhash(source['prompt'])[0] in ban):counts['eval_denied']+=1;continue
                if h in crossexact or nh in crossnear:counts['blocked_sharegpt']+=1;continue
                if nh in near:counts['duplicate_prompt']+=1;continue
                near.add(nh)
                if not 12<=len(text)<=12000 or not text.strip() or reject.search(text):continue
                eligible.append((digest(f'{SEED}:{rid}:{nh}'),rid,h,nh,text))
    eligible.sort();bins=[[],[],[]]
    for i in range(0,len(eligible),256):
        batch=eligible[i:i+256];enc=tok.apply_chat_template([[{'role':'user','content':r[-1]}] for r in batch],tokenize=True,add_generation_prompt=True,return_dict=False)
        for (rank,rid,h,nh,_),tokens in zip(batch,enc):
            n=len(tokens)
            if not 12<=n<=896:counts['outside_token_window']+=1;continue
            k=0 if n<=256 else 1 if n<=576 else 2
            bins[k].append(dict(id=rid,family=rid,prompt_sha256=h,near_sha256=nh,split='train',group=('short','medium','long')[k],prompt_ids=tokens,
                               max_new_tokens=640 if int(rank[8:16],16)%4==0 else 384))
    supplement=interleave(bins,sum(map(len,bins))); rows=existing+supplement;dest=a.out/'train-tail.jsonl';nbase=b['selections']['train']['count']
    with dest.open('w') as f:
        for i,row in enumerate(rows,nbase):row['index']=i;f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    manifest=dict(base_manifest_sha256=sha(a.base/'source-manifest.json'),base_train_rows=nbase,base_train_sha256=base_sha.hexdigest(),unchanged_train_prefix=True,
      sharegpt_eligible_train=nbase+len(existing),sharegpt_eligible_heldout=json.loads((a.eligible/'source-manifest.json').read_text())['selections']['heldout']['count'],
      sharegpt_original_train=8192,sharegpt_tail_rows=len(existing),ultrachat_source='HuggingFaceH4/ultrachat_200k',ultrachat_split='train_sft',ultrachat_revision=ULTRA_REV,
      source_shards=SHARDS,ultrachat_counts=dict(counts),ultrachat_eligible_train=len(supplement),ultrachat_groups=[len(x) for x in bins],
      blocked= json.loads((a.block/'manifest.json').read_text()),blocked_eval_sha256=DENY_SHA,total_train_candidates=nbase+len(rows),
      tail_rows=len(rows),tail_sha256=sha(dest),schedule='all remaining eligible ShareGPT candidates, then UltraChat first-user prompts in half-buckets 50..99, ordered by seeded length strata')
    assert manifest['total_train_candidates']>=35000
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
