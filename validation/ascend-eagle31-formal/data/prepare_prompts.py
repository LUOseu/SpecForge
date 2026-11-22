"""Freeze first-turn ShareGPT teacher prompts before generation. Private token files are local artifacts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import runpy
import unicodedata

DATA_REV='8b0048ad6ae8c22f46a78c15559dec98feef5539'
SOURCE_SHA='dd5828821b7e707ca3dc4d0de07e2502c3ce278fcf1a74b81a3464f26006371e'
DENY_SHA='a7694a26eb43f6ff7b9d55cdef0b1cd879ed65a2c0308552bf6c42464aa136fa'
TARGET_REV='0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'
TOKENIZER_SHA='aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
SEED=20261121


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def digest(s): return hashlib.sha256(s.encode()).hexdigest()


def rowhash(text):
    n=' '.join(unicodedata.normalize('NFKC',text).casefold().split())
    near=' '.join(re.sub(r'[\W_]+',' ',n,flags=re.UNICODE).split())
    return digest(n),digest(near)


def rows(path):
    dec=json.JSONDecoder(); buf=''; more=True; started=False
    with path.open(encoding='utf8') as f:
        while True:
            buf=buf.lstrip()
            if not started and buf:
                if buf[0]!='[': raise ValueError('source must be a JSON array')
                started=True;buf=buf[1:].lstrip()
            if buf.startswith(','): buf=buf[1:].lstrip()
            if buf.startswith(']'): return
            try:
                row,end=dec.raw_decode(buf)
            except json.JSONDecodeError:
                chunk=f.read(1<<20)
                if not chunk: raise ValueError('truncated source')
                buf+=chunk; continue
            yield row
            buf=buf[end:]


def interleave(bins, limit):
    # 65% short, 25% medium, 10% long until any reservoir is exhausted.
    pattern=[0,1,0,0,2,0,1,0,0,1,0,0,2,0,1,0,0,1,0,0]
    out=[]; indices=[0,0,0]
    while len(out)<limit:
        k=pattern[len(out)%len(pattern)]
        choices=[i for i in [k,0,1,2] if indices[i]<len(bins[i])]
        if not choices: break
        i=choices[0]; out.append(bins[i][indices[i]]);indices[i]+=1
    return out


def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('source','deny','smoke','model','repo','out'):p.add_argument('--'+a,required=True,type=Path)
    p.add_argument('--all-eligible',action='store_true');args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    assert sha(args.source)==SOURCE_SHA and sha(args.deny)==DENY_SHA and sha(args.model/'tokenizer.json')==TOKENIZER_SHA
    denied=set(args.deny.read_text().splitlines());smoke=json.loads(args.smoke.read_text());old_ids=set(smoke['train_ids']+smoke['held_out_ids']);old_families={re.sub(r'_\d+$','',x) for x in old_ids}
    convert=runpy.run_path(str(args.repo/'scripts/prepare_data.py'))['process_sharegpt_row']
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    all_candidates=[]; counts=Counter(); seen_families=set(); seen_near=set(); old_hashes=set()
    excluded=re.compile(r'(?i)(?:\b(?:ransomware|bioweapon|exploit payload|reverse shell|shellcode|pipe bomb|child porn|chemical weapon|make a bomb|build a bomb|malware|how to kill)\b|<\|(?:im_start|im_end|endoftext)\|>)')
    for row in rows(args.source):
        counts['source_rows']+=1
        rid=str(row.get('id',''))
        if not rid.endswith('_0') and rid not in old_ids: continue
        try: transformed,_=convert(row);messages=transformed['conversations'];first=messages[0]
        except (KeyError,TypeError,IndexError):continue
        if first['role']!='user' or not isinstance(first['content'],str):continue
        text=first['content'];h,near=rowhash(text);family=re.sub(r'_\d+$','',rid)
        if rid in old_ids: old_hashes.update((h,near))
        if not rid.endswith('_0') or family in old_families:continue
        if family in seen_families or near in seen_near:continue
        seen_families.add(family);seen_near.add(near)
        if h in denied:counts['denied_eval_source_matches']+=1;continue
        if not 12<=len(text)<=12000 or not text.strip() or excluded.search(text): continue
        bucket=int(near[:8],16)%100
        if bucket<30:continue
        all_candidates.append((digest(f'{SEED}:{family}:{near}'),rid,family,h,near,bucket,text))
    all_candidates.sort(); bins={'train':[[],[],[]],'heldout':[[],[],[]]}
    for start in range(0,len(all_candidates),128):
        batch=all_candidates[start:start+128]; kept=[r for r in batch if r[3] not in old_hashes and r[4] not in old_hashes]
        if not kept:continue
        prompts=tokenizer.apply_chat_template([[{'role':'user','content':r[-1]}] for r in kept],tokenize=True,add_generation_prompt=True,return_dict=False)
        for r,ids in zip(kept,prompts):
            rank,rid,family,h,near,bucket,text=r;n=len(ids)
            if not 12<=n<=896:continue
            split='train' if bucket<95 else 'heldout';k=0 if n<=256 else 1 if n<=576 else 2
            bins[split][k].append(dict(id=rid,family=family,prompt_sha256=h,near_sha256=near,split=split,group=('short','medium','long')[k],prompt_ids=ids,
                                        max_new_tokens=640 if int(rank[8:16],16)%4==0 else 384))
    if not old_ids or len(old_hashes)<len(old_ids):raise ValueError('failed to reconstruct previous smoke source prompts')
    selections={k:interleave(v,sum(map(len,v)) if args.all_eligible else (12800 if k=='train' else 256)) for k,v in bins.items()}
    if len(selections['train'])<9000 or len(selections['heldout'])<160:raise ValueError(f'insufficient data: {[(k,len(v)) for k,v in selections.items()]}')
    summary={}
    for k,selected in selections.items():
        path=args.out/(k+'-prompts.jsonl')
        with path.open('w') as f:
            for i,row in enumerate(selected):row['index']=i;f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        summary[k]=dict(count=len(selected),groups=dict(Counter(r['group'] for r in selected)),max_new_tokens=dict(Counter(r['max_new_tokens'] for r in selected)),sha256=sha(path))
    combined=sum(selections.values(),[])
    for key in ['id','family','prompt_sha256','near_sha256']:assert len({r[key] for r in combined})==len(combined)
    assert not {r['prompt_sha256'] for r in combined}&denied
    (args.out/'source-manifest.json').write_text(json.dumps(dict(source='Aeala/ShareGPT_Vicuna_unfiltered',revision=DATA_REV,source_file=args.source.name,source_sha256=SOURCE_SHA,
      deny_sha256=DENY_SHA,denied_hashes=len(denied),selected_overlap=0,excluded_smoke_families=len(old_families),counts=counts,
      target='Qwen/Qwen3-30B-A3B-Instruct-2507',target_revision=TARGET_REV,tokenizer_sha256=TOKENIZER_SHA,seed=SEED,
      sampling='sha256(seed:family:near), within 65/25/10 token-length strata; near-question hash buckets 0-29 original,30-94 training,95-99 holdout',selections=summary),indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
