"""Independently verify a published token file against the original prompt and target's committed answer IDs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for s in iter(lambda:f.read(1<<20),b''):h.update(s)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--stage',required=True);p.add_argument('--vocab',type=Path);a=p.parse_args()
    manifest=json.loads((a.root/(a.stage+'.manifest.json')).read_text());split=manifest['split'];file=a.root/(a.stage+'.jsonl')
    for name,value in manifest['files'].items():assert sha(a.root/name)==value,name
    prompts={i['id']:i for l in (a.root/'prompts'/(split+'-prompts.jsonl')).open() if (i:=json.loads(l))}
    if manifest.get('extension_manifest_sha256') and split=='train':
        ext=a.root/'prompts-extension-v1';assert sha(ext/'manifest.json')==manifest['extension_manifest_sha256']
        with (ext/'train-tail.jsonl').open() as f:
            for line in f:
                i=json.loads(line); assert i['id'] not in prompts;prompts[i['id']]=i
    conn=sqlite3.connect('file:'+str(a.root.resolve()/'regeneration.sqlite3')+'?mode=ro',uri=True,timeout=60)
    labels=None
    if a.vocab:
        import torch
        labels=torch.load(str(a.vocab),map_location='cpu',weights_only=True)['t2d'].tolist(); assert len(labels)==151936 and sum(labels)==32000
    count=Counter();reasons=Counter();by_source={};ids=set();observed_answer_ids=set();previous=-1
    with file.open() as f,(a.root/(a.stage+'.audit.jsonl')).open() as af:
        for row,audit in zip(f,af,strict=True):
            row,audit=json.loads(row),json.loads(audit);rid=row['id'];source=prompts[rid]; full=row['input_ids'];mask=row['loss_mask'];offset=audit['prompt_tokens'];prefix=full[:offset];answer=full[offset:];i=source['index']
            assert rid not in ids and rid==audit['id'] and previous<i; ids.add(rid);previous=i
            stored,finish,port=conn.execute('select answer,finish,port from generations where split=? and i=? and id=? and kept=1',(split,i,rid)).fetchone()
            assert prefix==source['prompt_ids'] and answer==json.loads(stored) and mask==[0]*len(prefix)+[1]*len(answer)
            assert sha_token_ids(answer)==audit['answer_ids_sha256'] and finish==audit['finish_reason'] and len(answer)==audit['answer_tokens'] and 8<=len(answer)<=source['max_new_tokens']
            assert len(full)<=1536 and source['prompt_sha256']==audit['prompt_sha256'] and source['family']==audit['family'];reasons[finish]+=1
            count['rows']+=1;count['tokens']+=len(full);count['supervised']+=len(answer);count[f'port_{port}_rows']+=1
            observed_answer_ids.update(answer)
            population='UltraChat train_sft user prompts' if rid.startswith('ultrachat/') else 'ShareGPT user prompts'
            sub=by_source.setdefault(population,Counter());sub['rows']+=1;sub['total_tokens']+=len(full);sub['supervised_target_tokens']+=len(answer);sub[finish+'_rows']+=1
            if labels is not None: count['mapped_answer_tokens']+=sum(labels[t] for t in answer)
    assert count['rows']==manifest['rows'] and count['supervised']==manifest['supervised_tokens'] and count['tokens']==manifest['tokens'] and dict(reasons)==manifest['finish']
    out=dict(passed=True,stage=a.stage,data_sha256=sha(file),counts=dict(count),finish_counts=dict(reasons),by_prompt_source=by_source,distinct_answer_token_ids=len(observed_answer_ids),last_reserved_index=previous)
    if labels is not None:out.update(fixed_vocab_sha256=sha(a.vocab),mapped_distinct_answer_token_ids=sum(labels[t] for t in observed_answer_ids),answer_token_coverage=count['mapped_answer_tokens']/count['supervised'])
    print(json.dumps(out,indent=2))


def sha_token_ids(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
if __name__=='__main__':main()
