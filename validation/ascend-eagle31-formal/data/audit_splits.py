"""Independently check exact/near prompt and family disjointness without emitting conversations."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('original','teacher','deny'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--original-ultra',type=Path);p.add_argument('--tail',type=Path)
    a=p.parse_args();denied=set(a.deny.read_text().splitlines());groups={}
    files={'original_sharegpt':[a.original], 'teacher_train':[a.teacher/'train-prompts.jsonl'],'teacher_heldout':[a.teacher/'heldout-prompts.jsonl']}
    if a.original_ultra:files['original_ultrachat']=[a.original_ultra]
    if a.tail:files['teacher_train'].append(a.tail)
    for name,paths in files.items():
        vals={'family':set(),'normalized':set(),'near':set()};n=0
        for path in paths:
            for line in path.open():
                row=json.loads(line);vals['family'].add(row['family']);vals['normalized'].add(row.get('normalized_sha256') or row['prompt_sha256']);vals['near'].add(row['near_sha256']);n+=1
        assert all(len(v)==n for v in vals.values()),(name,n,{k:len(v) for k,v in vals.items()});assert not(denied & vals['normalized']);groups[name]=vals
    out={}
    for l,r in itertools.combinations(groups,2):
        out[l+'__'+r]={k:len(groups[l][k]&groups[r][k]) for k in ('family','normalized','near')};assert not any(out[l+'__'+r].values())
    print(json.dumps(dict(passed=True,counts={k:len(v['family']) for k,v in groups.items()},cross_split_overlap=out,
                          denied_sha256=hashlib.sha256(a.deny.read_bytes()).hexdigest(),external_eval_overlap={k:len(denied&v['normalized']) for k,v in groups.items()}),indent=2))
if __name__=='__main__':main()
