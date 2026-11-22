"""Build a cross-corpus exclusion list from all eligible ShareGPT training/holdout and bootstrap rows."""
import argparse
import json
from pathlib import Path
from prepare_prompts import rows,rowhash,sha


def main():
    p=argparse.ArgumentParser();
    for key in ('original','eligible','source','smoke','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True);exact=set();near=set();counts={}
    for name,path in [('original',a.original),('train',a.eligible/'train-prompts.jsonl'),('heldout',a.eligible/'heldout-prompts.jsonl')]:
        n=0
        for l in path.open():
            r=json.loads(l);exact.add(r.get('normalized_sha256') or r['prompt_sha256']);near.add(r['near_sha256']);n+=1
        counts[name]=n
    smoke=json.loads(a.smoke.read_text());left=set(smoke['train_ids']+smoke['held_out_ids']);counts['smoke']=len(left)
    for r in rows(a.source):
        if r['id'] in left:
            text=r['conversations'][0]['value'];h,nh=rowhash(text);exact.add(h);near.add(nh);left.remove(r['id'])
        if not left:break
    assert not left
    for label,values in [('normalized',exact),('near',near)]: (a.out/(label+'.sha256')).write_text('\n'.join(sorted(values))+'\n')
    info=dict(counts=counts,normalized=len(exact),near=len(near),files={k:sha(a.out/k) for k in ['normalized.sha256','near.sha256']});(a.out/'manifest.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))
if __name__=='__main__':main()
