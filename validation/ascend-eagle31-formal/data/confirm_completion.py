"""Publish the single formal generation service-release marker after all committed data checks."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--train',type=int,default=32768);p.add_argument('--heldout',type=int,default=128);a=p.parse_args();root=a.root
    lock=(root/'completion-monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    goal={'train':a.train,'heldout':a.heldout};marker=root/f'formal-{a.train}.complete.json'
    done_path=root/f'regeneration-done-{a.train}.json';fields=['train','heldout'];names={k:f'{k}-first{goal[k]:04d}' for k in fields}
    while not marker.exists():
        try:
            d=json.loads(done_path.read_text());assert d['status']=='complete' and d['requested']==d['kept']==goal and d['outstanding_http_futures']==0
            cmd=Path(f'/proc/{d["worker_pid"]}/cmdline')
            if cmd.exists() and str(root/'regenerate_ids.py').encode() in cmd.read_bytes():time.sleep(3);continue
            summaries={};hashes={}
            for split in fields:
                base=names[split];audit=root/'evidence'/(base+'.check.json');manifest=root/(base+'.manifest.json');m=json.loads(manifest.read_text());r=json.loads(audit.read_text())
                assert r['passed'] and r['counts']['rows']==m['rows']==goal[split] and r['data_sha256']==m['files'][base+'.jsonl'] and r['counts']['supervised']==m['supervised_tokens']
                hashes[split]=dict(data=r['data_sha256'],manifest=sha(manifest),audit=sha(audit));summaries[split]=dict(accepted=m['rows'],input_tokens=m['tokens'],supervised_tokens=m['supervised_tokens'])
            with sqlite3.connect('file:'+str(root.resolve()/'regeneration.sqlite3')+'?mode=ro',uri=True,timeout=60) as db:
                counts=dict(db.execute('select split,sum(kept) from generations group by split'))
            assert counts==goal
            data=dict(schema_version=1,status='complete',services_may_release=True,accepted=goal,sqlite_accepted=counts,outstanding_http_futures=0,worker_exited=True,final_worker_pid=d['worker_pid'],
                      audit_passed={'train':True,'heldout':True},stage_hashes=hashes,token_counts=summaries,done_sha256=sha(done_path),time_unix=time.time())
            tmp=marker.with_suffix('.tmp');
            with tmp.open('w') as f:f.write(json.dumps(data,indent=2)+'\n');f.flush();os.fsync(f.fileno())
            os.replace(tmp,marker);print(json.dumps(data),flush=True)
        except FileNotFoundError:time.sleep(15)
if __name__=='__main__':main()
