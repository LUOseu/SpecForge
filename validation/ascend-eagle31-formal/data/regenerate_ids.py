"""Durable local-SGLang regeneration, with native token IDs as the training contract."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED,ThreadPoolExecutor,wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import urllib.parse
import urllib.request

STAGES=(16,128,512,1024,2048,4096,8192,12288,16384,24576,32768,40960,49152,57344)
TARGET_CONFIG='a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438'
TOKENIZER_CONFIG='a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3'


def serialized(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for s in iter(lambda:f.read(1<<20),b''):h.update(s)
    return h.hexdigest()


def atomic(path,value):
    tmp=path.with_name('.'+path.name+'.tmp')
    with tmp.open('w') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def request(url,data=None,timeout=10):
    body=None if data is None else serialized(data).encode()
    req=urllib.request.Request(url,data=body,headers={'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req,timeout=timeout) as r:return json.load(r)


def server(url):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url+'/health',timeout=3) as health:
        if health.status!=200:raise OSError('target health is not ready')
    info=request(url+'/server_info',timeout=3)
    model=Path(info['model_path']); configuration=sha(model/'config.json');tokenizer=sha(Path(info.get('tokenizer_path') or model)/'tokenizer_config.json')
    if not (configuration==TARGET_CONFIG and tokenizer==TOKENIZER_CONFIG and info['device']=='npu' and info['dtype']=='bfloat16'
            and not info.get('speculative_algorithm') and info['context_length']>=1536):
        raise ValueError('Unexpected model, tokenizer, speculative setting, dtype, device or context length')
    return {k:info.get(k) for k in ('device','dtype','tp_size','context_length','max_running_requests','attention_backend','speculative_algorithm','version')}


def generate(url,row):
    then=time.monotonic(); prompt=row['prompt_ids']; cap=row['max_new_tokens']
    data=request(url+'/generate',{'input_ids':prompt,'sampling_params':{'temperature':0,'max_new_tokens':cap,'ignore_eos':False},'stream':False},timeout=900)
    out=data['output_ids'];meta=data['meta_info'];reason=meta['finish_reason']['type']
    if not (isinstance(out,list) and 0<len(out)<=cap and all(type(t) is int and 0<=t<151936 for t in out) and
            len(out)==meta['completion_tokens'] and meta['prompt_tokens']==len(prompt) and not meta.get('spec_verify_ct') and reason in ('stop','length') and len(prompt)+len(out)<=1536):
        raise ValueError(f'invalid token boundary for {row["split"]} {row["index"]}')
    return out,reason,time.monotonic()-then


def stages(db,prompts,completed,out,desired,tail_identity=None):
    for split,limits in [('train',[n for n in STAGES if n<=desired['train']]),('heldout',[desired['heldout']])]:
        # Completion order depends on batching. Publish only a complete prefix of reserved prompts.
        end=0
        while (split,end) in completed:end+=1
        rows=db.execute('select i,answer,finish from generations where split=? and i<? and kept=1 order by i',(split,end))
        selections=None;previous=0
        for n in limits:
            name=f'{split}-first{n:04d}'; manifest=out/(name+'.manifest.json')
            if manifest.exists():previous=n;continue
            if end<n:break
            if selections is None: selections=list(rows)
            if len(selections)<n:break
            dest=out/(name+'.jsonl');delta=out/f'{split}-slice{previous:04d}-{n:04d}.jsonl'
            tmp=dest.with_suffix('.tmp'); dt=delta.with_suffix('.tmp');token_count=0;supervised=0; stops=Counter();groups=Counter();lengths=[]
            audit=out/(name+'.audit.jsonl');at=audit.with_suffix('.tmp');selected_hashes=set()
            with tmp.open('w') as f,dt.open('w') as g,at.open('w') as a:
                for idx,(i,answer,finish) in enumerate(selections[:n]):
                    row=prompts[split][i];answer=json.loads(answer);pref=row['prompt_ids'];ids=pref+answer;mask=[0]*len(pref)+[1]*len(answer)
                    assert mask.index(1)==len(pref) and ids[:len(pref)]==pref and ids[len(pref):]==answer and len(answer)>=8 and len(mask)==len(ids)
                    assert row['near_sha256'] not in selected_hashes;selected_hashes.add(row['near_sha256'])
                    line=serialized(dict(id=row['id'],input_ids=ids,loss_mask=mask))+'\n';f.write(line)
                    if idx>=previous:g.write(line)
                    a.write(serialized(dict(id=row['id'],family=row['family'],prompt_sha256=row['prompt_sha256'],near_sha256=row['near_sha256'],prompt_tokens=len(pref),answer_tokens=len(answer),finish_reason=finish,answer_ids_sha256=hashlib.sha256(serialized(answer).encode()).hexdigest()))+'\n')
                    token_count+=len(ids);supervised+=len(answer);groups[row['group']]+=1;stops[finish]+=1;lengths.append(len(ids))
                for file in (f,g,a):file.flush();os.fsync(file.fileno())
            for temp,destination in [(tmp,dest),(dt,delta),(at,audit)]:os.replace(temp,destination)
            atomic(manifest,dict(split=split,rows=n,visited_source_prefix=selections[n-1][0]+1,tokens=token_count,supervised_tokens=supervised,
              finish=dict(stops),prompt_groups=dict(groups),sequence_min=min(lengths),sequence_p50=sorted(lengths)[len(lengths)//2],sequence_max=max(lengths),
              prompt_source='first-user-only',answer_source='target-only greedy SGLang output_ids',temperature=0,max_new_tokens=[384,640],minimum_answer_tokens=8,
              source_manifest_sha256=sha(out/'prompts/source-manifest.json'),extension_manifest_sha256=tail_identity,files={p.name:sha(p) for p in (dest,delta,audit)}))
            print('released',name,'supervised',supervised,flush=True);previous=n


def run(args):
    out=args.root;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'regeneration.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for url in args.servers:
        parsed=urllib.parse.urlparse(url)
        if parsed.scheme!='http' or parsed.hostname not in ('localhost','127.0.0.1','::1') or parsed.path not in ('','/'):
            raise ValueError('generation endpoint must be a local HTTP server')
    cfg=json.loads((out/'prompts/source-manifest.json').read_text()); prompts={}
    for split in ['train','heldout']:
        file=out/'prompts'/(split+'-prompts.jsonl');assert sha(file)==cfg['selections'][split]['sha256'];prompts[split]=[json.loads(x) for x in file.open()]
    tail_identity=None
    if args.extension:
        ext=args.extension; meta=json.loads((ext/'manifest.json').read_text());tail_identity=sha(ext/'manifest.json')
        assert meta['base_manifest_sha256']==sha(out/'prompts/source-manifest.json') and meta['base_train_sha256']==cfg['selections']['train']['sha256']
        assert meta['base_train_rows']==len(prompts['train']) and sha(ext/'train-tail.jsonl')==meta['tail_sha256']
        with (ext/'train-tail.jsonl').open() as f:
            for line in f:
                r=json.loads(line);assert r['split']=='train' and r['index']==len(prompts['train']);prompts['train'].append(r)
        assert len(prompts['train'])==meta['total_train_candidates'] and args.train in STAGES and args.train<=len(prompts['train'])
    db=sqlite3.connect(out/'regeneration.sqlite3',timeout=60)
    if tail_identity:
        db.execute('create table if not exists extension_identity(sha text primary key)');db.execute('insert or ignore into extension_identity values(?)',(tail_identity,));assert list(db.execute('select sha from extension_identity'))==[(tail_identity,)];db.commit()
    db.execute('pragma journal_mode=delete');db.execute('pragma synchronous=full');db.execute('create table if not exists generations(split text,i integer,id text,answer text,finish text,kept integer,port integer,elapsed real,ntokens integer,primary key(split,i))')
    db.execute('create table if not exists identity(source text primary key)');identity=sha(out/'prompts/source-manifest.json');db.execute('insert or ignore into identity values(?)',(identity,));assert list(db.execute('select source from identity'))==[(identity,)];db.commit()
    completed={(s,i) for s,i in db.execute('select split,i from generations')}; kept=Counter(dict(db.execute('select split,sum(kept) from generations group by split')))
    desired={'train':args.train,'heldout':args.heldout};stages(db,prompts,completed,out,desired,tail_identity)
    pos=dict(train=0,heldout=0);flying={};pending_keys=set();retry_at={};failures=Counter();healthy={};ever={};busy=Counter();byp=Counter();last_probe=-100.;last_write=0.;started=time.monotonic();first_request=None;last_stats=started;before_tokens=0;new_tokens=0;submitted=0
    def next_index(split):
        i=pos[split]
        while (split,i) in completed or (split,i) in pending_keys or retry_at.get((split,i),0)>time.monotonic():i+=1
        if i>=len(prompts[split]):return None
        return i
    with ThreadPoolExecutor(max_workers=max(1,args.per_server*len(args.servers))) as pool:
        while any(kept[s]<desired[s] for s in desired) or flying:
            now=time.monotonic()
            # The second replica returns to evaluation once the first 512 regenerated
            # training rows are durable. A requested drain never cancels work in flight.
            applied=out/'auto-drain-30101-stage0512.json'
            if (out/'train-first0512.manifest.json').exists() and not applied.exists():
                (out/'drain-30101.requested').touch()
                atomic(applied,dict(stage=512,time_unix=time.time()))
            draining=set()
            for url in args.servers:
                port=urllib.parse.urlparse(url).port
                if (out/f'drain-{port}.requested').exists():
                    draining.add(url)
                    if not busy[url]:
                        healthy.pop(url,None)
                        ack=out/f'drained-{port}.json'
                        if not ack.exists():
                            db.commit()
                            rows,tokens=db.execute('select count(*),coalesce(sum(ntokens),0) from generations where port=?',(port,)).fetchone()
                            atomic(ack,dict(port=port,outstanding=0,committed_rows=rows,output_tokens=tokens,time_unix=time.time()))
                            print('drained',port,rows,flush=True)
            if now-last_probe>=10:
                for url in args.servers:
                    if url in healthy or url in draining:continue
                    try:info=server(url)
                    except (OSError,TimeoutError,KeyError,json.JSONDecodeError):continue
                    healthy[url]=min(args.per_server,int(info.get('max_running_requests') or args.per_server));ever[url]=info;print('ready',url,healthy[url],flush=True)
                last_probe=now
            for url,capacity in list(healthy.items()):
                if url in draining:continue
                while busy[url]<capacity:
                    splits=['heldout','train'] if kept['train']>=512 and (submitted%6==0 or kept['train']>=desired['train']) else ['train','heldout']
                    split=next((s for s in splits if kept[s]+byp[s]<desired[s] and (s!='heldout' or kept['train']>=min(512,desired['train']))),None)
                    if split is None:break
                    i=next_index(split)
                    if i is None:raise RuntimeError('exhausted reserved '+split+' prompts')
                    if first_request is None:first_request=time.monotonic()
                    future=pool.submit(generate,url,prompts[split][i]);flying[future]=(split,i,url);pending_keys.add((split,i));busy[url]+=1;byp[split]+=1;submitted+=1
            finished=wait(flying,timeout=3,return_when=FIRST_COMPLETED)[0] if flying else set()
            if not flying and not finished:time.sleep(2)
            for future in finished:
                split,i,url=flying.pop(future);busy[url]-=1;byp[split]-=1;pending_keys.remove((split,i))
                try:answer,reason,elapsed=future.result()
                except Exception as exc:
                    failures[split,i]+=1;retry_at[split,i]=time.monotonic()+min(60,2**min(6,failures[split,i]));healthy.pop(url,None)
                    print('retry',split,i,end=' ',flush=True);print(type(exc).__name__,str(exc)[:180],flush=True)
                    if failures[split,i]>=8:raise RuntimeError(f'repeated failure at {split}/{i}') from exc
                    continue
                retained=int(len(answer)>=8);row=prompts[split][i];completed.add((split,i));kept[split]+=retained;new_tokens+=len(answer)
                db.execute('insert into generations values(?,?,?,?,?,?,?,?,?)',(split,i,row['id'],serialized(answer),reason,retained,urllib.parse.urlparse(url).port,elapsed,len(answer)))
                while (split,pos[split]) in completed:pos[split]+=1
            if now-last_write>=2 or not flying:
                db.commit();stages(db,prompts,completed,out,desired,tail_identity);last_write=now
            if now-last_stats>=20:
                snapshot=dict(kept=dict(kept),pending=len(flying),counts_by_endpoint={str(p):dict(rows=n,tokens=t,mean_answer_tokens=round(t/n,2)) for p,n,t in db.execute('select port,count(*),sum(ntokens) from generations group by port')},
                 elapsed_process_s=round(now-started,2),elapsed_since_first_submission_s=round(now-first_request,2) if first_request is not None else None,
                 output_toks_per_s_since_first_submission=round(new_tokens/max(1,now-first_request),3) if first_request is not None else 0,output_toks_per_s_since_start=round(new_tokens/max(1,now-started),3),last_window_toks_per_s=round((new_tokens-before_tokens)/max(1,now-last_stats),3),
                 ready={urllib.parse.urlparse(u).port:inf for u,inf in ever.items()},time_unix=time.time())
                atomic(out/'regeneration-progress.json',snapshot);print('progress',serialized(snapshot),flush=True);last_stats=now;before_tokens=new_tokens
    db.commit();stages(db,prompts,completed,out,desired,tail_identity)
    total=dict(db.execute('select split,sum(ntokens) from generations group by split'))
    result=dict(status='complete',requested=desired,kept=dict(kept),outstanding_http_futures=len(flying),worker_pid=os.getpid(),
                output_tokens=total,elapsed_process_seconds=time.monotonic()-started,new_tokens_this_process=new_tokens)
    assert not flying and dict(kept)==desired
    atomic(out/f'regeneration-done-{args.train}.json',result);atomic(out/'regeneration-done.json',result);db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--extension',type=Path);p.add_argument('--servers',nargs='+',required=True);p.add_argument('--per-server',type=int,default=24);p.add_argument('--train',type=int,default=8192);p.add_argument('--heldout',type=int,default=128);run(p.parse_args())
