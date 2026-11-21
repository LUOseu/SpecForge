"""Run one instrumented request after the uninstrumented, matched benchmark releases devices.

Run with the same native SGLang interpreter and its environment, with a fresh --out.
The benchmark manifest supplies the successful EAGLE3.1 command. The diagnostic overrides
only its port and graph capture, launches private workers, and tears down its own process group.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import requests


def expected_hashes(serving):
    import torch
    from safetensors import safe_open
    names={'fc_norm.0.weight','fc_norm.1.weight','fc_norm.2.weight','norm.weight'}
    values={}
    for shard in serving.glob('*.safetensors'):
        with safe_open(shard,framework='pt',device='cpu') as f:
            for name in names.intersection(f.keys()):
                weights=f.get_tensor(name).contiguous().view(torch.uint8).numpy()
                values[name]=hashlib.sha256(weights).hexdigest()
    assert set(values)==names
    return values


def summarize(events, hashes):
    reports=[]
    workers=[e for e in events if e['event']=='constructed']
    assert workers
    for rank in sorted({e['rank'] for e in workers}):
        rank_events=[e for e in events if e['rank']==rank]
        norms={}
        for key,digest in hashes.items():
            loads=[e for e in rank_events if e.get('key')==key and e['event']=='load']
            assert loads and all(e['device']=='npu' and e['source_sha256']==e['destination_sha256']==digest for e in loads),(rank,key,'load')
            data=[e for e in rank_events if e.get('key')==key and e['event']=='norm']
            assert data and all(e['finite'] and e['device']=='npu' and e['width']==2048 for e in data),(rank,key,'forward')
            delta=max(e['max_change'] for e in data)
            error=max(e['ref_relative_l2'] for e in data)
            assert delta>0 and error<.02,(rank,key,delta,error)
            norms[key]=dict(request_calls=len(data),sampled_rows=sum(e['rows'] for e in data),
                            loader_calls=len(loads),weight_sha256=digest,max_input_output_change=delta,
                            max_relative_l2_vs_cpu_formula=error,finite=True)
        feedback=[e for e in rank_events if e['event']=='feedback']
        assert feedback and all(e['norm_output'] and e['identical_return_object'] for e in feedback)
        reports.append(dict(tp_rank=rank,norms=norms,feedback_calls=len(feedback),all_aux_returned_normalized_head_input=True))
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-script',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,default=30973)
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('e31_benchmark_helper',args.benchmark_script)
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    manifest=json.loads(args.manifest.read_text())
    cmd=list(manifest['commands']['eagle31']); assert cmd[1:3]==['-m','sglang.launch_server']
    target=Path(cmd[cmd.index('--model-path')+1]); draft=Path(cmd[cmd.index('--speculative-draft-model-path')+1]); hashes=expected_hashes(draft)
    assert cmd[cmd.index('--tp-size')+1]=='2' and cmd[cmd.index('--speculative-algorithm')+1]=='EAGLE3'
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='0,1'
    cmd[cmd.index('--port')+1]=str(args.port)
    if '--disable-cuda-graph' not in cmd:
        cmd.append('--disable-cuda-graph')
    args.out.mkdir(mode=0o700,parents=True,exist_ok=False)
    os.environ['E31_NORM_LOGDIR']=str(args.out.resolve())
    os.environ['PYTHONPATH']=str(Path(__file__).resolve().parent)+os.pathsep+os.environ.get('PYTHONPATH','')
    os.environ.pop('SPECFORGE_SPEC_CAPTURE',None)
    base=f'http://127.0.0.1:{args.port}'
    session=requests.Session();session.trust_env=False
    def generate(prompt,steps):
        r=session.post(base+'/generate',json={'text':prompt,'sampling_params':{'temperature':0,'max_new_tokens':steps,'ignore_eos':True}},timeout=300)
        r.raise_for_status();return r.json()['meta_info']
    with bench.server(cmd,args.port,args.out/'private-server.log',session) as info:
        assert info['speculative_draft_model_path']==str(draft) and info['model_path']==str(target) and info['device']=='npu'
        generate('Compute one plus one.',1)
        gate=args.out/'request.active';gate.touch(exist_ok=False)
        try:
            meta=generate('What is two plus two? Please answer in a short sentence.',8)
        finally:
            gate.unlink(missing_ok=True)
        assert int(meta['spec_verify_ct'])>0 and int(meta['completion_tokens'])==8
        events=[json.loads(line) for path in args.out.glob('worker-*.jsonl') for line in path.read_text().splitlines()]
        reports=summarize(events,hashes)
        public=dict(passed=True,device='Ascend910B3',visible_devices='0,1',target='Qwen/Qwen3-30B-A3B-Instruct-2507',
            checkpoint_step=8,sglang_commit='c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0',
            module_sha256='b25e5f839b088ddf50138bce4a3c4569acc31f8203727b9a7e2a53c337287dac',
            dtype=info['dtype'],tp_size=info['tp_size'],attention_backend=info['attention_backend'],
            graph_capture_disabled_for_diagnostic=True, observer_sha256=hashlib.sha256((Path(__file__).parent/'eagle31_norm_observer.py').read_bytes()).hexdigest(),
            completion_tokens=int(meta['completion_tokens']),verification_calls=int(meta['spec_verify_ct']),workers=reports)
        (args.out/'summary.json').write_text(json.dumps(public,indent=2)+'\n')
        print(json.dumps(public,indent=2))


if __name__=='__main__':
    main()
