"""Read-only forward and parameter-loader observations restricted to the draft norms."""

import functools
import hashlib
import json
import os
from pathlib import Path


def digest(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def emit(event, **fields):
    path=Path(os.environ['E31_NORM_LOGDIR'])/f'worker-{os.getpid()}.jsonl'
    with path.open('a') as out:
        out.write(json.dumps(dict(event=event, **fields),allow_nan=False)+'\n')


def observe(model, module):
    import torch
    from sglang.srt.runtime_context import get_parallel
    rank=get_parallel().tp_rank
    armed=Path(os.environ['E31_NORM_LOGDIR'])/'request.active'
    norms={f'fc_norm.{i}.weight': op for i,op in enumerate(model.model.fc_norm)}
    norms['norm.weight']=model.model.norm
    if len(norms)!=4 or model.model.norm_output is not True:
        raise RuntimeError('The inspected model must be the EAGLE3.1 export')
    for key,op in norms.items():
        weight=op.weight
        loader=getattr(weight,'weight_loader',module.default_weight_loader)
        def load(dest,source,*args,_key=key,_loader=loader,**kwargs):
            result=_loader(dest,source,*args,**kwargs)
            emit('load',rank=rank,key=_key,device=dest.device.type,dtype=str(dest.dtype),
                 source_sha256=digest(source),destination_sha256=digest(dest))
            return result
        weight.weight_loader=load
        # Capture before dispatch: some fused residual+norm operators update inputs in place.
        snapshots=[]
        def before(mod,args,_snap=snapshots):
            if not armed.is_file() or not args[0].numel():
                return
            x=args[0].detach().reshape(-1,args[0].shape[-1])[:2].cpu().float().clone()
            if len(args)>1 and args[1] is not None:
                x+=args[1].detach().reshape(-1,x.shape[-1])[:x.shape[0]].cpu().float()
            _snap.append(x)
        def after(mod,args,result,_key=key,_snap=snapshots):
            if not _snap:
                return
            x=_snap.pop()
            out=result[0] if isinstance(result,tuple) else result
            y=out.detach().reshape(-1,x.shape[-1])[:x.shape[0]].cpu().float()
            ref=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+mod.variance_epsilon)*mod.weight.detach().cpu().float()
            emit('norm',rank=rank,key=_key,rows=x.shape[0],width=x.shape[1],device=out.device.type,dtype=str(out.dtype),
                 input_sha256=digest(x),output_sha256=digest(y),finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
                 max_change=float((x-y).abs().max()),ref_relative_l2=float((y-ref).norm()/ref.norm().clamp_min(1e-12)))
        op.register_forward_pre_hook(before)
        op.register_forward_hook(after)
    def selected(mod,args,result):
        if armed.is_file() and result[0].numel():
            logits_hidden, aux=result
            emit('feedback',rank=rank,norm_output=mod.norm_output,identical_return_object=logits_hidden is aux[0])
    model.model.register_forward_hook(selected)
    emit('constructed',rank=rank,fc_norm_count=len(model.model.fc_norm),norm_output=model.model.norm_output)


def install(module):
    actual=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if actual!='b25e5f839b088ddf50138bce4a3c4569acc31f8203727b9a7e2a53c337287dac':
        raise RuntimeError('SGLang EAGLE3 module does not match the pinned file')
    cls=module.LlamaForCausalLMEagle3
    original=cls.__init__
    @functools.wraps(original)
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        observe(self,module)
    cls.__init__=init
