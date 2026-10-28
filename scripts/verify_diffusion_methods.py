"""Numerical comparison with an external, unmodified reference checkout.

Usage: python scripts/verify_diffusion_methods.py --reference /path/to/reference
Imports reference code only in this verification process, never in the package.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inv_framework.operators.base import LinearOperator
from inv_framework.solvers.diffusion import (
    DPSSolver, MCGSolver, PGDMSolver, DDIMDDSSolver, DMPlugSolver, REDDiffSolver,
    HybridRegSolver, PSLDSolver, ReSampleSolver, DiffStateGradSolver,
    NormalizedPosteriorSampling, DiffusersScheduleAdapter, DiffusersUNetWrapper,
)
from inv_framework.models.latent_codec import VQCodec

METHODS = [
 ('DPS', DPSSolver, 'PosteriorSampling'), ('MCG', MCGSolver, 'ManifoldConstraintGradient'),
 ('PGDM', PGDMSolver, 'PGDM'), ('DDS', DDIMDDSSolver, 'DDS'), ('DMPlug', DMPlugSolver, 'DMPlug'),
 ('RedDiff', REDDiffSolver, 'RedDiff'), ('HybridReg', HybridRegSolver, 'HybridReg'),
 ('PSLD', PSLDSolver, 'PSLD'), ('ReSample', ReSampleSolver, 'Resample'),
 ('DiffStateGrad', DiffStateGradSolver, 'DiffStateGrad')]

class DiagonalOperator(LinearOperator):
    domain_shape = range_shape = (1, 8, 8)
    def forward(self, x): return x * torch.linspace(.7, 1.3, 8, device=x.device)[None,None,None,:]
    def __call__(self, x): return self.forward(x)
    def adjoint(self, y): return self.forward(y)
    transpose = adjoint
    def pseudo_inverse(self, y): return y / torch.linspace(.7, 1.3, 8, device=y.device)[None,None,None,:]
    def project(self,x,y,**kwargs):return x - self.adjoint(self.forward(x)) + self.adjoint(y)


def run(reference, *, deep=False, device='cpu', checkpoint=None, repeat_reference=False, latent_only=False):
    from diffusers import UNet2DModel, VQModel, DDPMScheduler, DDIMScheduler
    sys.path.insert(0, str(Path(reference).resolve()))
    pipes = importlib.import_module('pipelines')
    conds = importlib.import_module('condition_methods')
    scheds = importlib.import_module('schedulers')
    from inv_framework.solvers.diffusion.resample_schedule import ReSampleDDIMScheduler
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    report = {}
    for name, cls, cond in METHODS:
        if checkpoint and ((name in ('PSLD','ReSample','DiffStateGrad')) != latent_only):continue
        latent = name in ('PSLD','ReSample','DiffStateGrad')
        S = DDPMScheduler if name in ('DPS','MCG','PSLD') else DDIMScheduler
        kwargs = {'num_inference_steps': 30 if deep and name in ('ReSample','DiffStateGrad') else 4}
        if name=='DMPlug':
            kwargs['epochs']=2
            if checkpoint:kwargs['num_inference_steps']=1
        if name=='ReSample':kwargs.update(pixel_maxiters=2,latent_maxiters=2)
        arrays, traces, errors = [], [], []
        for migrated in (False, False if repeat_reference else True):
            torch.manual_seed(7)
            if checkpoint:
                unet = UNet2DModel.from_pretrained(checkpoint, subfolder='unet').to(device).eval()
                scheduler = S.from_pretrained(checkpoint, subfolder='scheduler')
            else:
                unet = UNet2DModel(sample_size=8, in_channels=1, out_channels=1, layers_per_block=1,
                    block_out_channels=(8,), down_block_types=('DownBlock2D',),
                    up_block_types=('UpBlock2D',), norm_num_groups=4).to(device).eval()
                scheduler = S(num_train_timesteps=60 if deep else 20)
            unet.requires_grad_(False)
            codec = VQModel(in_channels=1,out_channels=1,latent_channels=1,vq_embed_dim=1,
                num_vq_embeddings=8,block_out_channels=(8,),norm_num_groups=4,
                sample_size=8,scaling_factor=1).to(device).eval() if latent else None
            if checkpoint and latent:
                codec=VQModel.from_pretrained(checkpoint,subfolder='vqvae').to(device).eval()
            if name in ('ReSample','DiffStateGrad'):
                scheduler = (ReSampleDDIMScheduler if migrated else scheds.DDIMSchedulerReSampler).from_config(scheduler.config)
            A = DiagonalOperator()
            if checkpoint:
                size=codec.config.sample_size if latent else unet.config.sample_size
                A.domain_shape=A.range_shape=(1,size,size)
                # Keep a deterministic exact operator for checkpoint parity.
                A.forward=lambda x:x*.9
                A.adjoint=lambda y:y*.9
                A.transpose=A.adjoint
                A.pseudo_inverse=lambda y:y/.9
            y=torch.full((1,*A.range_shape),.3,device=device)
            history=[]
            original_step=scheduler.step
            def traced(*args, **kw):
                value=original_step(*args, **kw)
                history.append(value.prev_sample.detach().cpu().clone())
                return value
            # Preserve the signature required by scheduler dispatch/introspection.
            import functools
            scheduler.step=functools.wraps(original_step)(traced)
            if migrated:
                if name=='DPS':
                    solver=DPSSolver(DiffusersUNetWrapper(unet),DiffusersScheduleAdapter(scheduler),
                        conditioning=NormalizedPosteriorSampling(A),**kwargs)
                else:solver=cls(unet,scheduler,codec=VQCodec(codec) if latent else None,**kwargs)
            else:
                opts={'unet':unet,'scheduler':scheduler}
                if latent:opts['vqvae']=codec
                solver=getattr(pipes,('LDM' if latent else 'DDPM')+'Pipeline'+name)(**opts)
                solver.measurement_condition=getattr(conds,cond)(A,None)
                solver.set_progress_bar_config(disable=True)
            if deep and name=='DiffStateGrad':
                # Bound expensive optimization while retaining both branch formulas.
                for opt in ('pixel_optimization','latent_optimization'):
                    fn=getattr(solver,opt)
                    def bounded(*args,_fn=fn,**kw):
                        kw['max_iters']=2
                        return _fn(*args,**kw)
                    setattr(solver,opt,bounded)
            try:
                g=torch.Generator(device='cpu' if latent else device).manual_seed(7)
                if migrated: result=solver.solve(y,A,generator=g)
                else:result=torch.from_numpy(solver(measurement=y,output_type='np',generator=g,**kwargs).images)
                arrays.append(result.detach().cpu());errors.append(None)
            except Exception as e:
                arrays.append(None);errors.append(type(e).__name__+': '+str(e))
            traces.append(history)
        row={'parameters':kwargs,'reference_repeat':repeat_reference,'reference_error':errors[0],'framework_error':errors[1], 'scheduler_steps':[len(t) for t in traces]}
        if not any(errors):
            row['max_abs_error']=float((arrays[0]-arrays[1]).abs().max())
            row['relative_l2']=float(torch.linalg.vector_norm(arrays[1]-arrays[0])/torch.linalg.vector_norm(arrays[0]).clamp_min(1e-30))
            row['trace_max_abs_error']=max((float((a-b).abs().max()) for a,b in zip(*traces)), default=0.0)
            row['passed']=len(traces[0])==len(traces[1]) and torch.allclose(arrays[0],arrays[1],rtol=1e-5,atol=1e-6) and all(torch.allclose(a,b,rtol=1e-5,atol=1e-6) for a,b in zip(*traces))
        else:row['passed']=False
        report[name]=row
        print(name,json.dumps(row),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--deep',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--checkpoint');p.add_argument('--output');p.add_argument('--repeat-reference',action='store_true');p.add_argument('--latent-only',action='store_true')
    args=p.parse_args();result=run(args.reference,deep=args.deep,device=args.device,checkpoint=args.checkpoint,repeat_reference=args.repeat_reference,latent_only=args.latent_only)
    if args.output:
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    raise SystemExit(0 if all(r['passed'] for r in result.values()) else 1)
