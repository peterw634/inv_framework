"""Compare diffusion reconstruction algorithms using an explicit JSON case.

CPU smoke: python examples/diffusion_ct_compare.py --smoke --output outputs/diffusion-smoke
See docs/diffusion_methods.md for real weights, measurement units and case fields.
"""
import argparse
import gc
import inspect
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inv_framework.solvers import diffusion as ds
from inv_framework.models.latent_codec import VQCodec
from inv_framework.operators.sirt_projection import SIRTProjectionOperator
from inv_framework.utils.reconstruction_comparison import model_measurement, reconstruction_metrics

ALGORITHMS = {'mcg':ds.MCGSolver, 'pgdm':ds.PGDMSolver, 'dds':ds.DDIMDDSSolver,
 'dmplug':ds.DMPlugSolver,'reddiff':ds.REDDiffSolver,'hybridreg':ds.HybridRegSolver,
 'psld':ds.PSLDSolver,'resample':ds.ReSampleSolver,'diffstategrad':ds.DiffStateGradSolver}
METHODS = ['dps', *ALGORITHMS, 'meanflow', 'meanflow_lora']
LATENT = {'psld','resample','diffstategrad'}


@lru_cache(maxsize=16)
def weight_identity(location):
    path = Path(location).resolve()
    files = ([path, path.with_suffix(".json")] if path.is_file() else
             sorted(path.rglob("*.safetensors")) + sorted(path.rglob("*.json")))
    result = {}
    for file in files:
        if not file.is_file():
            continue
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        result[str(file)] = {"bytes": file.stat().st_size, "sha256": digest.hexdigest()}
    return result


def effective_parameters(solver):
    sampling = getattr(solver, "_sample", None)
    fn = sampling if sampling else type(solver).__init__
    result = {}
    for name, parameter in inspect.signature(fn).parameters.items():
        value = parameter.default if sampling else getattr(solver, name, parameter.default)
        if value is None or isinstance(value, (bool, str, float, int)):
            if name not in ("measurement", "generator", "batch_size"):
                result[name] = value
    result.update(getattr(solver, "parameters", {}))
    cond = getattr(solver, "conditioning", None)
    if cond is not None:
        result["conditioning"] = type(cond).__name__
        result["scale"] = cond.scale
    elif hasattr(solver, "scale"):
        result["scale"] = solver.scale
    return result


def build_solver(name, config, operator, device, smoke=False):
    params = dict(config.get('parameters', {}))
    if name.startswith('meanflow'):
        if smoke:
            prior=ds.TinyVelocityPrior(img_size=operator.domain_shape[-1],base=8).to(device)
        else:
            from inv_framework.applications.nasn_ct.weights import load_pmf
            pmf=load_pmf(config['weights'],config.get('model_config'),device)
            prior=ds.PMFVelocityPrior(pmf)
        if prior.img_size != operator.domain_shape[-1]:
            raise ValueError('mean-flow prior and operator grids differ')
        if name=='meanflow_lora':
            if smoke:groups=ds.inject_lora_linear(prior.net,r=config.get('lora_rank',4))
            else:groups=ds.adapt_pmfdit(prior.net,r=config.get('lora_rank',8),verbose=False)
            lr=params.pop("adapt_lr",1e-4)
            solver=ds.LoRAAdaptiveDDSSolver(prior,adapt_params=[p for group in groups for p in group],
                adapt_optim_factory=lambda p:torch.optim.Adam(p,lr=lr),**params)
        else:solver=ds.MeanFlowDDSSolver(prior,**params)
        return solver, prior, operator
    from diffusers import UNet2DModel,VQModel,DDIMScheduler,DDPMScheduler
    kind=DDPMScheduler if name in ('dps','mcg','psld') else DDIMScheduler
    codec=None
    if smoke:
        size=operator.domain_shape[-1]
        net=UNet2DModel(sample_size=size,in_channels=1,out_channels=1,layers_per_block=1,
            block_out_channels=(8,),down_block_types=('DownBlock2D',),up_block_types=('UpBlock2D',),norm_num_groups=4)
        schedule=kind(num_train_timesteps=60)
        if name in LATENT:
            vq=VQModel(in_channels=1,out_channels=1,latent_channels=1,vq_embed_dim=1,num_vq_embeddings=8,
                block_out_channels=(8,),norm_num_groups=4,sample_size=size,scaling_factor=1)
            codec=VQCodec(vq.to(device).eval())
    else:
        net=UNet2DModel.from_pretrained(config['weights'],subfolder='unet',local_files_only=True)
        schedule=kind.from_pretrained(config['weights'],subfolder='scheduler',local_files_only=True)
        if name in LATENT:
            codec=VQCodec(VQModel.from_pretrained(config['weights'],subfolder='vqvae',local_files_only=True).to(device).eval())
        if 'model_grid' in config:
            # Explicit experiment override for fully convolutional UNets only.
            net.register_to_config(sample_size=config['model_grid'])
    net=net.to(device).eval()
    if name not in LATENT:
        size=net.config.sample_size
        if tuple((size,size) if isinstance(size,int) else size)!=tuple(operator.domain_shape[-2:]):
            raise ValueError('model grid mismatch; an explicit model_grid override is required for a new-resolution experiment')
    A=SIRTProjectionOperator(operator,iterations=config.get('sirt_iterations',100))
    if name=='dps':
        scale=params.pop('scale',1.)
        solver=ds.DPSSolver(ds.DiffusersUNetWrapper(net),ds.DiffusersScheduleAdapter(schedule),
            conditioning=ds.NormalizedPosteriorSampling(A,scale=scale),**params)
    else:solver=ALGORITHMS[name](net,schedule,codec=codec,**params)
    return solver, net, A


def load_case(config, device):
    import tifffile
    data=config['data'];kind=data['kind']
    if kind=='nasn-real':
        from inv_framework.applications.nasn_ct.measurement import TruncatedMeasurement,load_scale_stats
        case=TruncatedMeasurement(data['sinogram_dir'],load_scale_stats(data['scale_stats']))
        return case.operator(device),case.load(data.get('slice',1150),device),None
    if kind=='nasn-synthetic':
        from inv_framework.operators.ct import TomoPyCUDARadon2D
        raw=tifffile.imread(data['reference']).astype(np.float32)
        truth=torch.from_numpy((raw+1)/2)[None,None].to(device)
        A=TomoPyCUDARadon2D(2048,2048,512,1024.,device=device)
        return A,A(truth),truth
    if kind=='measurement':
        from inv_framework.operators.ct import TomoPyCUDARadon2D
        A=TomoPyCUDARadon2D(device=device,**data['geometry'])
        y=torch.from_numpy(tifffile.imread(data['measurement']).astype(np.float32))[None,None].to(device)
        ref=None if 'reference' not in data else torch.from_numpy(tifffile.imread(data['reference']).astype(np.float32))[None,None].to(device)
        return A,y,ref
    raise ValueError('unsupported data kind')


def smoke_case(device):
    from inv_framework.operators.base import LinearOperator
    class DenseProjection(LinearOperator):
        domain_shape=(1,8,8);range_shape=(1,4,8)
        def __init__(self):
            self.matrix=torch.rand(32,64,device=device,generator=torch.Generator(device=device).manual_seed(13)) / 64
        def forward(self,x):return (x.flatten(1)@self.matrix.T).reshape(-1,*self.range_shape)
        def adjoint(self,y):return (y.flatten(1)@self.matrix).reshape(-1,*self.domain_shape)
    A=DenseProjection();truth=torch.linspace(.1,.9,64,device=device).reshape(1,1,8,8)
    return A,A(truth),truth


def run(config, output, device='cpu', smoke=False):
    import tifffile
    device=torch.device(device);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    A,y,reference=smoke_case(device) if smoke else load_case(config,device)
    report={'configuration':config,'smoke_untrained':smoke,'results':{},'versions':{'torch':torch.__version__}}
    import diffusers
    report['versions']['diffusers']=diffusers.__version__
    for name,spec in config['methods'].items():
        if name not in METHODS:raise ValueError(f'unknown method: {name}')
        seed=int(spec.get('seed',config.get('seed',0)))
        scale=float(spec.get('image_scale',1.));offset=float(spec.get('image_offset',0.))
        row={'configuration':spec,'seed':seed,'status':'failed'}
        try:
            torch.manual_seed(seed);np.random.seed(seed)
            solver,model,op=build_solver(name,spec,A,device,smoke)
            row['effective_parameters']=effective_parameters(solver)
            if not smoke:row['checkpoint']=weight_identity(spec['weights'])
            target=model_measurement(y,A,scale,offset)
            nfe=[0]
            def count(*args):nfe[0]+=1
            hook_model=model if isinstance(model,torch.nn.Module) else model.net
            handle=hook_model.register_forward_hook(count)
            if device.type=='cuda':torch.cuda.synchronize(device)
            start=time.perf_counter()
            if name.startswith('meanflow'):
                from inv_framework.applications.nasn_ct.random import BatchGenerator
                noise=BatchGenerator(device,[seed]).randn((1,*A.domain_shape))
                result=solver.solve(target,op,initial_noise=noise)
            else:result=solver.solve(target,op,generator=torch.Generator(device='cpu' if name in LATENT else device).manual_seed(seed))
            if device.type=='cuda':torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-start;handle.remove()
            image=result*scale+offset
            if not torch.isfinite(image).all():raise RuntimeError('reconstruction contains nonfinite values')
            row.update(status='passed',seconds=elapsed,network_evaluations=nfe[0],
                metrics=reconstruction_metrics(image,y,A,reference,config.get('roi_radius')))
            tifffile.imwrite(out/(name+'.tif'),image[0,0].detach().cpu().numpy())
        except Exception as e:
            row['error']=type(e).__name__+': '+str(e)
        finally:
            if 'handle' in locals():handle.remove();del handle
        report['results'][name]=row
        (out/'summary.json').write_text(json.dumps(report,indent=2))
        print(name,row['status'],row.get('error',''),flush=True)
        if 'solver' in locals():del solver
        if 'model' in locals():del model
        if 'result' in locals():del result
        if 'image' in locals():del image
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--device',default='cpu');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not args.smoke and args.case is None:parser.error('--case or --smoke is required')
    config=json.loads(args.case.read_text()) if args.case else {'seed':7,'methods':{
        name:{'parameters':{'num_inference_steps':4,**({'epochs':2} if name=='dmplug' else {})}}
        for name in METHODS}}
    report=run(config,args.output,args.device,args.smoke)
    raise SystemExit(0 if all(r['status']=='passed' for r in report['results'].values()) else 1)
