"""Algorithm and public-interface checks independent of external checkouts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import pytest
from inv_framework.operators.base import LinearOperator
from inv_framework.operators.sirt_projection import SIRTProjectionOperator
from inv_framework.utils.reconstruction_comparison import model_measurement
from inv_framework.solvers.diffusion import DDSSolver, MeanFlowDDSSolver

class Matrix(LinearOperator):
    domain_shape = range_shape = (1,2,2)
    def __init__(self):self.mat=torch.tensor([[2.,.1,0,0],[.2,1.,0,0],[0,0,3.,.1],[0,0,.2,2.]])
    def forward(self,x):return (x.flatten(1)@self.mat.T).reshape(-1,1,2,2)
    def adjoint(self,y):return (y.flatten(1)@self.mat).reshape(-1,1,2,2)


def test_affine_measurement_preserves_linear_problem():
    A=Matrix();x=torch.rand(2,1,2,2)
    y=A(.5*x+.25)
    torch.testing.assert_close(model_measurement(y,A,.5,.25),A(x))


def test_sirt_projection_and_inverse_against_dense_reference():
    A=Matrix();op=SIRTProjectionOperator(A,iterations=100)
    x=torch.rand(2,1,2,2);y=A(x)
    start=torch.rand_like(x)
    R=1/A.mat.sum(1);C=1/A.mat.sum(0)
    expected=start.flatten(1)-( (start.flatten(1)@A.mat.T-y.flatten(1))*R)@A.mat*C
    torch.testing.assert_close(op.project(start,y).flatten(1),expected)
    torch.testing.assert_close(op.pseudo_inverse(y),x,atol=1e-5,rtol=1e-5)


def test_core_import_does_not_load_optional_packages():
    code='import sys; from inv_framework.solvers import diffusion; assert "diffusers" not in sys.modules; assert "astra" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],check=True)
    assert MeanFlowDDSSolver is DDSSolver


def test_all_methods_and_meanflow_comparison(tmp_path):
    pytest.importorskip('diffusers')
    from examples.diffusion_ct_compare import run,METHODS
    config={'seed':7,'methods':{name:{'parameters':{'num_inference_steps':4,**({'epochs':2} if name=='dmplug' else {})}} for name in METHODS}}
    result=run(config,tmp_path,smoke=True)
    assert all(r['status']=='passed' for r in result['results'].values()),result['results']
    assert all(r['network_evaluations']>0 for r in result['results'].values())
    assert len(list(tmp_path.glob('*.tif')))==12


def test_reference_deep_parity():
    reference=os.environ.get('DIFFUSION_REFERENCE')
    if not reference:pytest.skip('set DIFFUSION_REFERENCE to an external reference checkout')
    pytest.importorskip('diffusers')
    from scripts.verify_diffusion_methods import run
    result=run(reference,deep=True)
    assert all(r['passed'] for r in result.values()),result
    assert result['ReSample']['scheduler_steps'][0]>30
    assert result['DiffStateGrad']['scheduler_steps'][0]>30


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA regression')
def test_unet_adapter_moves_cpu_timesteps_to_model_device():
    pytest.importorskip('diffusers')
    from diffusers import UNet2DModel
    from inv_framework.solvers.diffusion import DiffusersUNetWrapper
    net=UNet2DModel(sample_size=8,in_channels=1,out_channels=1,layers_per_block=1,
        block_out_channels=(8,),down_block_types=('DownBlock2D',),up_block_types=('UpBlock2D',),norm_num_groups=4).cuda().eval()
    x=torch.randn(1,1,8,8,device='cuda')
    with torch.no_grad():
        torch.testing.assert_close(DiffusersUNetWrapper(net)(x,torch.tensor(2)),net(x,torch.tensor(2)).sample)


def test_dps_generator_controls_builtin_stochastic_schedule():
    from inv_framework.solvers.diffusion import DPSSolver, VPSchedule
    class Noise(torch.nn.Module):
        def forward(self,x,t):return .1*x
    A=Matrix();y=torch.ones(1,1,2,2)*.3
    solver=DPSSolver(Noise(),VPSchedule(num_train_timesteps=20),num_inference_steps=4,eta=1.,scale=.1)
    first=solver.solve(y,A,generator=torch.Generator().manual_seed(99))
    torch.randn(100)
    second=solver.solve(y,A,generator=torch.Generator().manual_seed(99))
    torch.testing.assert_close(first,second,rtol=0,atol=0)
