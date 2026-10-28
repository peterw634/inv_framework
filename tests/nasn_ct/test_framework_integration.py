"""Contract tests for the migrated application and shared DDS extensions."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from inv_framework.operators.base import LinearOperator
from inv_framework.solvers.diffusion import CGDataConsistency, DDSSolver, MeanFlowSchedule


class Identity(LinearOperator):
    domain_shape = range_shape = (1, 4, 4)

    def forward(self, x):
        return x

    def adjoint(self, y):
        return y


class Velocity:
    noise_scale = 2.0

    def __call__(self, x, t):
        return 0.2 * x + t[:, None, None, None] * 0.1


@pytest.mark.parametrize("steps,t_start,affine", [(1, 1.0, False), (7, .83, False), (16, 1.0, True)])
def test_dds_matches_reference_with_analytic_cg(steps, t_start, affine):
    """Compare against closed-form identity-operator DC, not another solver."""
    torch.manual_seed(12)
    noise, initial, target = [torch.randn(2, 1, 4, 4) for _ in range(3)]
    lam = .7
    model = Velocity()
    scale, offset = (.5, .5) if affine else (1., 0.)
    cond = CGDataConsistency(Identity(), lam=lam, cg_iter=5,
                             image_scale=scale, image_offset=offset)
    solver = DDSSolver(model, num_inference_steps=steps, conditioning=cond, t_start=t_start)
    result = solver.solve(target, Identity(), x_init=initial, initial_noise=noise)
    expected = (1 - t_start) * initial + t_start * noise * model.noise_scale
    grid = torch.linspace(t_start, 0., steps + 1)
    for t, r in zip(grid[:-1], grid[1:]):
        x0 = expected - t * model(expected, t.expand(2))
        dc_anchor = x0 * scale + offset
        dc = (target + lam * dc_anchor) / (1 + lam)
        corrected = (dc - offset) / scale
        implied_noise = (expected - (1 - t) * x0) / t
        expected = (1 - r) * corrected + r * implied_noise
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)


def test_reencoding_avoids_terminal_cancellation():
    schedule = MeanFlowSchedule()
    schedule.set_timesteps(1)
    x = torch.full((1, 1, 4, 4), 1e8)
    schedule.step(torch.zeros_like(x), torch.ones(1), x)
    corrected = torch.full_like(x, .125)
    torch.testing.assert_close(schedule.reencode_x0(corrected), corrected, rtol=0, atol=0)


def test_initial_noise_shape_is_checked():
    with pytest.raises(ValueError, match="initial_noise"):
        DDSSolver(Velocity()).solve(torch.ones(1, 1, 4, 4), Identity(), initial_noise=torch.ones(1))


def test_scale_file_must_match_corrected_center(tmp_path):
    from inv_framework.applications.nasn_ct.measurement import load_scale_stats
    path = tmp_path / "scale.json"
    path.write_text(json.dumps({"center_2560": 1280, "final_scale": 2}))
    with pytest.raises(ValueError, match="1172"):
        load_scale_stats(path)
    path.write_text(json.dumps({"center_2560": 1172, "final_scale": 2}))
    assert load_scale_stats(path) == 2


def test_config_defaults_work_outside_repository(tmp_path):
    from inv_framework.applications.nasn_ct.cli import build_parser
    from inv_framework.applications.nasn_ct.config import load_config
    parser = build_parser()
    args = parser.parse_args(["data", "prepare-target", "--raw-root", str(tmp_path), "--output", str(tmp_path)])
    assert load_config(args.config)["source_center"] == 1172
    manifest = parser.parse_args(["data", "fetch"]).manifest
    assert Path(manifest).is_file()


def test_core_import_does_not_load_optional_nasn_dependencies():
    code = """
import sys
from inv_framework.operators.ct import TomoPyCUDARadon2D
from inv_framework.solvers.diffusion import DDSSolver
assert not any(name.startswith(('nasn_ct', 'inv_framework.models.pmf',
                               'inv_framework.operators.ct.radon_cuda', 'lpips', 'cv2'))
               for name in sys.modules)
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True,
                   cwd=Path(__file__).resolve().parents[2])
