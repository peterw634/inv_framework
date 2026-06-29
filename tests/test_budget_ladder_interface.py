import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.applications.boed_budget_ladder import (
    build_external_job,
    get_scenario,
    list_algorithms,
)
from inv_framework.operators import CallableSubsetForwardOperator
from inv_framework.solvers.diffusion.budget_dps import (
    BudgetSegmentAction,
    BudgetedPosteriorSampling,
    equispaced_subset,
    grad_cost,
    make_equispaced_schedule,
    reformulated_gradient,
)


def test_equispaced_schedule_preserves_full_aperture():
    assert equispaced_subset(pool_size=32, k=8) == (0, 4, 8, 12, 16, 20, 24, 28)
    assert equispaced_subset(pool_size=32, k=8, sub_k=4) == (0, 8, 16, 24)
    schedule = make_equispaced_schedule([8, 4, 2], k=8, pool_size=32)
    assert [len(a.subset) for a in schedule] == [8, 4, 2]
    assert grad_cost(schedule, k=8) == (8 + 4 + 2) / (3 * 8)


def test_reformulated_gradient_variants():
    grad = torch.ones(2, 1, 4, 4)
    drift = reformulated_gradient(grad, t=0.5)
    norms = drift.reshape(2, -1).norm(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, 5.0), atol=1e-5)

    raw = reformulated_gradient(grad, mode="raw_gradient", source_scale=0.25)
    assert torch.allclose(raw, torch.full_like(grad, -0.25))


def test_subset_conditioning_slices_measurements_and_backprops():
    weights = torch.arange(1.0, 5.0)

    def forward_fn(x, source_indices):
        outs = [x * weights[i].to(device=x.device, dtype=x.dtype)
                for i in source_indices]
        return torch.stack(outs, dim=1)

    op = CallableSubsetForwardOperator(
        forward_fn=forward_fn,
        domain_shape=(1, 4, 4),
        range_shape=(4, 1, 4, 4),
        num_sources=4,
    )
    x_prev = torch.ones(1, 1, 4, 4, requires_grad=True)
    x_t = torch.zeros_like(x_prev)
    x0_hat = x_prev * 2.0
    full_y = torch.zeros(1, 4, 1, 4, 4)
    cond = BudgetedPosteriorSampling(
        op, [BudgetSegmentAction(subset=(0, 2))], scale=0.1)

    x_next, residual = cond.apply(
        x_t, x0_hat, full_y, x_prev=x_prev, step_index=0, num_steps=1)

    assert residual.item() > 0.0
    assert x_next.shape == x_t.shape
    assert cond.grad_calls_total == 2


def test_boed_registry_and_command_builder():
    usct = get_scenario("usct")
    assert usct["teacher_k"] == 64
    assert usct["n_segments"] == 6
    names = {a["name"] for a in list_algorithms()}
    assert names == {
        "full_dps",
        "fixed_subset_dps",
        "budgeted_dps",
        "policy_budget_ladder_dps",
        "dps_raw_gradient",
        "dps_normalized_drift",
    }

    job = build_external_job("fttt", task="infer", sample="data_0",
                             lambda_g=5e-5, n_workers=4)
    assert job["scenario"] == "fttt"
    assert job["command"][-2:] == ["--n-workers", "4"]
    assert "12_policy_dps_infer_fttt.py" in job["script"]


if __name__ == "__main__":
    test_equispaced_schedule_preserves_full_aperture()
    test_reformulated_gradient_variants()
    test_subset_conditioning_slices_measurements_and_backprops()
    test_boed_registry_and_command_builder()
    print("All budget-ladder interface tests passed.")
