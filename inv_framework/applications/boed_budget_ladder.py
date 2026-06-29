"""USCT/FTTT budget-ladder DPS application registry.

The heavy domain implementations live in ``Diffusion_Policy``. This module is
the stable backend-facing interface: it documents the two scenarios, exposes
unsupervised DPS algorithm/baseline metadata, and can build/run the existing
inference scripts without the agent needing to know their paths or CLI details.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_DIFFUSION_POLICY_ROOT = Path("/home/caoxiang/Desktop/Diffusion_Policy")


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    display_name: str
    family: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    display_name: str
    modality: str
    equation: str
    inverse_setting: str
    unknown: str
    measurement: str
    operator_kind: str
    domain_shape: Sequence[int]
    range_shape: Sequence[int]
    pool_size: int
    teacher_k: int
    ladder_ks: Sequence[int]
    n_segments: int
    n_steps: int
    default_noise: Mapping[str, Any]
    default_paths: Mapping[str, str]
    scripts: Mapping[str, str]


ALGORITHMS: Dict[str, AlgorithmSpec] = {
    "full_dps": AlgorithmSpec(
        name="full_dps",
        display_name="Full-aperture DPS teacher",
        family="diffusion_posterior_sampling",
        description=(
            "Dense unsupervised DPS using all teacher sources at every reverse "
            "step. This is the quality teacher/reference, not a deployable "
            "budgeted policy."
        ),
        parameters={"uses_policy": False, "gradient_mode": "normalized_drift"},
    ),
    "fixed_subset_dps": AlgorithmSpec(
        name="fixed_subset_dps",
        display_name="Fixed equispaced subset DPS",
        family="diffusion_posterior_sampling",
        description=(
            "Budget baseline: use an equispaced k-source subset and keep the "
            "same sub_k schedule across key segments."
        ),
        parameters={"subset": "equispaced", "adaptive": False},
    ),
    "budgeted_dps": AlgorithmSpec(
        name="budgeted_dps",
        display_name="Budgeted DPS",
        family="budgeted_diffusion_posterior_sampling",
        description=(
            "Unsupervised DPS whose measurement guidance uses a source subset "
            "schedule across reverse-time key segments."
        ),
        parameters={
            "subset": "equispaced",
            "schedule": "per-segment sub_k_t",
            "cost": "sum(sub_k_t)/(NSEG*k)",
        },
    ),
    "policy_budget_ladder_dps": AlgorithmSpec(
        name="policy_budget_ladder_dps",
        display_name="Policy budget-ladder DPS",
        family="budget_ladder_dps",
        description=(
            "Deploy-time scalar_div policy chooses dyadic down/keep/up inner "
            "sub_k_t actions, while the outer ladder uses a convergence proxy "
            "to stop or add sources."
        ),
        parameters={
            "policy_recipe": "scalar_div",
            "deploy_sample": True,
            "actions": ["down", "keep", "up"],
        },
    ),
    "dps_raw_gradient": AlgorithmSpec(
        name="dps_raw_gradient",
        display_name="DPS raw-gradient guidance",
        family="dps_gradient_variant",
        description=(
            "DPS ablation that injects the raw physical likelihood gradient "
            "without Frobenius drift normalization."
        ),
        parameters={"gradient_mode": "raw_gradient"},
    ),
    "dps_normalized_drift": AlgorithmSpec(
        name="dps_normalized_drift",
        display_name="DPS normalized-drift guidance",
        family="dps_gradient_variant",
        description=(
            "Adopted USCT/FTTT reformulated-gradient DPS variant. Frobenius "
            "normalization fixes injected drift magnitude at each time."
        ),
        parameters={"gradient_mode": "normalized_drift"},
    ),
}


SCENARIOS: Dict[str, ScenarioSpec] = {
    "usct": ScenarioSpec(
        name="usct",
        display_name="USCT sound-speed reconstruction",
        modality="ultrasound_computed_tomography",
        equation=(
            "(nabla^2 + omega^2 / c(r)^2) u(r; omega) = -s(r; omega); "
            "A(c) samples the complex Helmholtz wavefield at receivers."
        ),
        inverse_setting=(
            "Recover a 240x240 sound-speed field from sparse ring-array "
            "transmitter/receiver measurements at 250 kHz. The problem is "
            "strongly underdetermined and often prior-limited."
        ),
        unknown="sound speed c(r), internally normalized on a 240x240 grid",
        measurement="complex boundary wavefield for selected transmitters and 128 receivers",
        operator_kind="nonlinear_pde_helmholtz_cbs",
        domain_shape=(1, 240, 240),
        range_shape=(64, 128),
        pool_size=128,
        teacher_k=64,
        ladder_ks=(8, 16, 32, 64),
        n_segments=6,
        n_steps=30,
        default_noise={"type": "precomputed_dobs_250k", "frequency_hz": 250000},
        default_paths={
            "data": "Datasets/AI4Scup2_simulated_CBS/{speed,dobs_250k}/{train,eval}",
            "score": "diffusion_model/score_sde_pytorch_training/workdir/AI4Scup2_full/240/checkpoints/checkpoint_20.pth",
            "policy": "boed_diffusion_budget_ladder_USCT/data/policy/usct_policy_div.pt",
        },
        scripts={
            "infer": "boed_diffusion_budget_ladder_USCT/scripts/12_policy_dps_infer_usct.py",
        },
    ),
    "fttt": ScenarioSpec(
        name="fttt",
        display_name="FTTT first-arrival travel-time tomography",
        modality="first_arrival_travel_time_tomography",
        equation=(
            "|nabla T(r)|^2 = 1 / c(r)^2 with T(source)=0; "
            "A(c) samples first-arrival travel times at receivers."
        ),
        inverse_setting=(
            "Recover a 128x128 Marmousi velocity/slowness image from sparse "
            "source/receiver travel times. Compared with USCT it is more "
            "measurement-limited and saturates around a smaller k."
        ),
        unknown="Marmousi medium image on a 128x128 grid, scaled to [0, 1]",
        measurement="first-arrival travel times for selected sources and receivers",
        operator_kind="nonlinear_pde_eikonal_pykonal",
        domain_shape=(1, 128, 128),
        range_shape=(32, 128),
        pool_size=32,
        teacher_k=32,
        ladder_ks=(4, 8, 16),
        n_segments=8,
        n_steps=40,
        default_noise={"type": "gaussian", "sigma_y": 0.01},
        default_paths={
            "data": "Datasets/Marmousi/images/{train,eval}",
            "score": "diffusion_model/score_sde_pytorch_training/workdir/seismic/128/checkpoint_7.pth",
            "policy": "boed_diffusion_budget_ladder/data/policy/fttt_policy_div.pt",
        },
        scripts={
            "infer": "boed_diffusion_budget_ladder/scripts/12_policy_dps_infer_fttt.py",
        },
    ),
}


def _to_plain(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def list_scenarios() -> List[Dict[str, Any]]:
    return [_to_plain(s) for s in SCENARIOS.values()]


def get_scenario(name: str) -> Dict[str, Any]:
    key = str(name).lower()
    if key not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose one of {sorted(SCENARIOS)}")
    return _to_plain(SCENARIOS[key])


def list_algorithms() -> List[Dict[str, Any]]:
    return [_to_plain(a) for a in ALGORITHMS.values()]


def get_algorithm(name: str) -> Dict[str, Any]:
    key = str(name).lower()
    if key not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}; choose one of {sorted(ALGORITHMS)}")
    return _to_plain(ALGORITHMS[key])


def describe_agent_interface() -> Dict[str, Any]:
    """Return the small set of backend calls the agent should wrap as tools."""
    return {
        "backend": "inv_framework.applications.boed_budget_ladder",
        "functions": [
            "list_scenarios",
            "get_scenario",
            "list_algorithms",
            "get_algorithm",
            "build_external_job",
            "run_external_job",
        ],
        "recommended_agent_tools": [
            "list_inverse_problem_scenarios",
            "describe_inverse_problem_scenario",
            "list_inverse_problem_algorithms",
            "build_inverse_problem_job",
            "run_inverse_problem_job",
        ],
    }


def _add_arg(cmd: List[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _add_bool(cmd: List[str], flag: str, enabled: Any) -> None:
    if bool(enabled):
        cmd.append(flag)


def build_external_job(
    scenario: str,
    task: str = "infer",
    project_root: Optional[os.PathLike] = None,
    python: Optional[str] = None,
    **params: Any,
) -> Dict[str, Any]:
    """Build a runnable job dict for the existing Diffusion_Policy scripts.

    The return value is JSON-serializable and intentionally dry-run friendly:
    the agent can show or store it without launching a GPU/CPU-heavy job.
    """
    scenario_key = str(scenario).lower()
    if scenario_key not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}")
    spec = SCENARIOS[scenario_key]
    task_key = str(task)
    if task_key not in spec.scripts:
        raise KeyError(f"scenario {scenario!r} has no task {task!r}")

    root = Path(project_root) if project_root is not None else DEFAULT_DIFFUSION_POLICY_ROOT
    script = root / spec.scripts[task_key]
    py = str(python or sys.executable)
    cmd: List[str] = [py, str(script)]

    if task_key == "infer":
        default_sample = 24 if scenario_key == "usct" else "data_0"
        _add_arg(cmd, "--sample", params.get("sample", default_sample))
        _add_arg(cmd, "--lambda-g", params.get("lambda_g"))
        _add_arg(cmd, "--policy", params.get("policy"))
        _add_arg(cmd, "--J", params.get("J"))
        _add_arg(cmd, "--stop-tol", params.get("stop_tol"))
        _add_arg(cmd, "--out-dir", params.get("out_dir"))
        if scenario_key == "usct":
            _add_arg(cmd, "--split", params.get("split"))
            _add_arg(cmd, "--cbs-iters", params.get("cbs_iters"))
            _add_bool(cmd, "--quick", params.get("quick", False))
            _add_bool(cmd, "--top-k64", params.get("top_k64", False))
        else:
            _add_arg(cmd, "--seed", params.get("seed"))
            _add_arg(cmd, "--n-workers", params.get("n_workers"))

    return {
        "scenario": scenario_key,
        "task": task_key,
        "cwd": str(script.parent.parent if task_key == "infer" else script.parent.parent),
        "script": str(script),
        "command": cmd,
        "exists": script.exists(),
        "params": params,
    }


def run_external_job(job: Mapping[str, Any],
                     timeout: Optional[float] = None,
                     dry_run: bool = False) -> Dict[str, Any]:
    """Run a job from ``build_external_job`` and return a JSON-safe result."""
    cmd = list(job["command"])
    cwd = str(job.get("cwd") or Path(job["script"]).parent)
    if dry_run:
        return {"status": "dry_run", "command": cmd, "cwd": cwd}
    proc = subprocess.run(
        cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": int(proc.returncode),
        "command": cmd,
        "cwd": cwd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def dumps(obj: Any) -> str:
    """JSON helper for thin agent wrappers."""
    return json.dumps(_to_plain(obj), ensure_ascii=False, indent=2)


__all__ = [
    "AlgorithmSpec",
    "ScenarioSpec",
    "ALGORITHMS",
    "SCENARIOS",
    "build_external_job",
    "describe_agent_interface",
    "dumps",
    "get_algorithm",
    "get_scenario",
    "list_algorithms",
    "list_scenarios",
    "run_external_job",
]
