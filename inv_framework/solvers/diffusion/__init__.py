from .scheduler import NoiseSchedule, VPSchedule
from .conditioning import ConditioningMethod, PosteriorSampling, CGDataConsistency
from .dps import DPSSolver
from .velocity_prior import VelocityPrior, TinyVelocityPrior
from .dds import DDSSolver
from .dds_lora import LoRAAdaptiveDDSSolver
from .lora import (
    LoraInjectedLinear,
    LoraInjectedConv2d,
    inject_lora_linear,
    adapt_pmfdit,
    set_lora_scale,
    save_lora_state,
    load_lora_state,
    count_trainable,
)
from .budget_dps import (
    BudgetedDPSSolver,
    BudgetedPosteriorSampling,
    equispaced_subset,
    grad_cost,
    make_equispaced_schedule,
    reformulated_gradient,
)

__all__ = [
    "NoiseSchedule",
    "VPSchedule",
    "ConditioningMethod",
    "PosteriorSampling",
    "CGDataConsistency",
    "DPSSolver",
    "VelocityPrior",
    "TinyVelocityPrior",
    "DDSSolver",
    "LoRAAdaptiveDDSSolver",
    "LoraInjectedLinear",
    "LoraInjectedConv2d",
    "inject_lora_linear",
    "adapt_pmfdit",
    "set_lora_scale",
    "save_lora_state",
    "load_lora_state",
    "count_trainable",
    "BudgetedDPSSolver",
    "BudgetedPosteriorSampling",
    "equispaced_subset",
    "grad_cost",
    "make_equispaced_schedule",
    "reformulated_gradient",
]

try:
    from .pmf_compat import PMFVelocityPrior  # noqa: F401
    __all__.append("PMFVelocityPrior")
except ImportError:
    pass

try:
    from .diffusers_compat import (
        DiffusersScheduleAdapter,
        DiffusersUNetWrapper,
    )
    __all__.extend(["DiffusersScheduleAdapter", "DiffusersUNetWrapper"])
except ImportError:
    pass
