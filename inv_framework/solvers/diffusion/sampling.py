"""Shared tensor/model/schedule plumbing for diffusion inverse solvers."""
import inspect
from types import SimpleNamespace
import torch
from ..base import InverseProblemSolver
from ...operators.base import LinearOperator
from .diffusers_compat import DiffusersScheduleAdapter, DiffusersUNetWrapper


def randn_tensor(*args, **kwargs):
    from diffusers.utils.torch_utils import randn_tensor as draw
    return draw(*args, **kwargs)


class SamplingSolver(InverseProblemSolver):
    """Common interface; method parameters are accepted by the constructor or solve.

    ``model(x,t)`` must predict noise. ``schedule`` may be a raw Diffusers
    scheduler or DiffusersScheduleAdapter. Latent methods additionally take a
    codec with encode/decode and scaling_factor. All outputs remain tensors in
    model image units. Full reference reproducibility also requires resetting
    global torch RNGs: some methods intentionally use torch.randn_like.
    """
    schedule_kind = 'ddim'
    latent = False
    requires_linear = False
    condition_type = None

    def __init__(self, model, schedule=None, *, codec=None, scale=1.0,
                 conditioning=None, **parameters):
        raw = getattr(model, 'unet', model)
        self.model = model if isinstance(model, DiffusersUNetWrapper) else DiffusersUNetWrapper(model)
        self.model_config = getattr(raw, 'config', None)
        if schedule is None:
            from diffusers import DDIMScheduler, DDPMScheduler
            if self.schedule_kind == 'resample':
                from .resample_schedule import ReSampleDDIMScheduler
                schedule = ReSampleDDIMScheduler()
            else:
                schedule = (DDPMScheduler if self.schedule_kind == 'ddpm' else DDIMScheduler)()
        self.schedule = schedule if isinstance(schedule, DiffusersScheduleAdapter) else DiffusersScheduleAdapter(schedule)
        if self.schedule_kind == 'resample':
            from .resample_schedule import ReSampleDDIMScheduler
            if not isinstance(self.schedule._sched, ReSampleDDIMScheduler):
                self.schedule = DiffusersScheduleAdapter(ReSampleDDIMScheduler.from_config(self.schedule.config))
        self.codec = codec
        if self.latent and codec is None:
            raise ValueError('latent sampling requires a codec')
        self.scale = scale
        self.conditioning = conditioning
        self.parameters = dict(parameters)
        self._validate_parameters(parameters)

    def _validate_parameters(self, parameters):
        allowed = set(inspect.signature(self._sample).parameters) - {'batch_size', 'measurement', 'output_type', 'return_dict', 'kwargs'}
        unknown = set(parameters) - allowed
        if unknown:
            raise TypeError(f'unsupported method parameters: {sorted(unknown)}')

    @staticmethod
    def _steps(timesteps):
        return timesteps

    def solve(self, measurement, operator, *, generator=None, noiser=None, **parameters):
        if self.requires_linear:
            if not isinstance(operator, LinearOperator):
                raise TypeError(f'{type(self).__name__} requires a LinearOperator')
        if measurement.ndim != len(operator.range_shape) + 1:
            raise ValueError('measurement must include a batch dimension')
        self._validate_parameters(parameters)
        self.device, self.dtype = measurement.device, measurement.dtype
        if self.model_config is None:
            if self.latent:
                raise ValueError('latent noise model must expose config.sample_size and in_channels')
            self.model_config = SimpleNamespace(in_channels=operator.domain_shape[0], sample_size=operator.domain_shape[1:])
        if not self.latent:
            size = self.model_config.sample_size
            shape = (self.model_config.in_channels, *((size, size) if isinstance(size, int) else size))
            if tuple(shape) != tuple(operator.domain_shape):
                raise ValueError(f'model grid {shape} differs from operator domain {operator.domain_shape}')
        self.model.eval()
        self.measurement_condition = self.conditioning
        if self.measurement_condition is None:
            self.measurement_condition = (self.condition_type(operator, noiser, scale=self.scale)
                if self.condition_type else SimpleNamespace(operator=operator, noiser=noiser))
        elif self.measurement_condition.operator is not operator:
            raise ValueError('conditioning and solve must use the same operator')
        opts = {**self.parameters, **parameters}
        if 'scale' in inspect.signature(self._sample).parameters:
            opts.setdefault('scale', self.scale)
        with torch.enable_grad():
            result = self._sample(batch_size=measurement.shape[0], generator=generator,
                                  measurement=measurement, **opts)
        if tuple(result.shape[1:]) != tuple(operator.domain_shape):
            raise ValueError('decoded output grid differs from operator domain')
        return result
