"""PSLD inverse-problem solver. Reference behavior is documented in docs/diffusion_methods.md."""
from typing import Optional, Union, List
import inspect
import torch
from .sampling import SamplingSolver, randn_tensor

class PSLDSolver(SamplingSolver):
    requires_linear = True
    schedule_kind = 'ddpm'
    latent = True
    condition_type = None

    def _sample(self, batch_size: int=1, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, eta: float=0.0, num_inference_steps: int=50, gamma: float=0.9, omega: float=0.1, measurement=None, **kwargs):
        if self.measurement_condition is None:
            raise ValueError('Measurement condition is not set.')
        latents = randn_tensor((batch_size, self.model_config.in_channels, self.model_config.sample_size, self.model_config.sample_size), generator=generator)
        latents = latents.to(self.device)
        latents = latents * self.schedule.init_noise_sigma
        self.schedule.set_timesteps(num_inference_steps)
        accepts_eta = 'eta' in set(inspect.signature(self.schedule.step_output).parameters.keys())
        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs['eta'] = eta
        for t in self._steps(self.schedule.timesteps):
            latents = latents.requires_grad_()
            latent_model_input = self.schedule.scale_model_input(latents, t)
            noise_prediction = self.model(latent_model_input, t)
            out = self.schedule.step_output(noise_prediction, t, latents, **extra_kwargs)
            latents_prev = out.prev_sample
            pred_original_latents = out.pred_original_sample
            pred_original_latents = pred_original_latents / self.codec.scaling_factor
            pred_original_image = self.codec.decode(pred_original_latents)
            measurement_pred_original = self.measurement_condition.operator.forward(pred_original_image)
            min_nonzero = measurement.abs()[measurement.abs() != 0].min()
            denominator = measurement.abs().clone()
            denominator[denominator == 0] = min_nonzero
            measurement_difference = torch.linalg.norm(measurement_pred_original - measurement) / denominator
            measurement_difference = measurement_difference.mean()
            projected_pred_original = self.measurement_condition.operator.project(pred_original_image, measurement_pred_original)
            latents_projected_pred_original = self.codec.encode(projected_pred_original)
            latents_projected_pred_original = latents_projected_pred_original * self.codec.scaling_factor
            latents_error = torch.linalg.norm(latents_projected_pred_original - pred_original_latents)
            total_error = measurement_difference * omega + latents_error * gamma
            grad = torch.autograd.grad(total_error, latents)[0]
            latents = latents_prev - grad
            latents = latents.detach_()
        latents = latents / self.codec.scaling_factor
        image = self.codec.decode(latents)
        image = image.detach_()
        return image.detach()
