"""PGDM inverse-problem solver. Reference behavior is documented in docs/diffusion_methods.md."""
from typing import Optional, Union, List
import torch
from .sampling import SamplingSolver, randn_tensor
from .guidance import PseudoInverseGuidance

class PGDMSolver(SamplingSolver):
    requires_linear = True
    schedule_kind = 'ddim'
    latent = False
    condition_type = PseudoInverseGuidance

    def _sample(self, batch_size: int=1, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, num_inference_steps: int=100, measurement=None, scale: float=1.0):
        if self.measurement_condition is None:
            raise ValueError('Measurement condition is not set.')
        if isinstance(self.model_config.sample_size, int):
            image_shape = (batch_size, self.model_config.in_channels, self.model_config.sample_size, self.model_config.sample_size)
        else:
            image_shape = (batch_size, self.model_config.in_channels, *self.model_config.sample_size)
        if self.device.type == 'mps':
            image = randn_tensor(image_shape, generator=generator, dtype=self.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.dtype)
        if image.ndim == 4 and measurement.ndim == 3:
            measurement = measurement.unsqueeze(0)
        self.schedule.set_timesteps(num_inference_steps)
        pseduo_inv_x0 = self.measurement_condition.operator.pseudo_inverse(measurement)
        pseduo_inv_x0 = pseduo_inv_x0.to(image.dtype).to(image.device)
        alpha_prod_t = self.schedule.alphas_cumprod[self.schedule.timesteps[0]]
        image = alpha_prod_t.sqrt() * pseduo_inv_x0 + (1 - alpha_prod_t) * image
        for t in self._steps(self.schedule.timesteps):
            image = image.requires_grad_()
            model_output = self.model(image, t)
            out = self.schedule.step_output(model_output, t, image, generator=generator)
            prev_timestep = t - self.schedule.config.num_train_timesteps // self.schedule.num_inference_steps
            coeff = self.schedule.alphas_cumprod[prev_timestep].sqrt() * self.schedule.alphas_cumprod[t].sqrt() * scale
            self.measurement_condition.scale = coeff
            (image, distance) = self.measurement_condition.apply(x_t=out.prev_sample, measurement=measurement, x_prev=image, x_0_hat=out.pred_original_sample)
            image = image.detach_()
        return image.detach()
