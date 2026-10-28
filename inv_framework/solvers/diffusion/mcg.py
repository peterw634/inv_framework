"""MCG inverse-problem solver. Reference behavior is documented in docs/diffusion_methods.md."""
from typing import Optional, Union, List
import torch
from .sampling import SamplingSolver, randn_tensor
from .guidance import ManifoldConstraintGradient

class MCGSolver(SamplingSolver):
    requires_linear = True
    schedule_kind = 'ddpm'
    latent = False
    condition_type = ManifoldConstraintGradient

    def q_sample(self, x_start, t):
        noise = torch.randn_like(x_start)
        coef1 = self.schedule.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1).to(x_start.device)
        coef2 = (1 - self.schedule.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1).to(noise.device)
        return coef1 * x_start + coef2 * noise

    def _sample(self, batch_size: int=1, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, num_inference_steps: int=1000, measurement=None):
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
        self.schedule.set_timesteps(num_inference_steps)
        for t in self._steps(self.schedule.timesteps):
            image = image.requires_grad_()
            model_output = self.model(image, t)
            out = self.schedule.step_output(model_output, t, image, generator=generator)
            noisy_measurement = self.q_sample(measurement, t)
            (image, distance) = self.measurement_condition.apply(x_t=out.prev_sample, measurement=measurement, noisy_measurement=noisy_measurement, x_prev=image, x_0_hat=out.pred_original_sample)
            image = image.detach_()
        return image.detach()
