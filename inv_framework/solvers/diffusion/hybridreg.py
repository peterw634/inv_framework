"""HybridReg inverse-problem solver. Reference behavior is documented in docs/diffusion_methods.md."""
from typing import Optional, Union, List
from math import sqrt
import torch
from .sampling import SamplingSolver, randn_tensor

class HybridRegSolver(SamplingSolver):
    requires_linear = True
    schedule_kind = 'ddim'
    latent = False
    condition_type = None

    def _sample(self, batch_size: int=1, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, num_inference_steps: int=100, measurement=None, sigma=0, loss_measurement_weight=1.0, loss_noise_weight=1.0, lr=0.25, beta=0.2):
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
        image = pseduo_inv_x0
        mu = torch.autograd.Variable(image, requires_grad=True)
        optimizer = torch.optim.Adam([mu], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
        last_noise = 0
        for t in self._steps(self.schedule.timesteps):
            noise_x0 = torch.randn_like(mu, device=self.device)
            noise_xt = torch.randn_like(mu, device=self.device)
            x0_pred_original_sample = mu + sigma * noise_x0
            if t == self.schedule.timesteps[0]:
                xt_pred = self.schedule.alphas_cumprod[t].sqrt() * x0_pred_original_sample + (1 - self.schedule.alphas_cumprod[t]).sqrt() * noise_xt
            else:
                noise = sqrt(1 - beta) * last_noise + sqrt(beta) * noise_xt
                xt_pred = self.schedule.alphas_cumprod[t].sqrt() * x0_pred_original_sample + (1 - self.schedule.alphas_cumprod[t]).sqrt() * noise
            model_output = self.model(xt_pred, t)
            if t == self.schedule.timesteps[0]:
                last_noise = model_output.clone()
            else:
                loss_measurement = measurement - self.measurement_condition.operator.forward(x0_pred_original_sample)
                loss_measurement = (loss_measurement ** 2).mean() / 2
                loss_noise = torch.mul((model_output.detach_() - noise_xt).detach(), x0_pred_original_sample).mean()
                last_noise = model_output.clone()
            snr_inv = (1 - self.schedule.alphas_cumprod[t]).sqrt() / self.schedule.alphas_cumprod[t].sqrt()
            if t != self.schedule.timesteps[0] and t != self.schedule.timesteps[1]:
                loss = loss_measurement_weight * loss_measurement + snr_inv * loss_noise_weight * loss_noise
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        image = x0_pred_original_sample.detach_()
        return image.detach()
