"""DDS inverse-problem solver. Reference behavior is documented in docs/diffusion_methods.md."""
from typing import Optional, Union, List
import torch
from .sampling import SamplingSolver, randn_tensor

class DDIMDDSSolver(SamplingSolver):
    requires_linear = True
    schedule_kind = 'ddim'
    latent = False
    condition_type = None

    def _sample(self, batch_size: int=1, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, num_inference_steps: int=100, measurement=None, eta=0.85, cg_inner=5, cg_eps=1e-05, gamma=0):
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
        bcg = self.measurement_condition.operator.adjoint(measurement)
        for t in self._steps(self.schedule.timesteps):
            with torch.no_grad():
                model_output = self.model(image, t)
            out = self.schedule.step_output(model_output, t, image, generator=generator)
            x0_pred_original_sample = out.pred_original_sample
            if gamma > 0:
                bcg = x0_pred_original_sample + gamma * self.measurement_condition.operator.adjoint(measurement)
            r = bcg - self.measurement_condition.operator.adjoint(self.measurement_condition.operator(x0_pred_original_sample)) - gamma * x0_pred_original_sample
            p = r.clone()
            rsold = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            for i in range(cg_inner):
                Ap = self.measurement_condition.operator.adjoint(self.measurement_condition.operator(p)) + gamma * p
                a = rsold / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)
                x0_pred_original_sample = x0_pred_original_sample + a * p
                r = r - a * Ap
                rsnew = torch.matmul(r.view(1, -1), r.view(1, -1).T)
                if torch.sqrt(rsnew) < cg_eps:
                    break
                p = r + rsnew / rsold * p
                rsold = rsnew
            prev_timestep = t - self.schedule.config.num_train_timesteps // self.schedule.num_inference_steps
            alpha_prod_t = self.schedule.alphas_cumprod[t]
            alpha_prod_t_prev = self.schedule.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.schedule.final_alpha_cumprod
            variance = self.schedule._get_variance(t, prev_timestep)
            std_dev_t = eta * variance ** 0.5
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output
            prev_sample = alpha_prod_t_prev ** 0.5 * x0_pred_original_sample + pred_sample_direction
            if eta > 0:
                variance_noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
                variance = std_dev_t * variance_noise
                prev_sample = prev_sample + variance
            image = prev_sample
        return image.detach()
