"""ReSample inverse-problem solver. Reference behavior is documented in docs/diffusion_methods.md."""
from typing import Optional, Union, List
import inspect
import torch
from .sampling import SamplingSolver, randn_tensor

class ReSampleSolver(SamplingSolver):
    schedule_kind = 'resample'
    latent = True
    condition_type = None

    def _sample(self, batch_size: int=1, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, eta: float=0.0, num_inference_steps: int=50, measurement=None, inter_timesteps=5, pixel_lr=0.01, latent_lr=0.005, pixel_maxiters=2000, latent_maxiters=500, pixel_eps=0.001, latent_eps=0.001, **kwargs):
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
            index = t // 2
            a_t = self.schedule.alphas_cumprod[index]
            a_prev = self.schedule.alphas_cumprod[index - 1] if index > 0 else torch.tensor(1.0, device=a_t.device)
            latent_model_input = self.schedule.scale_model_input(latents, t)
            noise_prediction = self.model(latent_model_input, t)
            out = self.schedule.step_output(noise_prediction, t, latents, **extra_kwargs)
            latents_prev = out.prev_sample
            pred_original_latents = out.pred_original_sample
            pseudo_original_latents = out.pseudo_original_sample
            pseudo_original_latents = pseudo_original_latents / self.codec.scaling_factor
            pseudo_original_image = self.codec.decode(pseudo_original_latents)
            Ax = self.measurement_condition.operator.forward(pseudo_original_image)
            difference = measurement - Ax
            min_nonzero = measurement.abs()[measurement.abs() != 0].min()
            denominator = measurement.abs().clone()
            denominator[denominator == 0] = min_nonzero
            norm = torch.linalg.norm(difference) / denominator
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]
            latents = latents_prev - norm_grad * a_t * 0.5
            splits = 3
            index_split = num_inference_steps // splits
            if index <= num_inference_steps - index_split and index > 0:
                latents_cp = latents.detach().clone()
                if index % 10 == 0:
                    for k in range(num_inference_steps - index - 1, min(num_inference_steps - index - 1 + inter_timesteps, num_inference_steps - 1)):
                        step_ = self.schedule.timesteps[k + 1]
                        index_ = num_inference_steps - k - 1
                        noise_prediction = self.model(latents, step_)
                        out = self.schedule.step_output(noise_prediction, step_, latents, **extra_kwargs)
                        latents_prev = out.prev_sample.detach()
                        pred_original_latents = out.pred_original_sample.detach()
                        pseudo_original_latents = out.pseudo_original_sample.detach()
                        latents = latents_prev
                    if index >= 0:
                        sigma = 40 * (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)
                    else:
                        sigma = 0.5
                    if index >= index_split:
                        pseudo_original_latents = pseudo_original_latents.detach()
                        pseudo_original_latents = pseudo_original_latents / self.codec.scaling_factor
                        pseudo_original_image = self.codec.decode(pseudo_original_latents)
                        opt_image = self.pixel_optimization(measurement, pseudo_original_image, self.measurement_condition.operator.forward, eps=pixel_eps, max_iters=pixel_maxiters, lr=pixel_lr)
                        opt_image = opt_image.detach()
                        opt_latents = self.codec.encode(opt_image).detach()
                        latents = self.stochastic_resample(pseudo_x0=opt_latents, x_t=latents_cp, a_t=a_prev, sigma=sigma)
                        latents = latents.requires_grad_()
                    elif index < index_split:
                        (pseudo_original_latents, _) = self.latent_optimization(measurement=measurement, z_init=pseudo_original_latents.detach(), operator_fn=self.measurement_condition.operator.forward, lr=latent_lr, max_iters=latent_maxiters, eps=latent_eps)
                        sigma = 40 * (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)
                        latents = self.stochastic_resample(pseudo_x0=pseudo_original_latents, x_t=latents_cp, a_t=a_prev, sigma=sigma)
        pseudo_original_latents = pseudo_original_latents / self.codec.scaling_factor
        image = self.codec.decode(pseudo_original_latents)
        image = image.detach_()
        return image.detach()

    def pixel_optimization(self, measurement, x_prime, operator_fn, eps=0.001, max_iters=2000, lr=0.01):
        loss = torch.nn.MSELoss()
        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=lr)
        measurement = measurement.detach()
        for _ in range(max_iters):
            optimizer.zero_grad()
            measurement_loss = loss(measurement, operator_fn(opt_var))
            measurement_loss.backward()
            optimizer.step()
            if measurement_loss < eps ** 2:
                break
        return opt_var

    def latent_optimization(self, measurement, z_init, operator_fn, eps=0.001, max_iters=500, lr=0.005):
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW([z_init], lr=lr)
        measurement = measurement.detach()
        init_loss = 0
        losses = []
        for itr in range(max_iters):
            optimizer.zero_grad()
            output = loss(measurement, operator_fn(self.codec.decode(z_init)))
            if itr == 0:
                init_loss = output.detach().clone()
            output.backward()
            optimizer.step()
            cur_loss = output.detach().cpu().numpy()
            if itr < 200:
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)
            if cur_loss < eps ** 2:
                break
        return (z_init, init_loss)

    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        device = self.device
        noise = torch.randn_like(pseudo_x0, device=device)
        return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t) / (sigma + 1 - a_t) + noise * torch.sqrt(1 / (1 / sigma + 1 / (1 - a_t)))
