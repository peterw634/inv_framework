"""DiffStateGrad inverse-problem solver. Reference behavior is documented in docs/diffusion_methods.md."""
from typing import Optional, Union, List
import inspect
import torch
from .sampling import randn_tensor
from .resample import ReSampleSolver

class DiffStateGradSolver(ReSampleSolver):
    schedule_kind = 'resample'
    latent = True
    condition_type = None

    def compute_rank_for_explained_variance(self, singular_values, explained_variance_cutoff):
        import numpy as np
        total_rank = 0
        for channel_singular_values in singular_values:
            squared_singular_values = channel_singular_values ** 2
            cumulative_variance = np.cumsum(squared_singular_values) / np.sum(squared_singular_values)
            rank = np.searchsorted(cumulative_variance, explained_variance_cutoff) + 1
            total_rank += rank
        return int(total_rank / 3)

    def compute_svd_and_adaptive_rank(self, z_t, var_cutoff):
        (U, s, Vh) = torch.linalg.svd(z_t[0], full_matrices=False)
        s_numpy = s.detach().cpu().numpy()
        adaptive_rank = self.compute_rank_for_explained_variance([s_numpy], var_cutoff)
        return (U, s, Vh, adaptive_rank)

    def apply_diffstategrad(self, norm_grad, iteration_count, period, U=None, s=None, Vh=None, adaptive_rank=None):
        if period != 0 and iteration_count % period == 0:
            if any((param is None for param in [U, s, Vh, adaptive_rank])):
                raise ValueError('SVD components and adaptive_rank must be provided when iteration_count % period == 0')
            A = U[:, :, :adaptive_rank]
            B = Vh[:, :adaptive_rank, :]
            low_rank_grad = torch.matmul(A.permute(0, 2, 1), norm_grad[0]) @ B.permute(0, 2, 1)
            projected_grad = torch.matmul(A, low_rank_grad) @ B
            projected_grad = projected_grad.float().unsqueeze(0)
            return projected_grad
        return norm_grad

    def _sample(self, batch_size: int=1, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, eta: float=0.0, num_inference_steps: int=50, measurement=None, inter_timesteps=5, pixel_lr=0.01, latent_lr=0.005, var_cutoff=0.99, period=1, **kwargs):
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
                        image = self.codec.decode(latents.detach() / self.codec.scaling_factor)
                        opt_image = self.pixel_optimization(measurement=measurement, x_prime=pseudo_original_image, operator_fn=self.measurement_condition.operator.forward, eps=0.001, max_iters=2000, lr=pixel_lr, var_cutoff=var_cutoff, x_prev=image, period=period)
                        opt_image = opt_image.detach()
                        opt_latents = self.codec.encode(opt_image).detach()
                        latents = self.stochastic_resample(pseudo_x0=opt_latents, x_t=latents_cp, a_t=a_prev, sigma=sigma)
                        latents = latents.requires_grad_()
                    elif index < index_split:
                        (pseudo_original_latents, _) = self.latent_optimization(measurement=measurement, z_init=pseudo_original_latents.detach(), operator_fn=self.measurement_condition.operator.forward, var_cutoff=var_cutoff, z_prev=latents.detach(), period=period)
                        sigma = 40 * (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)
                        latents = self.stochastic_resample(pseudo_x0=pseudo_original_latents, x_t=latents_cp, a_t=a_prev, sigma=sigma)
        pseudo_original_latents = pseudo_original_latents / self.codec.scaling_factor
        image = self.codec.decode(pseudo_original_latents)
        image = image.detach_()
        return image.detach()

    def pixel_optimization(self, measurement, x_prime, operator_fn, eps=0.001, max_iters=2000, lr=None, var_cutoff=None, x_prev=None, period=None):
        loss = torch.nn.MSELoss()
        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        if lr is None:
            lr = 0.01
        optimizer = torch.optim.AdamW([opt_var], lr=lr)
        measurement = measurement.detach()
        (U, s, Vh, adaptive_rank) = self.compute_svd_and_adaptive_rank(x_prev, var_cutoff)
        for _ in range(max_iters):
            optimizer.zero_grad()
            measurement_loss = loss(measurement, operator_fn(opt_var))
            measurement_loss.backward()
            opt_var.grad = self.apply_diffstategrad(opt_var.grad, _, period, U, s, Vh, adaptive_rank)
            optimizer.step()
            if measurement_loss < eps ** 2:
                break
        return opt_var

    def latent_optimization(self, measurement, z_init, operator_fn, eps=0.001, max_iters=500, lr=None, var_cutoff=None, z_prev=None, period=None):
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()
        if lr is None:
            lr = 0.005
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW([z_init], lr=lr)
        measurement = measurement.detach()
        (U, s, Vh, adaptive_rank) = self.compute_svd_and_adaptive_rank(z_prev, var_cutoff)
        init_loss = 0
        losses = []
        for itr in range(max_iters):
            optimizer.zero_grad()
            output = loss(measurement, operator_fn(self.codec.decode(z_init)))
            if itr == 0:
                init_loss = output.detach().clone()
            output.backward()
            z_init.grad = self.apply_diffstategrad(z_init.grad, itr, period, U, s, Vh, adaptive_rank)
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
