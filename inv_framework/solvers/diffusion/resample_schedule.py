"""Optional DDIM schedule with the ReSample pseudo-clean prediction."""
from diffusers import DDIMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional
import torch

class ReSampleSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
        pseudo_original_sample
        Resample specific, \\hat{x}_0 is computed using Tweedie's formula
    """
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None
    pseudo_original_sample: Optional[torch.Tensor] = None

class ReSampleDDIMScheduler(DDIMScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, eta: float=0.0, use_clipped_model_output: bool=False, generator=None, variance_noise: Optional[torch.Tensor]=None, return_dict: bool=True):
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        if self.config.prediction_type == 'epsilon':
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.config.prediction_type == 'sample':
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        elif self.config.prediction_type == 'v_prediction':
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
            pred_epsilon = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
        else:
            raise ValueError(f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`')
        if self.config.prediction_type == 'epsilon':
            pseudo_original_sample = (sample - (1 - alpha_prod_t) * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.config.prediction_type == 'sample':
            pseudo_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pseudo_original_sample) / beta_prod_t ** 0.5
        elif self.config.prediction_type == 'v_prediction':
            pseudo_original_sample = alpha_prod_t ** 0.5 * sample - (1 - alpha_prod_t) * model_output
            pred_epsilon = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)
        if self.config.thresholding:
            pseudo_original_sample = self._threshold_sample(pseudo_original_sample)
        elif self.config.clip_sample:
            pseudo_original_sample = pseudo_original_sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** 0.5
        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError('Cannot pass both generator and variance_noise. Please make sure that either `generator` or `variance_noise` stays `None`.')
            if variance_noise is None:
                variance_noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance
        if not return_dict:
            return (prev_sample, pred_original_sample, pseudo_original_sample)
        return ReSampleSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample, pseudo_original_sample=pseudo_original_sample)
