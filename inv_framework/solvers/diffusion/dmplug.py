"""DMPlug inverse-problem solver. Reference behavior is documented in docs/diffusion_methods.md."""
from typing import Optional, Union, List
import torch
from .sampling import SamplingSolver, randn_tensor

class DMPlugSolver(SamplingSolver):
    schedule_kind = 'ddim'
    latent = False
    condition_type = None

    def _sample(self, batch_size: int=1, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, num_inference_steps: int=1000, measurement=None, epochs=1000, lr=0.01, criterion=torch.nn.MSELoss()):
        if self.measurement_condition is None:
            raise ValueError('Measurement condition is not set.')
        if isinstance(self.model_config.sample_size, int):
            image_shape = (batch_size, self.model_config.in_channels, self.model_config.sample_size, self.model_config.sample_size)
        else:
            image_shape = (batch_size, self.model_config.in_channels, *self.model_config.sample_size)
        if self.device.type == 'mps':
            start_image = randn_tensor(image_shape, generator=generator, dtype=self.dtype)
            start_image = start_image.to(self.device)
        else:
            start_image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.dtype)
        start_image = start_image.requires_grad_()
        self.schedule.set_timesteps(num_inference_steps)
        optimizer = torch.optim.Adam([start_image], lr=lr)
        for i in range(epochs):
            self.model.eval()
            optimizer.zero_grad()
            for t in self._steps(self.schedule.timesteps):
                if t == self.schedule.timesteps[0]:
                    model_output = self.model(start_image, t)
                    image = self.schedule.step_output(model_output, t, start_image, generator=generator).prev_sample
                else:
                    model_output = self.model(image, t)
                    image = self.schedule.step_output(model_output, t, image, generator=generator).prev_sample
            image = torch.clamp(image, -1, 1)
            loss = criterion(self.measurement_condition.operator.forward(image), measurement)
            loss.backward()
            optimizer.step()
        image = image.detach_()
        return image.detach()
