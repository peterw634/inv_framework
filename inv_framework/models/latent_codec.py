"""VQ codec adapter; scaling is explicit in the sampling algorithm."""
import torch.nn as nn


class VQCodec(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.scaling_factor = float(model.config.scaling_factor)

    def encode(self, image):
        return self.model.encode(image).latents

    def decode(self, latents):
        return self.model.decode(latents).sample
