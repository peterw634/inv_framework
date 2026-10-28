"""Measurement guidance with preserved reference loss conventions."""
import torch
from .conditioning import ConditioningMethod
from ...operators.base import LinearOperator


def normalized_measurement_loss(prediction, measurement):
    # Deliberately preserve the reference reduction order and all-zero failure.
    difference = measurement - prediction
    min_nonzero = measurement.abs()[measurement.abs() != 0].min()
    denominator = measurement.abs().clone()
    denominator[denominator == 0] = min_nonzero
    return (torch.linalg.norm(difference) / denominator).mean()


class NormalizedPosteriorSampling(ConditioningMethod):
    def apply(self, x_t, x_0_hat, measurement, x_prev, **kwargs):
        norm = normalized_measurement_loss(self.operator(x_0_hat), measurement)
        grad = torch.autograd.grad(norm, x_prev)[0]
        return x_t - grad * self.scale, norm


class ManifoldConstraintGradient(NormalizedPosteriorSampling):
    def __init__(self, operator, noiser=None, scale=1.0):
        if not isinstance(operator, LinearOperator):
            raise TypeError('MCG requires a LinearOperator with project')
        super().__init__(operator, noiser, scale)

    def apply(self, x_t, x_0_hat, measurement, x_prev, noisy_measurement, **kwargs):
        x_t, norm = super().apply(x_t, x_0_hat, measurement, x_prev)
        return self.operator.project(x_t, noisy_measurement), norm


class PseudoInverseGuidance(ConditioningMethod):
    def apply(self, x_t, x_0_hat, measurement, x_prev, **kwargs):
        mat = self.operator.pseudo_inverse(measurement) - self.operator.pseudo_inverse(self.operator(x_0_hat))
        grad = torch.autograd.grad((mat.detach() * x_0_hat).sum(), x_prev)[0]
        return x_t + grad * self.scale, None
