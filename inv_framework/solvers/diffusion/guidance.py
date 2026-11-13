"""Measurement guidance with preserved reference loss conventions."""
import torch
from .conditioning import ConditioningMethod
from ...operators.base import LinearOperator


def normalized_measurement_loss(prediction, measurement):
    """``||y - A x|| * mean(1 / |y|)`` over the measured entries (DM4CT loss).

    Exactly-zero entries carry no measurement (rays outside the field of view,
    truncated detector columns) and are left out of the mean. The reference
    replaced them by the smallest nonzero |y|, which inflates the step by the
    count of such rays (19x for a fan beam whose detector overhangs the
    volume); without exact zeros the value is unchanged. An all-zero
    measurement still fails, as in the reference.
    """
    difference = measurement - prediction
    magnitude = measurement.abs()
    measured = magnitude != 0
    magnitude[measured].min()          # raises on an all-zero measurement
    return torch.linalg.norm(difference) * (1.0 / magnitude[measured]).mean()


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
