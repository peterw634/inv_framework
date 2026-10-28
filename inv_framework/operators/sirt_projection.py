"""Linear-operator adapter with normalized SIRT projection and pseudo-inverse."""
import torch
from .base import LinearOperator


class SIRTProjectionOperator(LinearOperator):
    """Keep the underlying forward/adjoint; select SIRT projection semantics.

    Reference SIRT runs each batch member separately, in float32, using the
    original in-place operation order. Backprojection is detached, matching
    the inference-only reference projection rather than differentiable CG.
    """
    def __init__(self, operator, iterations=100):
        if not isinstance(operator, LinearOperator):
            raise TypeError('SIRT projection requires a LinearOperator')
        self.operator = operator
        self.domain_shape, self.range_shape = operator.domain_shape, operator.range_shape
        self.iterations = int(iterations)
        self._weights = {}

    def forward(self, x):
        return self.operator(x)

    def adjoint(self, y):
        return self.operator.adjoint(y).detach()

    def _scales(self, y, fresh=False):
        key = (y.device,)
        if fresh or key not in self._weights:
            C = self.adjoint(torch.ones((1, *self.range_shape), device=y.device))
            C[C < 1e-8] = float('inf')
            C.reciprocal_()
            R = self.forward(torch.ones((1, *self.domain_shape), device=y.device)).detach()
            R[R < 1e-8] = float('inf')
            R.reciprocal_()
            if fresh:
                return C, R
            self._weights[key] = C, R
        return self._weights[key]

    def project(self, x, y, **kwargs):
        C, R = self._scales(y)
        return x - C * self.adjoint(R * self.forward(x)) + C * self.adjoint(R * y)

    def pseudo_inverse(self, y, num_iterations=None, x_init=None, min_constraint=None, max_constraint=None):
        C, R = self._scales(y, fresh=True)
        results = []
        with torch.no_grad():
            for i in range(y.shape[0]):
                x = (torch.zeros((1, *self.domain_shape), device=y.device)
                     if x_init is None else x_init[i:i+1].clone())
                for _ in range(self.iterations if num_iterations is None else num_iterations):
                    residual = self.forward(x)
                    residual -= y[i:i+1]
                    residual *= R
                    update = self.adjoint(residual)
                    update *= C
                    x -= update
                    if min_constraint is not None or max_constraint is not None:
                        x.clamp_(min_constraint, max_constraint)
                results.append(x)
        return torch.cat(results)
