"""Explicit image-unit conversion and common reconstruction metrics."""
import math
import torch
from ..operators.base import LinearOperator


def model_measurement(measurement, operator, scale=1., offset=0.):
    if not isinstance(operator, LinearOperator):
        raise TypeError('affine measurement conversion requires a LinearOperator')
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset):
        raise ValueError('image scale must be positive and scale/offset finite')
    if offset == 0:
        return measurement / scale
    ones = measurement.new_full((measurement.shape[0], *operator.domain_shape), offset)
    return (measurement - operator(ones)) / scale


def reconstruction_metrics(image, measurement, operator, reference=None, roi_radius=None):
    with torch.no_grad():
        residual = (operator(image)-measurement).float()
        metrics = {'relative_measurement_l2': float(torch.linalg.vector_norm(residual) /
                   torch.linalg.vector_norm(measurement.float()).clamp_min(1e-30))}
    if reference is not None:
        import numpy as np
        from skimage.metrics import structural_similarity
        x, ref = image.detach().cpu().numpy(), reference.detach().cpu().numpy()
        if x.shape != ref.shape or x.shape[:2] != (1,1):
            raise ValueError('image metrics currently require matching single-channel, single-image tensors')
        x, ref = x[0,0], ref[0,0]
        dr = float(ref.max()-ref.min())
        if dr <= 0: raise ValueError('PSNR/SSIM require a nonconstant reference')
        mask = np.ones(ref.shape, dtype=bool)
        if roi_radius is not None:
            h,w=ref.shape; yy,xx=np.ogrid[:h,:w]
            mask=(yy-(h-1)/2)**2+(xx-(w-1)/2)**2 <= roi_radius**2
        mse=float(np.mean((x[mask]-ref[mask])**2))
        _, scoremap=structural_similarity(ref,x,data_range=dr,full=True)
        metrics.update(psnr_db=float('inf') if mse==0 else 10*math.log10(dr*dr/mse),
                       ssim=float(scoremap[mask].mean()), data_range=dr)
    return metrics
