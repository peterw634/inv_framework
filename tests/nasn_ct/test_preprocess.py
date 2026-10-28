import numpy as np

from inv_framework.applications.nasn_ct.data.preprocess import flat_correct, pad_pa_batch, rivers_1d


def test_flat_correction_without_dark():
    projection = np.full((2, 3), 50, dtype=np.uint16)
    flat = np.full((2, 3), 100, dtype=np.float32)
    np.testing.assert_allclose(flat_correct(projection, flat), 0.5)


def test_rivers_removes_detector_stationary_stripe():
    angles = np.arange(7, dtype=np.float32)[:, None]
    stripe = np.array([[0, 0, 4, 0, 0]], dtype=np.float32)
    sinogram = angles + stripe
    corrected = rivers_1d(sinogram, 3)
    np.testing.assert_allclose(corrected.mean(0)[2], corrected.mean(0)[1], atol=2e-6)


def test_pad_pa_shape_and_finiteness():
    value = np.full((1, 5, 7), 0.8, dtype=np.float32)
    output = pad_pa_batch(value, 1.1e-6, 0.147, 27.4, 0.01, device="cpu")
    assert output.shape == value.shape
    assert np.isfinite(output).all()
