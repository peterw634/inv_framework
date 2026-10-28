import numpy as np

from inv_framework.applications.nasn_ct.measurement import (
    IMAGE_SIZE, MEASURED_DETECTORS, PADDED_CENTER, PAD_LEFT, downsample_and_pad,
)


def test_center_pixel_mapping():
    assert PADDED_CENTER == 1006.283203125


def test_downsample_and_pad_layout():
    source = np.ones((3, 2560), dtype=np.float32)
    result = downsample_and_pad(source, scale=2)
    assert result.shape == (3, IMAGE_SIZE)
    assert np.count_nonzero(result[0]) == MEASURED_DETECTORS
    np.testing.assert_allclose(result[:, PAD_LEFT:PAD_LEFT + MEASURED_DETECTORS], 2)
    assert not result[:, :PAD_LEFT].any()
