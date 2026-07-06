import numpy as np
import pytest
from napari_dmc_segload.readers import to_16bit, _2d


def test_to_16bit_uint16_passthrough():
    a = np.array([[0, 65535]], dtype=np.uint16)
    out = to_16bit(a)
    assert out.dtype == np.uint16
    assert np.array_equal(out, a)


def test_to_16bit_uint8_scales_by_257():
    a = np.array([[0, 255]], dtype=np.uint8)
    out = to_16bit(a)
    assert out.dtype == np.uint16
    assert out[0, 0] == 0 and out[0, 1] == 65535


def test_to_16bit_float_rescales_to_full_range():
    a = np.array([[1.0, 3.0, 5.0]], dtype=np.float64)
    out = to_16bit(a)
    assert out.dtype == np.uint16
    assert out[0, 0] == 0 and out[0, 2] == 65535


def test_to_16bit_flat_input_returns_zeros():
    a = np.full((2, 2), 7.0)
    out = to_16bit(a)
    assert out.dtype == np.uint16
    assert np.all(out == 0)


def test_2d_squeezes_extra_leading_dims():
    a = np.zeros((1, 3, 4))
    assert _2d(a).shape == (3, 4)
