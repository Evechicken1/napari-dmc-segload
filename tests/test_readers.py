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


import tifffile
from napari_dmc_segload.readers import TiffReader, make_reader


def _write_multichannel_tif(path, n_ch=3, h=40, w=60):
    # distinct constant value per channel so crops are identifiable
    arr = np.stack(
        [np.full((h, w), (c + 1) * 1000, dtype=np.uint16) for c in range(n_ch)],
        axis=0,
    )
    tifffile.imwrite(str(path), arr)
    return arr


def test_tiffreader_reports_shape(tmp_path):
    p = tmp_path / "slide.tif"
    _write_multichannel_tif(p, n_ch=3, h=40, w=60)
    r = TiffReader(p)
    assert r.shape == (3, 40, 60)


def test_tiffreader_overview_downsamples(tmp_path):
    p = tmp_path / "slide.tif"
    _write_multichannel_tif(p, n_ch=3, h=40, w=60)
    r = TiffReader(p)
    ov = r.overview(0.1)  # step = 10 -> 4 x 6
    assert ov.shape == (3, 4, 6)


def test_tiffreader_crop_fullres_returns_region_of_right_channel(tmp_path):
    p = tmp_path / "slide.tif"
    _write_multichannel_tif(p, n_ch=3, h=40, w=60)
    r = TiffReader(p)
    crop = r.crop_fullres(1, 0, 10, 0, 5)  # channel 1 -> value 2000
    assert crop.shape == (10, 5)
    assert np.all(crop == 2000)


def test_make_reader_picks_tiff_for_tif_suffix(tmp_path):
    p = tmp_path / "slide.tif"
    _write_multichannel_tif(p, n_ch=2, h=20, w=20)
    assert isinstance(make_reader(p), TiffReader)
