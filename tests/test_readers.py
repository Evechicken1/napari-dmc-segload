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


from napari_dmc_segload.readers import VerticalFlipReader


class _FakeBase:
    some_attr = "hello"

    def __init__(self, arr):
        self.arr = arr
        self.n_ch, self.H, self.W = arr.shape

    @property
    def shape(self):
        return (self.n_ch, self.H, self.W)

    def overview(self, scale):
        step = max(1, int(round(1 / scale)))
        return self.arr[:, ::step, ::step]

    def crop_fullres(self, c, r0, r1, c0, c1):
        return self.arr[c, r0:r1, c0:c1]


def _demo_arr(n_ch=2, h=10, w=6):
    # value encodes channel/row/col so any mirror+crop is exactly checkable
    a = np.zeros((n_ch, h, w), dtype=np.uint16)
    for c in range(n_ch):
        for r in range(h):
            a[c, r, :] = c * 1000 + r * 10 + np.arange(w)
    return a


def test_vertical_flip_overview_reverses_rows():
    base = _FakeBase(_demo_arr())
    flip = VerticalFlipReader(base)
    np.testing.assert_array_equal(flip.overview(1.0), base.overview(1.0)[:, ::-1, :])


def test_vertical_flip_crop_equals_crop_of_flipped_image():
    arr = _demo_arr(h=10, w=6)
    flip = VerticalFlipReader(_FakeBase(arr))
    r0, r1, c0, c1 = 2, 5, 1, 4
    # flipping the whole image then cropping [r0:r1] is the ground truth
    expected = arr[0, ::-1, :][r0:r1, c0:c1]
    np.testing.assert_array_equal(flip.crop_fullres(0, r0, r1, c0, c1), expected)


def test_vertical_flip_full_height_crop_is_whole_reversed():
    arr = _demo_arr(h=10, w=6)
    flip = VerticalFlipReader(_FakeBase(arr))
    np.testing.assert_array_equal(flip.crop_fullres(1, 0, 10, 0, 6), arr[1, ::-1, :])


def test_vertical_flip_shape_unchanged_and_getattr_delegates():
    flip = VerticalFlipReader(_FakeBase(_demo_arr()))
    assert flip.shape == (2, 10, 6)
    assert flip.some_attr == "hello"


def _write_rowvarying_tif(path, n_ch=3, h=40, w=60):
    arr = np.stack(
        [(np.fromfunction(lambda r, x: r, (h, w), dtype=int) + (c + 1) * 1000).astype(np.uint16)
         for c in range(n_ch)],
        axis=0,
    )
    tifffile.imwrite(str(path), arr)
    return arr


def test_make_reader_flip_y_reverses_overview(tmp_path):
    p = tmp_path / "slide.tif"
    _write_rowvarying_tif(p)
    plain = make_reader(p, flip_y=False)
    flipped = make_reader(p, flip_y=True)
    np.testing.assert_array_equal(flipped.overview(0.1), plain.overview(0.1)[:, ::-1, :])


def test_make_reader_flip_y_false_is_plain_reader(tmp_path):
    p = tmp_path / "slide.tif"
    _write_rowvarying_tif(p)
    assert isinstance(make_reader(p, flip_y=False), TiffReader)
    assert isinstance(make_reader(p), TiffReader)
