"""Lazy whole-slide readers + dtype helpers for the section widget.

Ported from Histology/dmc_section.py. Pure I/O — no napari import here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

FORCE_16BIT = True   # upcast 8-bit images to 16-bit (DMC wants 16-bit TIFF)
ZERO_PAD = 3         # slice number padding -> s001, s002, ...


def _2d(a: np.ndarray) -> np.ndarray:
    """Squeeze an array down to 2D (take leading index of any extra dims)."""
    a = np.squeeze(np.asarray(a))
    while a.ndim > 2:
        a = a[0]
    return a


def to_16bit(a: np.ndarray) -> np.ndarray:
    if not FORCE_16BIT:
        return a
    if a.dtype == np.uint16:
        return a
    if a.dtype == np.uint8:
        return a.astype(np.uint16) * 257       # 255 -> 65535, preserves range
    a = a.astype(np.float64)                     # float/other: rescale to full 16-bit
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint16)
    return ((a - lo) / (hi - lo) * 65535.0).round().astype(np.uint16)


class CziReader:
    """Lazy CZI reader: downsampled overview + full-res region crops."""

    def __init__(self, path: Path):
        from aicspylibczi import CziFile
        self.czi = CziFile(str(path))
        self.mosaic = self.czi.is_mosaic()
        d = self.czi.get_dims_shape()[0]
        self.n_ch = (d["C"][1] - d["C"][0]) if "C" in d else 1
        if self.mosaic:
            bb = self.czi.get_mosaic_bounding_box()
            self.x0, self.y0, self.W, self.H = bb.x, bb.y, bb.w, bb.h
        else:
            self.x0 = self.y0 = 0
            self.W = d["X"][1] - d["X"][0]
            self.H = d["Y"][1] - d["Y"][0]
            self._cache: dict[int, np.ndarray] = {}

    @property
    def shape(self):
        return (self.n_ch, self.H, self.W)

    def overview(self, scale: float) -> np.ndarray:
        chans = []
        for c in range(self.n_ch):
            if self.mosaic:
                im = self.czi.read_mosaic(C=c, scale_factor=scale)
            else:
                im = _2d(self._full_channel(c))
                step = max(1, int(round(1 / scale)))
                im = im[::step, ::step]
            chans.append(_2d(im))
        return np.stack(chans, axis=0)

    def crop_fullres(self, c, r0, r1, c0, c1) -> np.ndarray:
        if self.mosaic:
            im = self.czi.read_mosaic(
                region=(self.x0 + c0, self.y0 + r0, c1 - c0, r1 - r0),
                scale_factor=1.0, C=c,
            )
            return _2d(im)
        return self._full_channel(c)[r0:r1, c0:c1]

    def _full_channel(self, c: int) -> np.ndarray:
        # non-mosaic only: read each channel once, cache it
        if c not in self._cache:
            im, _ = self.czi.read_image(C=c)
            self._cache[c] = _2d(im)
        return self._cache[c]


class TiffReader:
    """Lazy TIFF reader via a zarr store (handles tiled/compressed/pyramid)."""

    def __init__(self, path: Path):
        import zarr
        self.store = tifffile.imread(path, aszarr=True)
        z = zarr.open(self.store, mode="r")
        # pyramidal OME-TIFF -> group of levels; pick level 0 (largest)
        if hasattr(z, "array_keys"):
            keys = sorted(z.array_keys(), key=lambda k: -int(np.prod(z[k].shape)))
            self.full = z[keys[0]]
        else:
            self.full = z

        shp = self.full.shape
        non1 = [(i, s) for i, s in enumerate(shp) if s > 1]
        self.fixed = {i: 0 for i, s in enumerate(shp) if s == 1}
        if len(non1) <= 2:
            self.ca = None
            self.n_ch = 1
            sp = sorted(i for i, _ in non1)
        else:
            ci = min(non1, key=lambda t: t[1])[0]   # channel = smallest non-singleton axis
            self.ca = ci
            self.n_ch = shp[ci]
            sp = sorted(i for i, _ in non1 if i != ci)
        self.ya, self.xa = sp[0], sp[1]
        self.H, self.W = shp[self.ya], shp[self.xa]

    @property
    def shape(self):
        return (self.n_ch, self.H, self.W)

    def _read(self, c, ysl, xsl) -> np.ndarray:
        idx = [slice(None)] * len(self.full.shape)
        for i, v in self.fixed.items():
            idx[i] = v
        if self.ca is not None:
            idx[self.ca] = c
        idx[self.ya] = ysl
        idx[self.xa] = xsl
        return _2d(self.full[tuple(idx)])

    def overview(self, scale: float) -> np.ndarray:
        step = max(1, int(round(1 / scale)))
        sl = slice(None, None, step)
        return np.stack([self._read(c, sl, sl) for c in range(self.n_ch)], axis=0)

    def crop_fullres(self, c, r0, r1, c0, c1) -> np.ndarray:
        return self._read(c, slice(r0, r1), slice(c0, c1))


class VerticalFlipReader:
    """Wrap a base reader so the slide is mirrored top<->bottom (y-axis).

    A vertical flip is just row reversal, applied at read time so the full slide
    is never loaded: the overview is reversed in memory (it is downsampled), and
    each crop reads the mirror-image row band and reverses it. Every attribute
    other than the two flipped read methods is delegated to the base reader.
    """

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        # only reached when normal lookup fails, i.e. not for the members below
        return getattr(self._base, name)

    @property
    def shape(self):
        return self._base.shape

    def overview(self, scale):
        return self._base.overview(scale)[:, ::-1, :]

    def crop_fullres(self, c, r0, r1, c0, c1):
        H = self._base.shape[1]
        return self._base.crop_fullres(c, H - r1, H - r0, c0, c1)[::-1, :]


def make_reader(path, flip_y=False):
    if Path(path).suffix.lower() == ".czi":
        base = CziReader(Path(path))
    else:
        base = TiffReader(Path(path))
    return VerticalFlipReader(base) if flip_y else base
