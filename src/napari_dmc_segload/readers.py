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
