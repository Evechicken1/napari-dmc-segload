# Section-Slides Napari Widget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second napari dock widget, `SectionWidget`, to the existing `napari-dmc-segload` package that ports the standalone `Histology/dmc_section.py` sectioning tool — draw rectangles on a downsampled overview, number the sections, and export full-resolution single-channel 16-bit TIFFs in the DMC-BrainMap input layout.

**Architecture:** Pure I/O (readers, dtype conversion) goes in `readers.py`; the pure ordering logic goes in `numbering.py`; the Qt dock widget in `_section_widget.py` owns the viewer napari hands it and wires form + buttons + layer callbacks. This mirrors how the existing `SegLoaderWidget` separates `io.py` from `_widget.py`.

**Tech Stack:** napari, magicgui, qtpy, numpy, tifffile, zarr, aicspylibczi; pytest + pytest-qt for tests.

---

## File Structure

- Create: `src/napari_dmc_segload/readers.py` — `_2d`, `to_16bit`, `CziReader`, `TiffReader`, `make_reader`; constants `FORCE_16BIT`, `ZERO_PAD`.
- Create: `src/napari_dmc_segload/numbering.py` — `compute_slice_numbers(rects, ov_h, order_by, row_tol=ROW_TOL)`; constant `ROW_TOL`.
- Create: `src/napari_dmc_segload/_section_widget.py` — `SectionWidget(QWidget)`; constant `DISPLAY_CMAP` (section labels → colours).
- Modify: `src/napari_dmc_segload/napari.yaml` — register the second command + widget.
- Modify: `pyproject.toml` — add `tifffile`, `zarr`, `aicspylibczi` to dependencies.
- Test: `tests/test_readers.py`, `tests/test_numbering.py`, `tests/test_section_widget_smoke.py`.

The original `Histology/dmc_section.py` is left untouched as a standalone fallback.

---

## Task 1: Dependencies + dtype conversion (`readers.py` foundation)

**Files:**
- Modify: `pyproject.toml:11-18`
- Create: `src/napari_dmc_segload/readers.py`
- Test: `tests/test_readers.py`

- [ ] **Step 1: Add runtime dependencies**

In `pyproject.toml`, extend the `dependencies` list so it reads:

```toml
dependencies = [
    "napari",
    "numpy",
    "pandas",
    "natsort",
    "magicgui",
    "qtpy",
    "tifffile",
    "zarr",
    "aicspylibczi",
]
```

- [ ] **Step 2: Write the failing test for `to_16bit`**

Create `tests/test_readers.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_readers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'napari_dmc_segload.readers'`

- [ ] **Step 4: Create `readers.py` with `_2d` and `to_16bit`**

Create `src/napari_dmc_segload/readers.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_readers.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/napari_dmc_segload/readers.py tests/test_readers.py
git commit -m "feat: add readers module with dtype helpers + section deps"
```

---

## Task 2: `TiffReader` + `make_reader` (+ `CziReader`)

**Files:**
- Modify: `src/napari_dmc_segload/readers.py`
- Test: `tests/test_readers.py`

- [ ] **Step 1: Write the failing test for `TiffReader` + `make_reader`**

Append to `tests/test_readers.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_readers.py -k "tiffreader or make_reader" -v`
Expected: FAIL with `ImportError: cannot import name 'TiffReader'`

- [ ] **Step 3: Append the reader classes + `make_reader` to `readers.py`**

Append to `src/napari_dmc_segload/readers.py`:

```python
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


def make_reader(path: Path):
    if Path(path).suffix.lower() == ".czi":
        return CziReader(Path(path))
    return TiffReader(Path(path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_readers.py -v`
Expected: PASS (all readers tests green)

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/readers.py tests/test_readers.py
git commit -m "feat: add CziReader/TiffReader with make_reader dispatch"
```

---

## Task 3: Ordering logic (`numbering.py`)

**Files:**
- Create: `src/napari_dmc_segload/numbering.py`
- Test: `tests/test_numbering.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_numbering.py`:

```python
import numpy as np
from napari_dmc_segload.numbering import compute_slice_numbers


def _rect(y0, x0, y1, x1):
    """Rectangle as napari stores it: 4 corner rows of (row, col)."""
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def test_empty_returns_empty():
    assert compute_slice_numbers([], ov_h=100, order_by="grid") == []


def test_draw_order_is_input_order():
    rects = [_rect(50, 0, 60, 10), _rect(0, 0, 10, 10)]
    assert compute_slice_numbers(rects, ov_h=100, order_by="draw") == [1, 2]


def test_vertical_orders_by_top_edge():
    # second rect is higher up -> should be numbered 1
    rects = [_rect(50, 0, 60, 10), _rect(0, 0, 10, 10)]
    assert compute_slice_numbers(rects, ov_h=100, order_by="vertical") == [2, 1]


def test_grid_rows_top_to_bottom_left_to_right():
    # two rects on top row (left, right), one on bottom row
    top_left = _rect(0, 0, 10, 10)
    top_right = _rect(0, 50, 10, 60)
    bottom = _rect(80, 0, 90, 10)
    nums = compute_slice_numbers([top_right, top_left, bottom],
                                 ov_h=100, order_by="grid")
    # input order: top_right, top_left, bottom
    assert nums == [2, 1, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_numbering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'napari_dmc_segload.numbering'`

- [ ] **Step 3: Create `numbering.py`**

Create `src/napari_dmc_segload/numbering.py`:

```python
"""Slice-numbering strategies for the section widget.

Pure functions — no napari. `order_by` and `row_tol` were module globals in the
original dmc_section.py script; here they are explicit parameters.
"""

from __future__ import annotations

import numpy as np

ROW_TOL = 0.05   # grid mode: centroids within this fraction of overview height = same row


def compute_slice_numbers(rects, ov_h: int, order_by: str,
                          row_tol: float = ROW_TOL) -> list[int]:
    """Return a slice number per rectangle, in the SAME order as `rects`.

    Numbering follows `order_by` ("draw" / "vertical" / "grid"). Centroids are in
    overview (row, col) coords. `order_by == "manual"` is handled by the widget,
    not here; passing it falls through to the "grid" branch.
    """
    n = len(rects)
    if n == 0:
        return []
    cents = [(v[:, 0].mean(), v[:, 1].mean()) for v in rects]
    idx = list(range(n))

    if order_by == "draw":
        order = idx
    elif order_by == "vertical":
        order = sorted(idx, key=lambda i: cents[i][0])
    else:  # "grid": rows top->bottom, left->right within a row
        tol = row_tol * ov_h
        rows: list[list[int]] = []
        for i in sorted(idx, key=lambda i: cents[i][0]):
            for row in rows:
                if abs(cents[i][0] - cents[row[0]][0]) <= tol:
                    row.append(i)
                    break
            else:
                rows.append([i])
        order = [i for row in rows for i in sorted(row, key=lambda i: cents[i][1])]

    numbers = [0] * n
    for slice_no, i in enumerate(order, start=1):
        numbers[i] = slice_no
    return numbers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_numbering.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/numbering.py tests/test_numbering.py
git commit -m "feat: add slice-numbering strategies module"
```

---

## Task 4: The dock widget (`_section_widget.py`)

**Files:**
- Create: `src/napari_dmc_segload/_section_widget.py`
- Test: `tests/test_section_widget_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_section_widget_smoke.py`:

```python
import numpy as np
import tifffile
from napari_dmc_segload._section_widget import SectionWidget


def _write_multichannel_tif(path, n_ch=3, h=40, w=60):
    arr = np.stack(
        [np.full((h, w), (c + 1) * 1000, dtype=np.uint16) for c in range(n_ch)],
        axis=0,
    )
    tifffile.imwrite(str(path), arr)
    return arr


def _rect(y0, x0, y1, x1):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def test_load_adds_image_and_slices_layers(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    _write_multichannel_tif(p, n_ch=3)
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.overview_scale.value = 0.1
    w._load_overview()
    names = [ly.name for ly in viewer.layers]
    assert "green" in names and "cy3" in names and "dapi" in names
    assert "slices" in names


def test_export_writes_dmc_tree(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    _write_multichannel_tif(p, n_ch=3, h=40, w=60)
    out_root = tmp_path / "dmc_input"
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.animal_id.value = "m1"
    w.form.output_root.value = str(out_root)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.order_by.value = "draw"   # avoid manual click-numbering in a headless test
    w.form.overview_scale.value = 0.1
    w._load_overview()
    # one rectangle covering the whole 4x6 overview -> full 40x60 crop
    viewer.layers["slices"].data = [_rect(0, 0, 4, 6)]
    w._export()
    for label in ("green", "cy3", "dapi"):
        f = out_root / "m1" / "stitched" / label / "m1_s001.tif"
        assert f.exists(), f"missing {f}"
    crop = tifffile.imread(str(out_root / "m1" / "stitched" / "cy3" / "m1_s001.tif"))
    assert crop.dtype == np.uint16 and crop.shape == (40, 60)
    # channel 1 constant value 2000 preserved
    assert np.all(crop == 2000)
    # section layers cleared after export
    assert "slices" not in [ly.name for ly in viewer.layers]


def test_export_without_load_is_a_noop(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w._export()  # must not raise
    assert len(viewer.layers) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_section_widget_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'napari_dmc_segload._section_widget'`

- [ ] **Step 3: Create `_section_widget.py`**

Create `src/napari_dmc_segload/_section_widget.py`:

```python
"""Dock widget: section a whole-slide image into per-slice 16-bit TIFFs.

Ported from Histology/dmc_section.py. napari is already running and hands this
widget a viewer; the widget builds a downsampled overview, lets the user draw +
number one rectangle per section, and writes full-res crops into the tree
DMC-BrainMap expects:

    <output_root>/<animal_id>/stitched/<channel>/<animal_id>_sNNN.tif
"""

from pathlib import Path

import numpy as np
import tifffile
from napari import Viewer
from napari.utils.notifications import show_info
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .readers import make_reader, to_16bit, ZERO_PAD
from .numbering import compute_slice_numbers

# napari display colour per DMC label (purely cosmetic).
DISPLAY_CMAP = {"dapi": "blue", "green": "green", "n3": "cyan",
                "cy3": "red", "cy5": "magenta"}


def _build_form() -> FunctionGui:
    @magicgui(
        layout="vertical",
        input_path=dict(widget_type="FileEdit", mode="r",
                        filter="*.czi *.tif *.tiff", label="input image"),
        animal_id=dict(widget_type="LineEdit", label="animal_id"),
        output_root=dict(widget_type="FileEdit", mode="d", label="output root"),
        channel_labels=dict(widget_type="LineEdit", label="channel labels",
                            value="green,cy3,dapi"),
        order_by=dict(widget_type="ComboBox", label="order by",
                      choices=["manual", "grid", "draw", "vertical"],
                      value="manual"),
        overview_scale=dict(widget_type="FloatSpinBox", label="overview scale",
                            value=0.1, min=0.001, max=1.0, step=0.01),
        call_button=False,
    )
    def form(input_path, animal_id, output_root, channel_labels,
             order_by, overview_scale):
        pass

    # auto-fill animal_id from the chosen file's stem
    @form.input_path.changed.connect
    def _fill_animal_id(value):
        if value and not form.animal_id.value:
            form.animal_id.value = Path(str(value)).stem

    return form


class SectionWidget(QWidget):
    """Dock widget for sectioning whole-slide images into DMC input TIFFs."""

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        self.form = _build_form()
        self.layout().addWidget(self.form.native)

        for text, slot in [
            ("Load overview", self._load_overview),
            ("Draw rectangles", self._draw_mode),
            ("Number sections (click in order)", self._number_mode),
            ("Reset numbering", self._reset_numbering),
            ("Export crops", self._export),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            self.layout().addWidget(btn)

        # per-load state (None until a slide is loaded)
        self.reader = None
        self.labels = []
        self.H = self.W = self.ov_h = 0
        self.fy = self.fx = 1.0
        self._order_by = "manual"
        self._manual = False
        self._counter = 1

    # --- helpers -----------------------------------------------------------
    @property
    def _shapes(self):
        return self.viewer.layers["slices"] if "slices" in self.viewer.layers else None

    def _get_numbers(self):
        shapes = self._shapes
        if shapes is None:
            return []
        n = len(shapes.data)
        if self._manual:
            cur = [int(x) for x in list(shapes.features.get("slice", []))]
            return (cur + [0] * n)[:n]             # unassigned shapes show 0
        return compute_slice_numbers([np.asarray(s) for s in shapes.data],
                                     self.ov_h, self._order_by)

    def _set_numbers(self, nums):
        shapes = self._shapes
        if shapes is None:
            return
        shapes.features = {"slice": np.asarray(nums, dtype=int)}
        shapes.refresh_text()

    def _refresh_numbers(self, *_):
        self._set_numbers(self._get_numbers())

    # --- actions -----------------------------------------------------------
    def _load_overview(self):
        in_path = self.form.input_path.value
        if not in_path or not Path(str(in_path)).is_file():
            show_info("Select a valid input image (.czi/.tif/.tiff).")
            return
        in_path = Path(str(in_path))

        self.reader = make_reader(in_path)
        n_ch, self.H, self.W = self.reader.shape
        labels = [s.strip() for s in str(self.form.channel_labels.value).split(",")
                  if s.strip()]
        if len(labels) != n_ch:
            show_info(f"image has {n_ch} channels but {len(labels)} labels; "
                      f"using {labels[:n_ch]}")
        self.labels = labels[:n_ch]

        scale = float(self.form.overview_scale.value)
        ov = self.reader.overview(scale)          # (C, y, x)
        self.ov_h, ov_w = ov.shape[1], ov.shape[2]
        self.fy, self.fx = self.H / self.ov_h, self.W / ov_w

        # clear any previous section layers, then add fresh ones
        self._remove_section_layers()
        for ci, label in enumerate(self.labels):
            self.viewer.add_image(
                ov[ci], name=label, colormap=DISPLAY_CMAP.get(label, "gray"),
                blending="additive",
                contrast_limits=(float(ov[ci].min()), float(ov[ci].max())),
            )
        shapes = self.viewer.add_shapes(
            name="slices", shape_type="rectangle",
            edge_color="yellow", edge_width=4, face_color="transparent",
            features={"slice": np.zeros(0, dtype=int)},
            text={"string": "{slice}", "size": 14, "color": "white",
                  "anchor": "center"},
        )
        shapes.mode = "add_rectangle"

        self._order_by = str(self.form.order_by.value)
        self._manual = self._order_by == "manual"
        self._counter = 1
        shapes.events.data.connect(self._refresh_numbers)
        shapes.mouse_drag_callbacks.append(self._click_to_number)
        show_info(f"loaded {in_path.name}: {n_ch}ch {self.H}x{self.W}")

    def _click_to_number(self, layer, event):
        # only act on a plain click while in manual-numbering (select) mode
        if not self._manual or layer.mode != "select":
            return
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged:                                # ignore box moves
            return
        val = layer.get_value(event.position, world=True)
        idx = val[0] if isinstance(val, (tuple, list)) else val
        if idx is None:
            return
        nums = self._get_numbers()
        nums[int(idx)] = self._counter
        self._counter += 1
        self._set_numbers(nums)

    def _draw_mode(self):
        if self._shapes is None:
            show_info("Load an overview first.")
            return
        self._shapes.mode = "add_rectangle"

    def _number_mode(self):
        if self._shapes is None:
            show_info("Load an overview first.")
            return
        self._manual = True
        self._counter = 1
        self._set_numbers([0] * len(self._shapes.data))
        self._shapes.mode = "select"

    def _reset_numbering(self):
        if self._shapes is None:
            return
        self._counter = 1
        self._set_numbers([0] * len(self._shapes.data))

    def _remove_section_layers(self):
        for name in list(self.labels) + ["slices"]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])

    def _export(self):
        if self.reader is None or self._shapes is None:
            show_info("Load an overview first.")
            return
        rects = [np.asarray(s) for s in self._shapes.data]
        if not rects:
            show_info("No rectangles drawn.")
            return

        numbers = self._get_numbers()
        if self._manual:
            if any(x <= 0 for x in numbers):
                show_info("Some sections are unnumbered (showing 0); number them all.")
                return
            if sorted(numbers) != list(range(1, len(numbers) + 1)):
                show_info(f"Numbers are not a clean 1..N sequence: {sorted(numbers)}.")
                return

        boxes = []
        for v, num in zip(rects, numbers):
            ys, xs = v[:, 0], v[:, 1]              # overview coords (row, col)
            r0 = int(np.clip(round(ys.min() * self.fy), 0, self.H))
            r1 = int(np.clip(round(ys.max() * self.fy), 0, self.H))
            c0 = int(np.clip(round(xs.min() * self.fx), 0, self.W))
            c1 = int(np.clip(round(xs.max() * self.fx), 0, self.W))
            boxes.append((num, r0, r1, c0, c1))
        boxes.sort(key=lambda b: b[0])

        animal_id = str(self.form.animal_id.value).strip() or "animal"
        out_root = Path(str(self.form.output_root.value)) if self.form.output_root.value \
            else Path.cwd() / "dmc_input"
        out_base = out_root / animal_id / "stitched"
        for label in self.labels:
            (out_base / label).mkdir(parents=True, exist_ok=True)

        n = 0
        for num, r0, r1, c0, c1 in boxes:
            if r1 <= r0 or c1 <= c0:
                continue
            fname = f"{animal_id}_s{str(num).zfill(ZERO_PAD)}.tif"
            for ci, label in enumerate(self.labels):
                crop = to_16bit(self.reader.crop_fullres(ci, r0, r1, c0, c1))
                tifffile.imwrite(out_base / label / fname, crop)
            n += 1

        show_info(f"wrote {n} slices -> {out_base}")
        if n > 0:
            self._remove_section_layers()          # clear for the next slide
            self.reader = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_section_widget_smoke.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/_section_widget.py tests/test_section_widget_smoke.py
git commit -m "feat: add SectionWidget dock widget"
```

---

## Task 5: Register the widget in the manifest

**Files:**
- Modify: `src/napari_dmc_segload/napari.yaml`

- [ ] **Step 1: Add the command + widget to `napari.yaml`**

Replace the contents of `src/napari_dmc_segload/napari.yaml` with:

```yaml
name: napari-dmc-segload
display_name: DMC Segmentation Loader
contributions:
  commands:
    - id: napari-dmc-segload.SegLoaderWidget
      python_name: napari_dmc_segload._widget:SegLoaderWidget
      title: Open Segmentation Loader
    - id: napari-dmc-segload.SectionWidget
      python_name: napari_dmc_segload._section_widget:SectionWidget
      title: Section slides
  widgets:
    - command: napari-dmc-segload.SegLoaderWidget
      display_name: Segmentation loader
    - command: napari-dmc-segload.SectionWidget
      display_name: Section slides
```

- [ ] **Step 2: Verify the plugin manifest is valid + widget is discoverable**

Run: `python -c "from npe2 import PluginManifest; m = PluginManifest.from_file('src/napari_dmc_segload/napari.yaml'); print([w.display_name for w in m.contributions.widgets])"`
Expected: prints `['Segmentation loader', 'Section slides']`

(If `npe2` is unavailable, instead run `python -c "import napari_dmc_segload._section_widget as m; print(m.SectionWidget)"` to confirm the `python_name` path imports.)

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest -v`
Expected: all tests pass (existing segload tests + the three new test files).

- [ ] **Step 4: Commit**

```bash
git add src/napari_dmc_segload/napari.yaml
git commit -m "feat: register Section slides widget in napari manifest"
```

---

## Manual verification (after automated tasks)

1. `pip install -e .` in the `napari-dmc-segload` repo (inside the napari env).
2. Launch `napari`, open **Plugins → DMC Segmentation Loader → Section slides**.
3. Pick a real whole-slide `.czi` or `.tif`, set `animal_id` + output root, click **Load overview**.
4. Draw one rectangle per section → **Number sections** → click them in order → **Export crops**.
5. Confirm `<output_root>/<animal_id>/stitched/<label>/<animal_id>_sNNN.tif` files exist and open at full resolution, and that the section layers cleared after export.

---

## Self-Review Notes

- **Spec coverage:** readers.py (Task 1–2), numbering.py (Task 3), _section_widget.py + form controls + buttons + post-export clear (Task 4), manifest (Task 5), pyproject deps (Task 1), tests for all three modules (Tasks 1–4). Constants kept in code: `_CMAP`→`DISPLAY_CMAP`, `ZERO_PAD`, `FORCE_16BIT`, `ROW_TOL`. All spec sections mapped.
- **Naming consistency:** `_load_overview`, `_draw_mode`, `_number_mode`, `_reset_numbering`, `_export`, `_remove_section_layers`, `_get_numbers`/`_set_numbers`/`_refresh_numbers`, `_click_to_number` used consistently across widget + tests.
- **Deviation from script (intended, per spec):** no `napari.run()`/`sys.exit()`/tkinter; post-export removes section layers instead of closing the viewer; `order_by`/`channel_labels`/`overview_scale`/paths are form controls, not module globals.
