# Whole-slide Vertical Flip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the section widget crop a vertically-flipped (top↔bottom) whole slide via a load-time checkbox, without ever loading the full slide.

**Architecture:** A `VerticalFlipReader` adapter wraps any base reader (`CziReader`/`TiffReader`) and mirrors the y-axis at the two read methods (`overview`, `crop_fullres`); `make_reader` gains a `flip_y` flag that wraps when set. The section widget gets a `flip vertically` checkbox whose value is passed to `make_reader` at load — nothing else in the widget changes.

**Tech Stack:** Python, numpy, tifffile, magicgui/qtpy, napari, pytest. Conda env `napari-env`. Run tests from the main checkout with `python -m pytest`; from a worktree prepend `PYTHONPATH="<worktree>/src"`.

**Repo:** `C:\Users\xaand\Documents\PhD\Analysis\Informatics\Python\napari-dmc-segload`

---

### Task 1: `VerticalFlipReader` adapter + `make_reader(flip_y=...)`

**Files:**
- Modify: `src/napari_dmc_segload/readers.py` (add `VerticalFlipReader`; add `flip_y` to `make_reader`)
- Test: `tests/test_readers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_readers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_readers.py -v -k "vertical_flip or flip_y"`
Expected: FAIL — `ImportError: cannot import name 'VerticalFlipReader'`, and (once that import is added) `make_reader()` rejecting the `flip_y` kwarg.

- [ ] **Step 3: Add `VerticalFlipReader` and extend `make_reader`**

In `src/napari_dmc_segload/readers.py`, add this class immediately before `make_reader`:

```python
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
```

Then replace `make_reader` with:

```python
def make_reader(path, flip_y=False):
    if Path(path).suffix.lower() == ".czi":
        base = CziReader(Path(path))
    else:
        base = TiffReader(Path(path))
    return VerticalFlipReader(base) if flip_y else base
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_readers.py -v`
Expected: PASS (all reader tests, including the 6 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/readers.py tests/test_readers.py
git commit -m "feat: add VerticalFlipReader and make_reader(flip_y=...)"
```

---

### Task 2: `flip vertically` checkbox in the section widget

**Files:**
- Modify: `src/napari_dmc_segload/_section_widget.py` (form checkbox + pass `flip_y` at load)
- Test: `tests/test_section_widget_smoke.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_section_widget_smoke.py`:

```python
from napari_dmc_segload.readers import make_reader


def _write_topbottom_tif(path, n_ch=3, h=40, w=60):
    # top half a distinct value from the bottom half, per channel, so a
    # vertical flip is detectable in both the overview and an exported crop
    arr = np.zeros((n_ch, h, w), dtype=np.uint16)
    for c in range(n_ch):
        arr[c, :h // 2, :] = (c + 1) * 1000          # top half
        arr[c, h // 2:, :] = (c + 1) * 1000 + 500    # bottom half
    tifffile.imwrite(str(path), arr)
    return arr


def test_flip_vertical_reverses_displayed_overview(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    _write_topbottom_tif(p, n_ch=3, h=40, w=60)
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.overview_scale.value = 0.1
    w.form.flip_vertical.value = True
    w._load_overview()
    plain_ov = make_reader(p, flip_y=False).overview(0.1)   # (3, 4, 6)
    # cy3 is channel index 1; the displayed layer is its rows reversed
    np.testing.assert_array_equal(
        np.asarray(viewer.layers["cy3"].data), plain_ov[1][::-1, :])


def test_flip_vertical_export_pulls_from_bottom(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    _write_topbottom_tif(p, n_ch=3, h=40, w=60)
    out_root = tmp_path / "dmc_input"
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.animal_id.value = "m1"
    w.form.output_root.value = str(out_root)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.order_by.value = "draw"
    w.form.overview_scale.value = 0.1
    w.form.flip_vertical.value = True
    w._load_overview()
    # box over the TOP two overview rows (overview 4x6 -> full rows 0..20);
    # with flip on, that reads the BOTTOM half of the source
    viewer.layers["slices"].data = [_rect(0, 0, 2, 6)]
    w._export()
    crop = tifffile.imread(str(out_root / "m1" / "stitched" / "cy3" / "m1_s001.tif"))
    assert np.all(crop == 2500)          # cy3 bottom-half value


def test_no_flip_export_pulls_from_top(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    _write_topbottom_tif(p, n_ch=3, h=40, w=60)
    out_root = tmp_path / "dmc_input"
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.animal_id.value = "m1"
    w.form.output_root.value = str(out_root)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.order_by.value = "draw"
    w.form.overview_scale.value = 0.1
    w.form.flip_vertical.value = False
    w._load_overview()
    viewer.layers["slices"].data = [_rect(0, 0, 2, 6)]
    w._export()
    crop = tifffile.imread(str(out_root / "m1" / "stitched" / "cy3" / "m1_s001.tif"))
    assert np.all(crop == 2000)          # cy3 top-half value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_section_widget_smoke.py -v -k "flip_vertical or no_flip"`
Expected: FAIL — `AttributeError`/`ValueError` because `w.form.flip_vertical` does not exist yet.

- [ ] **Step 3: Add the checkbox to the form**

In `src/napari_dmc_segload/_section_widget.py`, inside `_build_form`'s `@magicgui(...)` decorator, add a `flip_vertical` control right after the `overview_scale` entry:

```python
        overview_scale=dict(widget_type="FloatSpinBox", label="overview scale",
                            value=0.1, min=0.001, max=1.0, step=0.01),
        flip_vertical=dict(widget_type="CheckBox", label="flip vertically",
                           value=False),
        call_button=False,
    )
    def form(input_path, animal_id, output_root, channel_labels,
             order_by, overview_scale, flip_vertical):
        pass
```

(The `def form(...)` signature must gain the `flip_vertical` parameter as shown.)

- [ ] **Step 4: Pass `flip_y` when building the reader**

In `_load_overview`, replace:

```python
        self.reader = make_reader(in_path)
```

with:

```python
        self.reader = make_reader(in_path, flip_y=bool(self.form.flip_vertical.value))
```

- [ ] **Step 5: Run the widget smoke module**

Run: `pytest tests/test_section_widget_smoke.py -v`
Expected: PASS (all, including the 3 new tests and the pre-existing ones).

- [ ] **Step 6: Commit**

```bash
git add src/napari_dmc_segload/_section_widget.py tests/test_section_widget_smoke.py
git commit -m "feat: add 'flip vertically' checkbox to the section widget"
```

---

### Task 3: full-suite regression

**Files:** none (verification only)

- [ ] **Step 1: Run the whole suite**

Run: `pytest` (from the worktree: `PYTHONPATH="<worktree>/src" "<napari-env python>" -m pytest`)
Expected: PASS — the prior 153 tests plus the tests added here (Task 1: 6 new; Task 2: 3 new), no failures.

- [ ] **Step 2: If anything fails, stop and diagnose**

Do not paper over a failure. Use superpowers:systematic-debugging on the first failing test before continuing.

---

## Notes for the implementer

- **No full load:** the flip never materializes the whole slide — `overview` reverses the already-downsampled thumbnail, and `crop_fullres` reverses only the requested band. Do not add any code that reads the full image.
- **Reader-agnostic:** the adapter relies only on `shape`/`overview`/`crop_fullres`, so it works for both `TiffReader` and `CziReader`. Do not special-case a format.
- **Coordinate remap:** `_export` already clips `r0`/`r1` to `[0, H]`; `H - r1 <= H - r0` stays in range, so a full-height crop (`r0=0, r1=H`) maps to the whole source reversed. No extra clamping needed.
- **Composition:** the existing per-section horizontal flip (`crop[:, ::-1]` in `_export`) is untouched and still applies on top of a vertically flipped slide.
