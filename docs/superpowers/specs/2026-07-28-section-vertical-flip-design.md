# Whole-slide vertical flip for sectioning

**Date:** 2026-07-28
**Status:** Design approved, pending implementation
**Component:** Section widget / whole-slide readers (`readers.py`, `_section_widget.py`)

## Problem

Some whole-slide images are scanned upside down (dorsoventrally inverted). The
user needs to section (crop numbered per-slice TIFFs from) such a slide with the
correct orientation, without being able to load the full slide into memory to
flip it first — these images are large and only ever read region-by-region.

Today the section widget has a *per-section horizontal* flip applied on export
(`crop[:, ::-1]`), but no way to flip the *whole slide vertically* (top↔bottom)
before sectioning.

## Chosen approach: mirror rows at the reader boundary

Every read in the section pipeline goes through exactly two reader methods —
`overview(scale)` (the downsampled thumbnail the user draws boxes on) and
`crop_fullres(c, r0, r1, c0, c1)` (the per-section full-res crop). A vertical
flip is "reverse row order," so we apply it at that boundary with a thin adapter
that wraps whatever `make_reader` returns. The rest of the widget — overview
display, box drawing, numbering, export, and the existing per-section horizontal
flip — is unchanged and unaware of the flip.

This never loads the full slide: the overview is already downsampled, and each
crop reads only its own (mirror-image) row band.

Rejected alternative: writing a physically flipped TIFF to disk (streaming,
tiled). More work (a full write path) and unnecessary for sectioning, since the
crops are the actual output.

Rejected alternative: a live "flip in place" toggle button. Rectangles drawn
before the flip would be in the old orientation and need remapping. A load-time
checkbox (decide before drawing) is simpler and less error-prone.

## Components

### 1. `VerticalFlipReader` adapter (`readers.py`)

A proxy wrapping a base reader (`CziReader` or `TiffReader`), mirroring the
y-axis:

- `shape` → the base's `(n_ch, H, W)`, unchanged.
- `overview(scale)` → `base.overview(scale)[:, ::-1, :]` (reverse the thumbnail's
  rows; it is small/downsampled, so an in-memory reverse is cheap).
- `crop_fullres(c, r0, r1, c0, c1)` →
  `base.crop_fullres(c, H - r1, H - r0, c0, c1)[::-1, :]`, where `H` is the
  base's full-res height. Reads the mirror-image row band `[H-r1 : H-r0]` and
  reverses it; columns are untouched. Memory stays one-crop-sized.
- `__getattr__` delegates any other attribute to the base reader, so the adapter
  is a transparent proxy apart from the two flipped methods.

The adapter depends only on the base exposing `shape`/`overview`/`crop_fullres`
(the shared reader interface), so it is reader-agnostic (TIFF and CZI alike).

### 2. `make_reader(path, flip_y=False)`

Add a `flip_y` keyword. When false, behavior is unchanged. When true, build the
base reader exactly as today, then return it wrapped in `VerticalFlipReader`.
Single construction point for both formats.

### 3. Form checkbox + wiring (`_section_widget.py`)

- Add `flip_vertical` to the form: `dict(widget_type="CheckBox",
  label="flip vertically", value=False)`, alongside `order_by` / `overview_scale`.
- In `_load_overview`, construct the reader as
  `make_reader(in_path, flip_y=bool(self.form.flip_vertical.value))`.

No other widget change. The scale factors `fy`/`fx`, box drawing, numbering, and
`_export`'s overview→full-res coordinate mapping all keep working because they
operate through the (now flipped) reader and its `shape`.

## Interaction with the existing per-section horizontal flip

Orthogonal and composable. The slide-wide vertical flip happens at read time
(rows); the per-section horizontal flip still happens on export (`crop[:, ::-1]`,
columns). A section flagged for horizontal flip on a vertically flipped slide is
effectively rotated 180°. Both mechanisms are independent and unchanged.

## Edge cases

- **Coordinate bounds:** `_export` already clips `r0`/`r1` to `[0, H]`. Then
  `H - r1` and `H - r0` are also in `[0, H]` and correctly ordered
  (`H - r1 <= H - r0`), so the mirrored band is valid; a full-height crop
  (`r0=0, r1=H`) maps to the full source reversed.
- **Off:** `flip_y=False` returns the base reader untouched — a strict no-op, so
  existing behavior and tests are unaffected.

## Testing

- **Adapter unit tests** against a fake base reader holding a known array
  (`shape`, `overview`, `crop_fullres` returning slices of that array):
  - `overview` rows are reversed vs the base.
  - `crop_fullres` of an off-center band equals `np.flipud` of the base's crop of
    the mirrored band `[H-r1:H-r0]` (exact pixels).
  - a full-height crop equals the base's full crop reversed.
  - `__getattr__` delegates an unknown attribute to the base.
- **`make_reader(path, flip_y=True)`** on a small real multi-channel TIFF written
  to a tmp path (mirroring `tests/test_readers.py`): its `overview` equals the
  `flip_y=False` overview reversed along rows; `flip_y=False` returns a plain
  base reader.
- **Widget smoke** (`tests/test_section_widget_smoke.py` style): load a synthetic
  slide with the checkbox on; assert the displayed overview layer equals the
  unflipped overview reversed, and that a fixed rectangle near the top of the
  drawn overview exports source content from the *bottom* of the slide when the
  checkbox is on (and the top when off).
