# Section widget — per-crop horizontal flip

**Date:** 2026-07-27
**Status:** approved (design)
**Component:** `src/napari_dmc_segload/_section_widget.py` (the "cropping tool")

## Problem

The Section widget lets the user draw + number one rectangle per section on a
downsampled overview, then **Export crops** writes full-res per-channel TIFFs
into the DMC-BrainMap tree. Each rectangle is cropped and written as-is. Some
sections are mounted mirror-reversed (left↔right) relative to the atlas and must
be horizontally flipped before registration. Today that has to be done by hand
outside the tool, per section.

## Goal

Let the user mark individual sections for a horizontal (left↔right) flip that is
applied to every channel of that section's crop at export time. Marking is
per-section and visible on the overview.

## Non-goals

- No vertical flip or rotation — horizontal mirror only.
- No global "flip everything" control — the choice is always per-section.
- No change to numbering, ordering, or the output tree layout.

## Design

### Data model

Add a per-shape boolean feature **`flip`** (default `False`) to the `slices`
shapes layer. This is the single source of truth for which sections are
mirrored. A derived string feature **`label`** drives the on-screen text so the
flip marker travels with the number.

The layer is created with both feature columns and matching
`feature_defaults = {"slice": 0, "flip": False}` so napari fills them when a new
rectangle is drawn.

### Interaction — a flip click-mode parallel to numbering

New button **"Toggle flip (click sections)"** calls `_toggle_flip_mode()`:
sets `self._flip_mode = True` and puts the shapes layer in `select` mode.

A `mouse_drag` callback `_click_to_flip(layer, event)` mirrors the existing
`_click_to_number`: on a plain click (not a drag) it resolves the clicked shape
index via `layer.get_value(..., world=True)` and toggles that shape's `flip`
value, then refreshes labels.

Because both numbering and flipping use `select` mode, the two click callbacks
are made **mutually exclusive** by an explicit mode flag:

- `_click_to_number` guards on `self._manual and not self._flip_mode and
  layer.mode == "select"`.
- `_click_to_flip` guards on `self._flip_mode and layer.mode == "select"`.
- `_number_mode()` sets `self._flip_mode = False`.
- `_toggle_flip_mode()` sets `self._flip_mode = True`.
- `_draw_mode()` sets `self._flip_mode = False` (belt-and-braces; add-rectangle
  mode fires neither callback).

`self._flip_mode` is initialised `False` in `__init__` and reset in
`_load_overview`.

### Visual indicator — marker in the label

The `slices` layer's text renders `"{label}"`. `label` is recomputed by a single
helper whenever numbers **or** flips change:

```
label = f"{slice}" + (" ⇄" if flip else "")
```

`0` (unassigned) sections still show `0`, plus `⇄` if flagged. Existing calls
that update numbers (`_set_numbers`, `_refresh_numbers`) and the new flip toggle
all route through this recompute so number and marker never drift apart.

### Feature persistence (load-bearing)

`_set_numbers` currently does `shapes.features = {"slice": ...}`, which
**replaces the whole feature table and would erase `flip`**. It must instead
assign the `slice` column in place (`shapes.features["slice"] = nums`) and then
recompute `label`, leaving `flip` untouched. This is the primary correctness
risk: without it, renumbering (which fires on every shape add/move via
`_refresh_numbers`) silently clears every flip flag.

### Export

`_export` gains a `flips` list aligned to the rectangles, read the same way as
`numbers` (from the layer's `flip` feature, padded to the shape count). Each box
tuple carries its flip flag. After cropping and converting each channel:

```
crop = to_16bit(self.reader.crop_fullres(ci, r0, r1, c0, c1))
if flip:
    crop = np.ascontiguousarray(crop[:, ::-1])   # left↔right mirror
tifffile.imwrite(out_base / label / fname, crop)
```

`np.ascontiguousarray` avoids handing tifffile a negative-stride view. Every
channel of a flipped section is mirrored identically; unflagged sections are
byte-for-byte unchanged.

## Decomposition

- **Pure helper** `flip_label(slice_num, flip) -> str` (module-level): the only
  genuinely pure logic; unit-tested directly.
- Everything else is Qt/napari-bound and covered by widget-level tests using the
  `make_napari_viewer` fixture, matching the existing smoke tests.

## Testing

1. **`flip_label`** — `(3, False) -> "3"`, `(3, True) -> "3 ⇄"`, `(0, True) ->
   "0 ⇄"`.
2. **Export mirrors flagged sections** — write a slide whose pixels vary along
   the x-axis (a horizontal gradient, so `[:, ::-1]` is detectably different),
   draw two rectangles, set one's `flip` feature `True`, export, read both TIFFs
   back: the flagged section's pixels equal `original[:, ::-1]`, the unflagged
   one equals `original`. Assert across all channels for the flagged section.
3. **Flip survives renumbering** — set a shape's `flip` True, trigger a
   renumber (`_refresh_numbers` / add a shape), assert the flag is still set and
   its label still ends in `⇄`.
4. **Mode guard** — in flip mode, a click toggles `flip` but does not change
   `slice`; in number mode, a click sets `slice` but does not toggle `flip`.

## Risks

- **Feature-table replacement** wiping `flip` (mitigated by in-place column
  assignment in `_set_numbers`; test 3 covers it).
- **Mode ambiguity** between numbering and flipping clicks (mitigated by the
  `_flip_mode` guards; test 4 covers it).
- **Negative-stride view** to tifffile (mitigated by `np.ascontiguousarray`).
