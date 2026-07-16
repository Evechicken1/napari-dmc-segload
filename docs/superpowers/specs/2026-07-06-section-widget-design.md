# napari-dmc-section widget — design

**Date:** 2026-07-06
**Status:** Approved (pending spec review)

## Goal

Turn the working standalone script `Histology/dmc_section.py` into a napari dock
widget, delivered as a **second widget inside the existing `napari-dmc-segload`
package**. The tool sections a whole-slide image (all slices for one mouse) into
per-slice, single-channel 16-bit TIFFs laid out in the tree DMC-BrainMap expects:

```
<output_root>/<animal_id>/stitched/<channel>/<animal_id>_sNNN.tif
```

The user draws one rectangle per section on a downsampled overview, numbers the
sections, and exports; each rectangle's bounding box is read back at **full
resolution** and written out. Sectioning is the prep step that produces the input
tree that the existing `segload` widget later loads for tracing.

## Why a widget (the inversion)

The standalone script:
- is launched from the CLI, uses tkinter dialogs for file / animal_id / output,
- **creates its own** `napari.Viewer`, adds image + shapes layers to it,
- calls `napari.run()`, and on export optionally closes the viewer and
  `sys.exit(0)`.

A plugin inverts control: napari is already running and hands the widget a
viewer. So the tkinter dialogs become form controls, `napari.run()` /
`sys.exit(0)` disappear, and the widget operates on `self.viewer` (the viewer
napari passes to its constructor).

## Architecture

Three new files in `src/napari_dmc_segload/`, plus edits to two existing files.

| File | Purpose |
|------|---------|
| `readers.py` *(new)* | `CziReader`, `TiffReader`, `make_reader`, `_2d`, `to_16bit`, module constants `FORCE_16BIT`, `ZERO_PAD`. Lifted verbatim from `dmc_section.py`. Pure I/O, no napari import. Independently testable. |
| `numbering.py` *(new)* | `compute_slice_numbers(rects, ov_h, order_by, row_tol=ROW_TOL)` — grid / draw / vertical ordering logic. Currently reads module globals `ORDER_BY` and `ROW_TOL`; these become an explicit parameter and a default. Pure function, no napari. Independently testable. |
| `_section_widget.py` *(new)* | `SectionWidget(QWidget)` — the dock widget: magicgui form + buttons, owns the viewer, ports the live text-numbering events, the manual click-to-number mouse callback, and the export/validation logic. |
| `napari.yaml` *(edit)* | Register a second command + widget ("Section slides"). |
| `pyproject.toml` *(edit)* | Add `tifffile`, `zarr`, `aicspylibczi` to `dependencies`. |

Constants that stay in code (not exposed in the UI): `_CMAP` (cosmetic per-label
display colours), `ZERO_PAD` (=3, filename padding), `FORCE_16BIT` (=True),
`ROW_TOL` (=0.05, grid row tolerance).

The original `Histology/dmc_section.py` is left untouched as a standalone
fallback.

## Widget form & flow

### Form controls (magicgui)

All four configurable groups the user selected become live controls:

| Control | Widget | Default |
|---------|--------|---------|
| `input_path` | `FileEdit` (file mode), filter `*.czi *.tif *.tiff` | — |
| `animal_id` | `LineEdit`, auto-filled from `input_path` stem when the file is chosen | image stem |
| `output_root` | `FileEdit` (directory mode) | — |
| `channel_labels` | `LineEdit`, comma-separated DMC labels in channel-index order | `green,cy3,dapi` |
| `order_by` | `ComboBox`: `manual` / `grid` / `draw` / `vertical` | `manual` |
| `overview_scale` | `FloatSpinBox` (min small, step 0.01) | `0.1` |

`channel_labels` is parsed to a list on load; if the image has fewer channels
than labels, the list is truncated (with a `show_info` warning), matching the
script's current `CHANNEL_LABELS[:n_ch]` behaviour.

### Buttons (top → bottom)

1. **Load overview** — reads `input_path` via `make_reader`, builds the overview
   at `overview_scale`, adds one image layer per channel (colormap from `_CMAP`,
   `blending="additive"`) plus an empty `slices` shapes layer (rectangle,
   yellow edge, transparent face, `slice` feature + centred text) to the
   **existing** viewer. Wires the live-numbering `events.data` callback and the
   manual click-to-number `mouse_drag_callbacks`. Stores widget state:
   `reader`, `H`, `W`, `fy = H/ov_h`, `fx = W/ov_w`, `labels`, and the
   ordering/counter state dict.
2. **Draw rectangles** — sets the `slices` layer mode to `add_rectangle`.
3. **Number sections** — enters manual click-ordering mode (freezes
   auto-numbering, resets the counter, sets layer mode to `select`); only
   meaningful when `order_by == "manual"`.
4. **Reset numbering** — clears assigned numbers back to 0.
5. **Export crops** — validates numbering, maps each overview box → full-res
   pixel box using `fy`/`fx` with clipping, writes 16-bit TIFFs to
   `<output_root>/<animal_id>/stitched/<label>/<animal_id>_sNNN.tif`, then
   **notifies and clears the section layers** (removes the overview image layers
   and the `slices` layer) so the viewer is clean for the next slide.

### Numbering behaviour (ported faithfully)

- Live relabel on add/remove/move via `shapes.events.data`.
- In `manual` mode, `slice` numbers are held in `shapes.features["slice"]`;
  a plain click (no drag) while the layer is in `select` mode assigns the next
  counter value to the clicked rectangle.
- In `grid`/`draw`/`vertical` modes, numbers are computed by
  `compute_slice_numbers`.
- Export validation for `manual` mode: reject if any section is still `0`
  (unnumbered), or if the assigned numbers are not a clean `1..N` sequence.

The only substantive change vs. the script: the widget acts on the viewer napari
provides, and there is no `napari.run()`, `sys.exit(0)`, tkinter, or viewer
`close()`.

## Manifest

`napari.yaml` gains a second command + widget alongside the existing
`SegLoaderWidget`:

```yaml
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

## Testing

Following segload's `tests/` pattern (`pytest` + `pytest-qt`):

- `test_numbering.py` — pure `compute_slice_numbers` cases: grid rows
  (top→bottom, left→right within a row), draw order, vertical order, and the
  degenerate/empty case.
- `test_readers.py` — `to_16bit` dtype conversions (uint8→uint16 ×257,
  uint16 passthrough, float rescale, flat/empty → zeros); `TiffReader`
  axis detection + `crop_fullres` against a small synthetic multi-channel TIFF
  written with `tifffile`.
- `test_section_widget_smoke.py` — instantiate
  `SectionWidget(make_napari_viewer())`, load a tiny synthetic multi-channel
  TIFF, programmatically add a rectangle to the `slices` layer, assign a number,
  export, and assert the expected files exist on disk with the right dtype/shape.

The CZI path (`aicspylibczi`) is left to manual testing unless a small sample
`.czi` is available; `TiffReader` covers the automated crop/export path.

## Out of scope

- Refactoring or changing the existing `SegLoaderWidget`.
- Changing DMC-BrainMap's expected on-disk layout.
- Pyramidal/tiled write optimisation beyond what `tifffile.imwrite` gives.
