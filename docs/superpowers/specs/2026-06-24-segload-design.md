# napari-dmc-segload — channel-aware segmentation loader

**Date:** 2026-06-24
**Status:** Design approved, pending spec review

## Problem

DMC-BrainMap's Segmentation widget loads imagery for manual segmentation in a way
that limits independent per-channel control:

- **RGB mode** reads one composite tif and splits B/G/R into `dapi`/`green`/`cy3`
  layers with fixed opacities. Channels `n3` and `cy5` cannot be loaded at all.
- **Single-channel mode** can load each selected channel, but the widget couples
  loading, layer creation, navigation, and saving into one "save data and load
  next image" flow, with little room to tune the reference imagery used while
  tracing.

The goal is a focused helper that makes the *loading* step for manual
segmentation easier: detect which channels actually exist for an animal, let the
user choose which to load and what role each plays, and drop each channel into
the viewer as its own image layer whose contrast/colormap/opacity can be tuned
freely — plus an empty tracing layer for the channels that carry the signal to
be segmented.

## Goals

- Detect the channels registered/available for an animal.
- Let the user assign each available channel a **reference** or **segmentation**
  role.
- Load each chosen channel as its own grayscale image layer with adjustable
  contrast, colormap, and opacity.
- For each segmentation channel, also create the empty tracing layer
  (points/shapes) appropriate to the segmentation type.
- Save traced layers to DMC-BrainMap's on-disk segmentation convention so the
  rest of the dmc pipeline (Results) consumes them unchanged.

## Non-goals

- No automatic image-to-image advance ("load next"); the image index is set
  manually.
- No presegmentation/centroid/mask computation — those remain in dmc.
- No editing of any installed dmc package file; no dependency on dmc runtime
  internals (`save_dict`, magicgui field names).

## Packaging

Standalone, pip-installable napari plugin in its own repo at
`Python/napari-dmc-segload/`, installed editable (`pip install -e .`) into the
`napari-env` conda environment. It contributes a single dock widget,
**Segmentation loader**, that loads into the active napari viewer alongside dmc.
This survives dmc upgrades and touches no dmc files.

Layout:

```
napari-dmc-segload/
  pyproject.toml
  src/napari_dmc_segload/
    __init__.py
    napari.yaml          # contributes one widget: Segmentation loader
    _widget.py           # SegLoaderWidget (Qt dock widget)
    channel_scanner.py   # ChannelScanner: available-channel detection
    io.py                # index->file resolution, image load, save
  tests/
    test_channel_scanner.py
    test_io.py
    test_widget_smoke.py
```

## On-disk conventions (dmc, treated as the stable contract)

- Per-channel images: `single_channel/<chan>/*.tif` (grayscale). Channels are a
  subset of `dapi, green, n3, cy3, cy5`.
- Registered channels: `params.json → general.chans_imaged`.
- Image identity: files are matched by index into the **natsorted** tif list of
  a channel folder (mirrors dmc's `get_path_to_im`). The image base name (suffix
  stripped) is used to name saved CSVs.
- Saved segmentation: `segmentation/<seg_type>/<chan>/<image_base>_<seg_type>.csv`
  with columns `Position Y`, `Position X`. For `injection_site`, polygons are
  written with an added `idx_shape` column, one block per shape. For
  `optic_fiber`/`neuropixels_probe`, channel id is `f"{seg_type}_{i}"` for
  `i in range(n_probes)`.

## Components

### 1. ChannelScanner (`channel_scanner.py`)

Pure logic, no Qt.

```
ChannelScanner(input_path: Path)
  .available_channels() -> list[str]
```

- Lists subfolders of `single_channel/` that exist and contain ≥1 `.tif`.
- Intersects with `params.json → general.chans_imaged` when `params.json` is
  present; if `params.json` is missing, falls back to the folder scan alone and
  emits a `show_info` note.
- Returns channels ordered by the canonical channel order
  (`dapi, green, n3, cy3, cy5`).

### 2. IO helpers (`io.py`)

Pure logic, no Qt.

- `resolve_image(input_path, chan, image_idx) -> Path` — natsorted tif at index,
  raises `IndexError` if out of range.
- `image_base_name(input_path, chan, image_idx) -> str` — file name with the
  channel folder's common suffix stripped (mirrors dmc's `get_image_list`
  suffix logic).
- `load_grayscale(path) -> np.ndarray` — `cv2.imread(..., IMREAD_GRAYSCALE)`.
- `save_points(layer_data, input_path, seg_type, chan, image_base)` and
  `save_shapes(...)` — write the dmc-format CSV; empty data writes nothing and
  removes a stale file of the same name if present.

### 3. SegLoaderWidget (`_widget.py`)

A `QWidget` dock widget. Uses the magicgui idiom where convenient so it matches
dmc's look. Controls:

- `input_path` (FileEdit, directory) + **Scan channels** button.
- **Reference channels** (multi-select) and **Segmentation channels**
  (multi-select), both populated by the scan. A channel may appear in only one
  list; assigning it to both is rejected with a notification.
- `seg_type` (combo: `cells, injection_site, optic_fiber, neuropixels_probe,
  projections`), `n_probes` (int), `point_size` (int), `image_idx` (int).
- Per-channel display defaults sourced from dmc's display colormap
  (`dapi:blue, green:green, n3:orange, cy3:red, cy5:pink`); reference channels
  default to lower opacity, segmentation channels to full opacity.
- **Load into viewer** button.
- **Save segmentation** button.

### 4. Load flow

For the selected `image_idx`:

1. For each selected channel (reference and segmentation):
   `add_image(load_grayscale(resolve_image(...)), name=f"{chan} channel",
   colormap=<display>, opacity=<role default>)`, then set `contrast_limits`.
2. For each **segmentation** channel, add the empty tracing layer matching
   `seg_type`:
   - `cells`, `projections` → `add_points(name=chan, size=point_size, ...)`
   - `injection_site` → `add_shapes(name=chan, ...)`
   - `optic_fiber`, `neuropixels_probe` → `n_probes` point layers named
     `f"{seg_type}_{i}"`.

Layer names match dmc so the on-disk save and any later dmc interaction stay
consistent.

### 5. Save flow

On **Save segmentation**, for the current `seg_type`/channels/`image_idx`:

- Resolve `image_base` from the first selected channel's folder.
- For each segmentation channel (or `f"{seg_type}_{i}"` for probes/fibers),
  read `viewer.layers[<name>].data` and write the dmc-format CSV under
  `segmentation/<seg_type>/<chan>/`. Injection-site shapes get the `idx_shape`
  column. Empty layers write nothing / remove a stale file.
- Image index is **not** advanced; the user sets it manually for the next image.

## Error handling

Each surfaces a `show_info` notification and aborts cleanly (no exception
escapes to the user):

- Missing `single_channel/` directory or empty channel folder.
- Missing `params.json` (scan falls back to folder listing).
- `image_idx` out of range.
- No channels selected, or a channel assigned to both roles.
- dmc-format save attempted with no segmentation channels chosen.

## Testing

- `test_channel_scanner.py` — fixture tree with a couple of
  `single_channel/<chan>/*.tif` folders and a `params.json`; assert available
  channels, ordering, and the params-missing fallback.
- `test_io.py` — index→file resolution (including `IndexError` out of range),
  suffix-stripped base name, and round-trip save producing the exact dmc CSV
  format (points and injection-site shapes).
- `test_widget_smoke.py` — using napari's `make_napari_viewer` fixture, scan +
  load against the fixture tree and assert the expected image layer names and
  the correct tracing-layer types per `seg_type`.

## Risks

- Depends on dmc's **on-disk** layout (`single_channel/`, `params.json`,
  `segmentation/<seg_type>/<chan>/` CSV format). This is dmc's stable public
  contract, not runtime internals, so the coupling is low-risk and visible.
- `cv2` and `natsort` are already dependencies in the napari-env (used by dmc),
  so no new heavy dependencies are introduced.
