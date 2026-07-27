# napari-dmc-segload

A standalone [napari](https://napari.org) plugin that makes the **manual
segmentation** step of [DMC-BrainMap](https://github.com/hejDMC/napari-dmc-brainmap)
easier. It scans the channels registered for an animal, lets you load each one as
its own adjustable image layer, and creates empty tracing layers for the channels
that carry the signal you want to segment. Traced layers are saved in DMC-BrainMap's
exact on-disk format, so the rest of the dmc pipeline consumes them unchanged.

It runs **alongside** DMC-BrainMap in the same napari window and does not modify
the dmc package.

## Requirements

> **This plugin reads per-channel images from `single_channel/<chan>/`.**
> Some dmc animals only have `rgb/` or `stitched/` exports. If an animal has no
> `single_channel/` folder, run dmc's preprocessing to export single-channel
> images first — otherwise the scan finds no channels.

It also reads `params.json` (`general.chans_imaged`) to know which channels were
imaged. If `params.json` is missing or invalid, the scan falls back to listing the
`single_channel/` folders alone (and tells you it did so).

## Install

Into the same environment as your napari / DMC-BrainMap install:

```bash
pip install -e .
```

## Usage

1. In napari: `Plugins → DMC Segmentation Loader → Segmentation loader`.
2. Set **input path** to an animal folder (the one containing `single_channel/`,
   `params.json`, etc.).
3. Click **Scan channels** — both channel lists populate with the channels that
   exist on disk *and* are listed in `params.json`.
4. Assign channels to roles:
   - **Reference channels** load as dimmer background images (e.g. `dapi`).
   - **Segmentation channels** load at full opacity *and* get an empty tracing
     layer to draw into.
5. Pick the **segmentation type** (`cells`, `projections`, `injection_site`,
   `optic_fiber`, `neuropixels_probe`), set `point size` / `number of
   fibers/probes` / `image index` as needed, then click **Load into viewer**.
6. Adjust each layer's contrast / colormap / opacity freely, then trace into the
   segmentation layer(s):
   - `cells` / `projections` → a Points layer per segmentation channel
   - `injection_site` → a Shapes layer per segmentation channel
   - `optic_fiber` / `neuropixels_probe` → one Points layer per probe
     (`<seg_type>_0`, `<seg_type>_1`, …)
7. Click **Save segmentation**. Results are written to
   `segmentation/<seg_type>/<chan>/<image>_<seg_type>.csv` in dmc's format.

There is **no automatic "load next image"** — set the image index manually for the
next section. **Clear the image and tracing layers between images** (the dmc way)
so a new load doesn't sit on top of stale layers.

## Layout this plugin expects / writes

```
<animal_id>/
  params.json                       # general.chans_imaged
  single_channel/<chan>/*.tif       # per-channel grayscale images (input)
  segmentation/<seg_type>/<chan>/    # CSV output (Position Y, Position X[, idx_shape])
```

## Development

Tests use pytest + pytest-qt and need a display (napari image layers require a GL
context, so a headless `QT_QPA_PLATFORM=offscreen` run will fail):

```bash
pip install -e ".[testing]"
pytest tests/ -v
```

## NP2.0 probe visualization

After running **NP2.0 electrodes**, open `Plugins → DMC Segmentation Loader →
NP2.0 probe visualization` to check the reconstruction.

1. Set **input path** to the same animal folder.
2. Click **Visualize probe**. The viewer switches to 3D and gains:
   - `np2 root` — the translucent brain outline,
   - `np2 region: <acronym>` — one surface per structure the electrodes pass
     through (the largest 12, ignoring structures with fewer than 10
     electrodes),
   - `np2 electrodes` — every reconstructed electrode from every shank.
3. Use **colour by** to switch between:
   - `shank` — one colour per shank,
   - `region` — the atlas colour of each electrode's structure,
   - `electrode index` — a viridis ramp over the physical electrode index,
     normalised across all shanks. A correctly placed shank shows a smooth
     tip→top gradient; a reversed gradient means the fit ran the wrong way.

The QC panel below the button reports, per shank, the electrode count inside the
brain, insertion depth, tip voxel, angle from the DV axis and the region ladder;
then every shank pair's tip distance against the expected 250 µm × index span
and the angle between them. Anything outside tolerance is listed under
`warnings`. The same text is written to
`results/neuropixels_probe/<animal>_np2_qc.txt`.

This widget only reads the electrode CSVs — it never recomputes the
reconstruction.
