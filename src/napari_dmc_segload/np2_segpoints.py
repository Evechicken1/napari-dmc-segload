"""Raw manually-clicked NP2.0 segmentation points, registered into the atlas.

The clicks live as 2-D image coordinates in
`segmentation/neuropixels_probe/neuropixels_probe_<i>/<animal>_s*_single_neuropixels_probe.csv`
(one file per section). To place them in the 3-D atlas scene they are rescaled
to the registration-downsampled image and pushed through dmc-brainmap's
landmark warp + angled-slice placement.

We reuse dmc's registration *primitives* (imported lazily) but not its
`SliceHandle`, whose constructor loads the full atlas annotation (~5 GB) that we
do not need — we want coordinates, not per-click region names. The lightweight
path here was verified to produce coordinates identical to
`SliceHandle.getBrainArea` on real data. See the design spec.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted

from .np2_results import shank_index

SEG_SUBDIR = ("segmentation", "neuropixels_probe")
SEG_CSV_SUFFIX = "_single_neuropixels_probe.csv"
REGI_SUFFIX = "_downsampled.tif"
NONE_CHANNEL = "(none)"


def registration_channels(input_path):
    """sharpy_track channels that carry a registration.json (drives the dropdown)."""
    st = Path(input_path) / "sharpy_track"
    if not st.is_dir():
        return []
    return natsorted(d.name for d in st.iterdir()
                     if d.is_dir() and (d / "registration.json").is_file())


def shank_click_csvs(input_path):
    """{shank_index: [click CSV paths]} under segmentation/neuropixels_probe.

    Keyed by the folder index (same rule as the results loaders); folders with
    no matching CSV are skipped. Empty dict if the segmentation tree is absent.
    """
    base = Path(input_path).joinpath(*SEG_SUBDIR)
    out = {}
    if not base.is_dir():
        return out
    for sd in natsorted((d for d in base.iterdir() if d.is_dir()), key=lambda p: p.name):
        csvs = natsorted(p for p in sd.glob(f"*{SEG_CSV_SUFFIX}") if p.is_file())
        if csvs:
            out[shank_index(sd.name)] = list(csvs)
    return out


def rescale_clicks(seg_df, full_hw, low_hw):
    """Rescale full-res (Position Y, Position X) clicks to registration-space
    integer (x, y).

    `full_hw` / `low_hw` are the *max index* extents (height-1, width-1) of the
    full-res segmentation image and the registration-downsampled image. Mirrors
    dmc: min-max scale with the full-res extent appended, then dropped.
    """
    from sklearn.preprocessing import minmax_scale
    full_h, full_w = full_hw
    low_h, low_w = low_hw
    y_pos = list(seg_df["Position Y"]) + [0, full_h]
    x_pos = list(seg_df["Position X"]) + [0, full_w]
    y = np.ceil(minmax_scale(y_pos, feature_range=(0, low_h)))[:-2].astype(int)
    x = np.ceil(minmax_scale(x_pos, feature_range=(0, low_w)))[:-2].astype(int)
    return np.stack([x, y], axis=1)


def stack_shank_points(by_shank):
    """(points (N,3) float, features['shank']) from {shank: (n,3) array}, shanks
    ascending — so the result colours through `point_colors(.., MODE_SHANK, ..)`."""
    pts, sh = [], []
    for shank in sorted(by_shank):
        arr = np.asarray(by_shank[shank], dtype=float).reshape(-1, 3)
        pts.append(arr)
        sh.append(np.full(len(arr), shank, dtype=int))
    points = np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))
    features = pd.DataFrame(
        {"shank": np.concatenate(sh) if sh else np.zeros(0, dtype=int)})
    return points, features


def _seg_image_maxhw(input_path, regi_chan, curr):
    """(H-1, W-1) of the section's full-res segmentation image, or None.

    All channels of a section share the image dimensions, so the registration
    channel's copy gives the correct rescale ratio. Tries single_channel, then
    stitched, then rgb."""
    import cv2
    input_path = Path(input_path)
    for candidate in (input_path / "single_channel" / regi_chan / f"{curr}_single.tif",
                      input_path / "stitched" / regi_chan / f"{curr}.tif",
                      input_path / "rgb" / f"{curr}.tif"):
        if candidate.is_file():
            img = cv2.imread(str(candidate))
            if img is not None:
                return (img.shape[0] - 1, img.shape[1] - 1)
    return None


def register_probe_clicks(input_path, regi_chan, atlas, *, warp=None):
    """Register every shank's raw segmentation clicks into atlas voxels.

    Returns {shank_index: (N,3) float ap/dv/ml voxel array}, only for shanks
    with at least one in-volume click. `warp` is an injectable callable
    `(coords_xy, section_key) -> list[[a,b,c]]`; the default is built from the
    registration JSON + dmc-brainmap primitives (see `_build_warp`) — injecting
    it keeps this orchestration testable without dmc or an atlas.
    """
    import cv2
    input_path = Path(input_path)
    by_shank_csvs = shank_click_csvs(input_path)
    if not by_shank_csvs:
        return {}

    regi_dir = input_path / "sharpy_track" / regi_chan
    reg_data = json.loads((regi_dir / "registration.json").read_text())
    name_to_key = {v: k for k, v in reg_data["imgName"].items()}
    if warp is None:
        warp = _build_warp(reg_data, input_path, regi_chan, atlas)

    out = {}
    for shank, csvs in by_shank_csvs.items():
        pts = []
        for csv in csvs:
            curr = csv.name[:-len(SEG_CSV_SUFFIX)]
            key = name_to_key.get(curr + REGI_SUFFIX)
            if key is None:
                continue
            low = cv2.imread(str(regi_dir / (curr + REGI_SUFFIX)))
            full_hw = _seg_image_maxhw(input_path, regi_chan, curr)
            if low is None or full_hw is None:
                continue
            coords = rescale_clicks(pd.read_csv(csv), full_hw,
                                    (low.shape[0] - 1, low.shape[1] - 1))
            pts.extend(warp(coords, key))
        if pts:
            out[shank] = np.asarray(pts, dtype=float)
    return out


def _build_warp(reg_data, input_path, regi_chan, atlas):
    """Default warp closure using dmc-brainmap's registration primitives.

    Returns `(coords_xy, section_key) -> list[[a,b,c]]` in atlas axis order.
    Reproduces `SliceHandle.getVolumeIndex` without loading the annotation.
    dmc-brainmap is imported here so the plugin still imports without it.
    """
    from napari_dmc_brainmap.utils.general_utils import create_regi_dict
    from napari_dmc_brainmap.utils.atlas_utils import (
        get_bregma, coord_mm_transform, xyz_atlas_transform)
    from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import (
        fitGeoTrans, mapPointTransform)

    regi_dir = Path(input_path) / "sharpy_track" / regi_chan
    regi_dict = create_regi_dict(Path(input_path), regi_dir)
    axes = atlas.space.axes_description
    z_idx = axes.index(regi_dict["xyz_dict"]["z"][0])
    bregma = get_bregma(regi_dict["atlas"])
    x_max = regi_dict["xyz_dict"]["x"][1]
    y_max = regi_dict["xyz_dict"]["y"][1]
    gy, gx = np.meshgrid(np.arange(y_max), np.arange(x_max), indexing="ij")
    grid = np.stack([gy, gx], axis=2)

    # landmark transforms per section, only where dmc considers the pairing valid
    tforms = {k: fitGeoTrans(reg_data["sampleDots"][k], dots)
              for k, dots in reg_data["atlasDots"].items() if len(dots) >= 5}

    def z_plane(k):
        x_angle, y_angle, z_mm = reg_data["atlasLocation"][k]
        z_coord = coord_mm_transform([z_mm], [bregma[z_idx]],
                                     [atlas.space.resolution[z_idx]], mm_to_coord=True)
        if x_angle == 0 and y_angle == 0:
            return np.full((y_max, x_max), z_coord, dtype=np.uint16)
        x_shift = int(np.tan(np.deg2rad(x_angle)) * x_max / 2)
        y_shift = int(np.tan(np.deg2rad(y_angle)) * y_max / 2)
        center = np.array([z_coord, y_max / 2, x_max / 2])
        vec_1 = np.array([z_coord + x_shift, y_max / 2, x_max - 1]) - center
        vec_2 = np.array([z_coord - y_shift, 0, x_max / 2]) - center
        vec_n = np.cross(vec_1, vec_2)
        return (-vec_n[1] * (grid[:, :, 0] - center[1])
                - vec_n[2] * (grid[:, :, 1] - center[2])) / vec_n[0] + center[0]

    def warp(coords_xy, section_key):
        if section_key not in tforms:
            return []
        tform = tforms[section_key]
        zp = z_plane(section_key)
        out = []
        for x_pre, y_pre in coords_xy:
            x_post, y_post = mapPointTransform(x_pre, y_pre, tform)
            x, y = int(x_post), int(y_post)
            try:
                z = int(zp[y, x])
            except IndexError:
                continue
            if not (x < 0 or y < 0 or z < 0):
                out.append(xyz_atlas_transform([x, y, z], regi_dict, axes))
        return out

    return warp
