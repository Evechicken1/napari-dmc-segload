"""Scene data for the NP2.0 probe visualization.

Pure numpy/pandas: builds the point array, per-point colours and atlas meshes
that the widget hands to napari. Imports no napari, and takes an atlas-like
object (`.structures`, `.mesh_from_structure`) so tests can inject a fake.
"""

import numpy as np
import pandas as pd

MODE_SHANK = "shank"
MODE_REGION = "region"
MODE_INDEX = "electrode index"
COLOUR_MODES = (MODE_SHANK, MODE_REGION, MODE_INDEX)

# Four visually distinct shank colours (RGBA 0-1).
SHANK_COLORS = np.array([
    [0.90, 0.29, 0.23, 1.0],   # red
    [0.20, 0.60, 0.86, 1.0],   # blue
    [0.95, 0.77, 0.06, 1.0],   # amber
    [0.18, 0.80, 0.44, 1.0],   # green
])
UNKNOWN_REGION_COLOR = np.array([0.5, 0.5, 0.5, 1.0])

# Viridis anchor stops, interpolated per channel — keeps this module dependency-free.
_INDEX_CMAP_STOPS = np.array([
    [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
])


def electrode_points(frames, coord_cols):
    """Stack every shank's electrodes into one point array + feature table.

    Shanks are concatenated in ascending index order, rows in file order.
    Returns (points (N,3) float voxel coords, features DataFrame with columns
    shank, acronym, electrode).
    """
    points, feats = [], []
    for shank in sorted(frames):
        df = frames[shank]
        points.append(df[coord_cols].to_numpy(dtype=float))
        feats.append(pd.DataFrame({
            "shank": df["shank"].to_numpy(dtype=int),
            "acronym": df["acronym"].astype(str).to_numpy(),
            "electrode": df["electrode"].to_numpy(dtype=int),
        }))
    return (np.concatenate(points, axis=0),
            pd.concat(feats, ignore_index=True))


def point_colors(features, mode, atlas):
    """RGBA (N,4) for each electrode under the given colour mode.

    `atlas` is only used by MODE_REGION and may be None otherwise.
    """
    if mode == MODE_SHANK:
        idx = features["shank"].to_numpy(dtype=int) % len(SHANK_COLORS)
        return SHANK_COLORS[idx]

    if mode == MODE_REGION:
        structures = atlas.structures
        out = np.tile(UNKNOWN_REGION_COLOR, (len(features), 1))
        for acronym in features["acronym"].unique():
            try:
                rgb = structures[acronym]["rgb_triplet"]
            except (KeyError, TypeError):
                continue
            mask = (features["acronym"] == acronym).to_numpy()
            out[mask, :3] = np.asarray(rgb, dtype=float) / 255.0
        return out

    if mode == MODE_INDEX:
        e = features["electrode"].to_numpy(dtype=float)
        lo, hi = float(e.min()), float(e.max())
        t = np.zeros_like(e) if hi <= lo else (e - lo) / (hi - lo)
        return _index_ramp(t)

    raise ValueError(f"unknown colour mode '{mode}'; expected one of {COLOUR_MODES}")


def _index_ramp(t):
    """Map t in [0,1] to (N,4) RGBA along the viridis anchor stops."""
    x = np.linspace(0.0, 1.0, len(_INDEX_CMAP_STOPS))
    rgb = np.stack([np.interp(t, x, _INDEX_CMAP_STOPS[:, k]) for k in range(3)], axis=1)
    return np.concatenate([rgb, np.ones((len(t), 1))], axis=1)
