"""Geometric QC for a reconstructed NP2.0 probe.

Pure numpy/pandas over the per-shank electrode DataFrames: no atlas, no napari,
no file IO. Everything is refitted from the electrode CSVs themselves, so QC
validates the files that were actually written rather than the tract they came
from.
"""

import numpy as np

from .np2_tract import recover_line, shank_direction

DIST_COL = "distance_to_tip_um"      # electrode CSV distance column
VOXEL_UM = 10.0                      # atlas voxel size assumed by the CSVs
DV_AXIS = 1                          # index of the 'si' (dorsoventral) axis
SHANK_PITCH_UM = 250.0               # NP2.0 4-shank inter-shank spacing

# Warning thresholds.
SPACING_TOL_UM = 50.0
ANGLE_TOL_DEG = 5.0
MIN_ELECTRODES = 50


def shank_metrics(df, coord_cols):
    """Geometry summary for one shank's electrode DataFrame.

    Keys: shank, n_electrodes, insertion_depth_um (deepest electrode still
    inside the brain), tip_voxel (coords of the smallest distance_to_tip_um),
    fit_tip (fitted intercept at distance_to_tip = 0, float voxels — the anchor
    used for inter-shank distances), direction (unit vector), angle_from_dv_deg,
    ladder.
    """
    point, slope = recover_line(df, coord_cols, dist_col=DIST_COL)
    direction = shank_direction(slope)
    tip_row = df.loc[df[DIST_COL].idxmin()]
    cos_dv = abs(float(direction[DV_AXIS]))
    return {
        "shank": int(df["shank"].iloc[0]),
        "n_electrodes": int(len(df)),
        "insertion_depth_um": float(df["depth_um"].max()),
        "tip_voxel": tuple(int(tip_row[c]) for c in coord_cols),
        "fit_tip": point,
        "direction": direction,
        "angle_from_dv_deg": float(np.degrees(np.arccos(np.clip(cos_dv, -1.0, 1.0)))),
        "ladder": region_ladder(df),
    }


def region_ladder(df):
    """Contiguous runs of acronym from the deepest electrode to the shallowest.

    Returns a list of {acronym, depth_from_um, depth_to_um, n_electrodes}.
    """
    if len(df) == 0:
        return []
    d = df.sort_values("depth_um", ascending=False)
    acronyms = d["acronym"].to_numpy()
    depths = d["depth_um"].to_numpy(dtype=float)
    runs = []
    start = 0
    for i in range(1, len(acronyms) + 1):
        if i == len(acronyms) or acronyms[i] != acronyms[start]:
            runs.append({
                "acronym": str(acronyms[start]),
                "depth_from_um": float(depths[start]),
                "depth_to_um": float(depths[i - 1]),
                "n_electrodes": int(i - start),
            })
            start = i
    return runs
