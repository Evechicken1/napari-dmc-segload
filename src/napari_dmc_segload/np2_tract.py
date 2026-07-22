"""Recover dmc's per-shank probe line from its tract CSV and place the NP2.0
electrode template along it. Pure numpy/pandas — no atlas, no napari."""

import numpy as np
import pandas as pd

DIST_COL = "distance_to_tip(um)"   # dmc tract CSV column
DEPTH_COL = "depth(um)"            # dmc tract CSV column


def load_tract_csv(path):
    """Read a dmc probe-tract CSV into a DataFrame."""
    return pd.read_csv(path)


def recover_line(tract_df, coord_cols):
    """Least-squares fit of each atlas-coord column against distance_to_tip.

    dmc writes coords = (probe_insert - dtt)/10 * direction + surface, i.e. a
    line in dtt. Returns (point, slope), both length-3 arrays in `coord_cols`
    order, where `point` is the coord at dtt=0 (the tip voxel) and `slope` is
    d(coord)/d(dtt). The many rows average out dmc's integer-voxel rounding.
    """
    if DIST_COL not in tract_df.columns:
        raise ValueError(f"tract CSV missing column '{DIST_COL}'")
    dtt = tract_df[DIST_COL].to_numpy(dtype=float)
    point = np.empty(3)
    slope = np.empty(3)
    for k, c in enumerate(coord_cols):
        if c not in tract_df.columns:
            raise ValueError(f"tract CSV missing column '{c}'")
        s, intercept = np.polyfit(dtt, tract_df[c].to_numpy(dtype=float), 1)
        slope[k] = s
        point[k] = intercept
    return point, slope


def recover_probe_insert(tract_df):
    """probe_insert (um) = median(depth + distance_to_tip) over all rows."""
    for c in (DEPTH_COL, DIST_COL):
        if c not in tract_df.columns:
            raise ValueError(f"tract CSV missing column '{c}'")
    return float(np.median(tract_df[DEPTH_COL].to_numpy(float)
                           + tract_df[DIST_COL].to_numpy(float)))


def shank_direction(slope):
    """Unit vector along a shank, from its line slope."""
    slope = np.asarray(slope, float)
    n = np.linalg.norm(slope)
    if n == 0:
        raise ValueError("degenerate line: zero slope")
    return slope / n


def lateral_axis(tip_min, tip_max, shank_dir):
    """Unit lateral (column) axis: the shank-to-shank vector with its component
    along the shank direction removed, so it is perpendicular to the shank."""
    v = np.asarray(tip_max, float) - np.asarray(tip_min, float)
    shank_dir = np.asarray(shank_dir, float)
    v = v - np.dot(v, shank_dir) * shank_dir
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("cannot determine lateral axis (shank tips coincide)")
    return v / n


def place_electrodes(point, slope, probe_insert, lat_axis, electrodes, coord_cols,
                     tip_offset_um=175.0, shank_centre_um=43.0, voxel_um=10.0):
    """Place template electrodes along the recovered line.

    For each electrode: distance_to_tip = z_um + tip_offset_um;
    coord = point + slope*dtt + ((x_um - shank_centre)/voxel_um) * lat_axis;
    depth = probe_insert - dtt. `electrodes` is a list of dicts with keys
    shank, electrode, row, col, x_um, z_um. Returns a DataFrame with those
    identity columns plus distance_to_tip_um, depth_um and the three int
    `coord_cols`.
    """
    point = np.asarray(point, float)
    slope = np.asarray(slope, float)
    lat_axis = np.asarray(lat_axis, float)
    rows = []
    for e in electrodes:
        dtt = e["z_um"] + tip_offset_um
        col_off = (e["x_um"] - shank_centre_um) / voxel_um
        coord = point + slope * dtt + col_off * lat_axis
        rec = {"shank": e["shank"], "electrode": e["electrode"], "row": e["row"],
               "col": e["col"], "x_um": e["x_um"], "z_um": e["z_um"],
               "distance_to_tip_um": dtt, "depth_um": probe_insert - dtt}
        for k, c in enumerate(coord_cols):
            rec[c] = int(round(coord[k]))
        rows.append(rec)
    return pd.DataFrame(rows)
