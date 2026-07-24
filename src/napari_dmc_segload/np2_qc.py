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
    # Stable sort: the two electrodes of a row share a depth but can sit in
    # different structures, so an unstable sort would reorder them and make the
    # written QC report differ run to run on identical input.
    d = df.sort_values("depth_um", ascending=False, kind="stable")
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


def cross_shank_metrics(frames, coord_cols):
    """Pairwise geometry between shanks: tip distance vs the expected
    250 um x index span, and the angle between fitted directions.

    Returns a list of {shank_a, shank_b, distance_um, expected_um, angle_deg},
    one per unordered pair, in ascending shank order.
    """
    metrics = {s: shank_metrics(df, coord_cols) for s, df in frames.items()}
    shanks = sorted(metrics)
    pairs = []
    for i, a in enumerate(shanks):
        for b in shanks[i + 1:]:
            ma, mb = metrics[a], metrics[b]
            dist = float(np.linalg.norm(mb["fit_tip"] - ma["fit_tip"]) * VOXEL_UM)
            cos = abs(float(np.dot(ma["direction"], mb["direction"])))
            pairs.append({
                "shank_a": a,
                "shank_b": b,
                "distance_um": dist,
                "expected_um": SHANK_PITCH_UM * abs(b - a),
                "angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
            })
    return pairs


def build_report(frames, coord_cols):
    """Full QC report: {'shanks': [...], 'pairs': [...], 'warnings': [...]}."""
    shanks = [shank_metrics(frames[s], coord_cols) for s in sorted(frames)]
    pairs = cross_shank_metrics(frames, coord_cols)
    warnings = []

    for m in shanks:
        if m["n_electrodes"] < MIN_ELECTRODES:
            warnings.append(
                f"shank {m['shank']}: only {m['n_electrodes']} electrodes inside the "
                f"brain (expected >= {MIN_ELECTRODES})")
        if {r["acronym"] for r in m["ladder"]} <= {"root"}:
            warnings.append(
                f"shank {m['shank']}: every electrode is in 'root' — the tract may "
                "lie outside the annotated brain")

    for p in pairs:
        off = abs(p["distance_um"] - p["expected_um"])
        if off > SPACING_TOL_UM:
            warnings.append(
                f"shanks {p['shank_a']}-{p['shank_b']}: tip spacing {p['distance_um']:.0f} um, "
                f"expected {p['expected_um']:.0f} um (off by {off:.0f} um)")
        if p["angle_deg"] > ANGLE_TOL_DEG:
            warnings.append(
                f"shanks {p['shank_a']}-{p['shank_b']}: angle between shanks "
                f"{p['angle_deg']:.1f} deg, expected < {ANGLE_TOL_DEG:.0f} deg")

    return {"shanks": shanks, "pairs": pairs, "warnings": warnings}


def format_report(report, title="NP2.0 probe QC"):
    """Fixed-width text rendering of `build_report`'s output."""
    lines = [title, "=" * max(len(title), 60), ""]
    for m in report["shanks"]:
        lines.append(f"shank {m['shank']}")
        lines.append(f"  electrodes inside brain : {m['n_electrodes']}")
        lines.append(f"  insertion depth (um)    : {m['insertion_depth_um']:.0f}")
        lines.append(f"  tip voxel (a,b,c)       : {m['tip_voxel']}")
        lines.append(f"  angle from DV axis      : {m['angle_from_dv_deg']:.1f} deg")
        lines.append("  region ladder (deep -> superficial):")
        for r in m["ladder"]:
            lines.append(
                f"    {r['acronym']:<10} {r['depth_from_um']:>7.0f} -> "
                f"{r['depth_to_um']:>7.0f} um   n={r['n_electrodes']}")
        lines.append("")

    lines.append("shank pairs")
    for p in report["pairs"]:
        lines.append(
            f"  {p['shank_a']}-{p['shank_b']}  tip distance {p['distance_um']:>6.0f} um "
            f"(expected {p['expected_um']:>6.0f})  angle {p['angle_deg']:>5.1f} deg")
    lines.append("")

    if report["warnings"]:
        lines.append("warnings")
        lines.extend(f"  ! {w}" for w in report["warnings"])
    else:
        lines.append("no warnings")
    lines.append("")
    return "\n".join(lines)
