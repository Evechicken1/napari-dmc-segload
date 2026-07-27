import numpy as np
import pandas as pd
import pytest
from napari_dmc_segload.np2_tract import (
    recover_line, recover_probe_insert, shank_direction, lateral_axis,
    place_electrodes)

COLS = ["ap_coords", "dv_coords", "ml_coords"]


def _synthetic_tract(a, b, insert, n=192):
    dtt = np.arange(n) * 20 + 175
    coords = a + np.outer(dtt, b)
    return pd.DataFrame({
        "distance_to_tip(um)": dtt,
        "depth(um)": insert - dtt,
        "ap_coords": np.round(coords[:, 0]).astype(int),
        "dv_coords": np.round(coords[:, 1]).astype(int),
        "ml_coords": np.round(coords[:, 2]).astype(int),
    })


def test_recover_line_returns_tip_point_and_slope():
    a = np.array([100.0, 200.0, 50.0])
    b = np.array([0.02, -0.10, 0.0])
    df = _synthetic_tract(a, b, insert=4000)
    point, slope = recover_line(df, COLS)
    np.testing.assert_allclose(point, a, atol=1.0)      # intercept = tip voxel (dtt=0)
    np.testing.assert_allclose(slope, b, atol=0.01)


def test_recover_probe_insert_from_depth_plus_dtt():
    df = _synthetic_tract(np.zeros(3), np.zeros(3), insert=4000)
    assert abs(recover_probe_insert(df) - 4000) < 1e-6


def test_recover_line_too_few_rows_raises():
    df = _synthetic_tract(np.zeros(3), np.zeros(3), insert=1000, n=1)
    with pytest.raises(ValueError):
        recover_line(df, COLS)


def test_recover_line_missing_column_raises():
    df = _synthetic_tract(np.zeros(3), np.zeros(3), insert=1000).drop(columns=["ml_coords"])
    with pytest.raises(ValueError):
        recover_line(df, COLS)


def test_lateral_axis_is_unit_and_perpendicular_to_shank():
    shank_dir = np.array([0.0, 1.0, 0.0])           # shank runs along DV
    lat = lateral_axis([0.0, 0.0, 0.0], [0.0, 5.0, 750.0], shank_dir)
    np.testing.assert_allclose(lat, [0.0, 0.0, 1.0], atol=1e-9)  # DV component removed
    assert abs(np.linalg.norm(lat) - 1.0) < 1e-9


def test_place_electrodes_depth_columns_and_offset():
    point = np.array([100.0, 200.0, 50.0])
    slope = np.array([0.0, -0.10, 0.0])
    lat = np.array([0.0, 0.0, 1.0])
    electrodes = [
        {"shank": 0, "electrode": 0, "row": 0, "col": 0, "x_um": 27, "z_um": 0},
        {"shank": 0, "electrode": 1, "row": 0, "col": 1, "x_um": 59, "z_um": 0},
        {"shank": 0, "electrode": 2, "row": 1, "col": 0, "x_um": 27, "z_um": 15},
    ]
    df = place_electrodes(point, slope, 4000, lat, electrodes, COLS)
    # z=0 -> distance_to_tip 175; depth = insert - dtt
    assert df.loc[0, "distance_to_tip_um"] == 175
    assert df.loc[0, "depth_um"] == 3825
    assert df.loc[2, "distance_to_tip_um"] == 190   # +15 um
    # both columns of a row share depth-axis coords (ap, dv); differ on lateral (ml)
    assert df.loc[0, "ap_coords"] == df.loc[1, "ap_coords"]
    assert df.loc[0, "dv_coords"] == df.loc[1, "dv_coords"]
    assert df.loc[1, "ml_coords"] > df.loc[0, "ml_coords"]
    # identity columns preserved
    assert list(df.columns[:6]) == ["shank", "electrode", "row", "col", "x_um", "z_um"]


def test_shank_direction_normalises_and_rejects_zero():
    np.testing.assert_allclose(shank_direction(np.array([0.0, -3.0, 0.0])), [0.0, -1.0, 0.0])
    with pytest.raises(ValueError):
        shank_direction(np.zeros(3))


def test_recover_line_accepts_alternate_distance_column():
    a = np.array([100.0, 200.0, 50.0])
    b = np.array([0.02, -0.10, 0.0])
    df = _synthetic_tract(a, b, insert=4000).rename(
        columns={"distance_to_tip(um)": "distance_to_tip_um"})
    point, slope = recover_line(df, COLS, dist_col="distance_to_tip_um")
    np.testing.assert_allclose(point, a, atol=1.0)
    np.testing.assert_allclose(slope, b, atol=0.01)


def test_recover_line_missing_alternate_distance_column_raises():
    df = _synthetic_tract(np.zeros(3), np.zeros(3), insert=1000)
    with pytest.raises(ValueError, match="distance_to_tip_um"):
        recover_line(df, COLS, dist_col="distance_to_tip_um")
