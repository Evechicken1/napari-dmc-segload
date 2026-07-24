import numpy as np
import pytest

from napari_dmc_segload.np2_qc import region_ladder, shank_metrics
from conftest import NP2_COORD_COLS as COLS
from conftest import make_electrode_frame


def test_shank_metrics_basic_geometry(np2_frames):
    m = shank_metrics(np2_frames[0], COLS)
    assert m["shank"] == 0
    assert m["n_electrodes"] == 200                      # 100 rows x 2 columns
    assert m["insertion_depth_um"] == pytest.approx(3825.0)   # 4000 - (0 + 175)
    deepest = np2_frames[0].loc[np2_frames[0]["distance_to_tip_um"].idxmin()]
    assert m["tip_voxel"] == tuple(int(deepest[c]) for c in COLS)
    assert m["angle_from_dv_deg"] == pytest.approx(0.0, abs=0.5)
    np.testing.assert_allclose(np.abs(m["direction"]), [0.0, 1.0, 0.0], atol=1e-2)


def test_shank_metrics_angle_for_tilted_shank():
    tilt = np.deg2rad(15.0)
    slope = 0.1 * np.array([np.sin(tilt), -np.cos(tilt), 0.0])
    m = shank_metrics(make_electrode_frame(0, slope=slope), COLS)
    assert m["angle_from_dv_deg"] == pytest.approx(15.0, abs=1.0)


def test_region_ladder_collapses_contiguous_runs(np2_frames):
    ladder = region_ladder(np2_frames[0])
    assert [r["acronym"] for r in ladder] == ["ORBl", "MOs"]
    assert ladder[0]["depth_from_um"] > ladder[0]["depth_to_um"]      # deep -> superficial
    assert ladder[0]["depth_to_um"] > ladder[1]["depth_from_um"]
    assert sum(r["n_electrodes"] for r in ladder) == len(np2_frames[0])


def test_region_ladder_single_region():
    df = make_electrode_frame(0, n_rows=10)              # all deep -> all ORBl
    ladder = region_ladder(df)
    assert [r["acronym"] for r in ladder] == ["ORBl"]
    assert ladder[0]["n_electrodes"] == 20


def test_region_ladder_empty_frame(np2_frames):
    assert region_ladder(np2_frames[0].iloc[0:0]) == []
