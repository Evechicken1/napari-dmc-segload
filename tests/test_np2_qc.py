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


from napari_dmc_segload.np2_qc import (
    build_report, cross_shank_metrics, format_report)
from conftest import NP2_SHANK_PITCH_VOX, NP2_TIP0


def test_cross_shank_metrics_recovers_pitch_and_parallelism(np2_frames):
    pairs = cross_shank_metrics(np2_frames, COLS)
    assert len(pairs) == 6                               # 4 shanks -> 6 pairs
    by_key = {(p["shank_a"], p["shank_b"]): p for p in pairs}
    assert by_key[(0, 1)]["distance_um"] == pytest.approx(250.0, abs=1.0)
    assert by_key[(0, 1)]["expected_um"] == pytest.approx(250.0)
    assert by_key[(0, 3)]["distance_um"] == pytest.approx(750.0, abs=1.0)
    assert by_key[(0, 3)]["expected_um"] == pytest.approx(750.0)
    assert all(p["angle_deg"] == pytest.approx(0.0, abs=0.5) for p in pairs)


def test_build_report_clean_input_has_no_warnings(np2_frames):
    report = build_report(np2_frames, COLS)
    assert len(report["shanks"]) == 4
    assert len(report["pairs"]) == 6
    assert report["warnings"] == []


def test_build_report_warns_on_bad_spacing(np2_frames):
    frames = dict(np2_frames)
    frames[1] = make_electrode_frame(
        1, tip=NP2_TIP0 + np.array([0.0, 0.0, 40.0]))     # 400 um, not 250
    warnings = build_report(frames, COLS)["warnings"]
    assert any("spacing" in w for w in warnings)
    assert any("shank 0-1" in w or "shanks 0-1" in w for w in warnings)


def test_build_report_warns_on_tilted_shank(np2_frames):
    tilt = np.deg2rad(15.0)
    frames = dict(np2_frames)
    frames[2] = make_electrode_frame(
        2, tip=NP2_TIP0 + np.array([0.0, 0.0, 2 * NP2_SHANK_PITCH_VOX]),
        slope=0.1 * np.array([np.sin(tilt), -np.cos(tilt), 0.0]))
    warnings = build_report(frames, COLS)["warnings"]
    assert any("angle" in w for w in warnings)


def test_build_report_warns_on_too_few_electrodes(np2_frames):
    frames = dict(np2_frames)
    frames[3] = make_electrode_frame(3, n_rows=10)        # 20 electrodes
    warnings = build_report(frames, COLS)["warnings"]
    assert any("electrodes inside the brain" in w for w in warnings)


def test_build_report_warns_when_all_root(np2_frames):
    frames = dict(np2_frames)
    rooted = make_electrode_frame(0).copy()
    rooted["acronym"] = "root"
    frames[0] = rooted
    warnings = build_report(frames, COLS)["warnings"]
    assert any("root" in w for w in warnings)
    root_warnings = [w for w in warnings if "root" in w]
    assert len(root_warnings) == 1                       # not both all-root and mostly-root


def test_build_report_warns_when_mostly_root(np2_frames):
    frames = dict(np2_frames)
    mostly_rooted = make_electrode_frame(0).copy()
    n = len(mostly_rooted)
    n_root = int(round(n * 0.6))                         # ~60% root, explicit fraction
    acronyms = mostly_rooted["acronym"].to_numpy(dtype=object).copy()
    acronyms[:n_root] = "root"
    mostly_rooted["acronym"] = acronyms
    frames[0] = mostly_rooted
    pct = 100.0 * n_root / n

    warnings = build_report(frames, COLS)["warnings"]
    root_warnings = [w for w in warnings if "root" in w]
    assert len(root_warnings) == 1
    assert "shank 0" in root_warnings[0]
    assert f"{pct:.0f}%" in root_warnings[0]


def test_build_report_no_root_warning_for_clean_input(np2_frames):
    warnings = build_report(np2_frames, COLS)["warnings"]
    assert not any("root" in w for w in warnings)


def test_format_report_is_readable_text(np2_frames):
    text = format_report(build_report(np2_frames, COLS), title="QC — animal01")
    assert "QC — animal01" in text
    assert "shank 0" in text
    assert "ORBl" in text
    assert "no warnings" in text
    assert text.endswith("\n")


def test_format_report_lists_warnings(np2_frames):
    frames = dict(np2_frames)
    frames[3] = make_electrode_frame(3, n_rows=10)
    text = format_report(build_report(frames, COLS))
    assert "warnings" in text
    assert "no warnings" not in text
