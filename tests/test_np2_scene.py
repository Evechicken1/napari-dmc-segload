import numpy as np
import pytest

from napari_dmc_segload.np2_scene import electrode_points, point_colors
from conftest import NP2_COORD_COLS as COLS
from conftest import make_electrode_frame


class _FakeMesh:
    def __init__(self, n_verts=12):
        self.points = np.random.default_rng(0).random((n_verts, 3)) * 1000.0
        cell = type("C", (), {"data": np.arange(n_verts).reshape(-1, 3)})()
        self.cells = [cell]


class _FakeAtlas:
    """Atlas-like: structures with rgb_triplet, meshes for some acronyms only."""

    resolution = (10.0, 10.0, 10.0)
    structures = {
        "ORBl": {"rgb_triplet": [36, 138, 94]},
        "MOs": {"rgb_triplet": [31, 157, 90]},
        "root": {"rgb_triplet": [255, 255, 255]},
        "ORBm": {"rgb_triplet": [200, 100, 50]},
        "PL": {"rgb_triplet": [10, 20, 30]},
    }
    meshless = {"MOs"}

    def mesh_from_structure(self, acronym):
        if acronym in self.meshless:
            raise KeyError(acronym)
        return _FakeMesh()


def test_electrode_points_stacks_shanks_in_ascending_order(np2_frames):
    points, features = electrode_points(np2_frames, COLS)
    assert points.shape == (800, 3)                       # 4 shanks x 200
    assert list(features.columns) == ["shank", "acronym", "electrode"]
    assert features["shank"].tolist()[:200] == [0] * 200
    assert features["shank"].tolist()[-200:] == [3] * 200
    # first row equals the first frame's first coord triple
    np.testing.assert_allclose(
        points[0], np2_frames[0].iloc[0][COLS].to_numpy(dtype=float))


def test_point_colors_by_shank_gives_one_colour_per_shank(np2_frames):
    _, features = electrode_points(np2_frames, COLS)
    colors = point_colors(features, "shank", None)
    assert colors.shape == (800, 4)
    assert len({tuple(c) for c in colors}) == 4


def test_point_colors_by_region_uses_atlas_rgb(np2_frames):
    _, features = electrode_points(np2_frames, COLS)
    colors = point_colors(features, "region", _FakeAtlas())
    orbl = features.index[features["acronym"] == "ORBl"][0]
    np.testing.assert_allclose(colors[orbl], [36 / 255, 138 / 255, 94 / 255, 1.0],
                               atol=1e-6)


def test_point_colors_by_region_unknown_acronym_is_grey(np2_frames):
    _, features = electrode_points(np2_frames, COLS)
    features.loc[0, "acronym"] = "NOT_A_REGION"
    colors = point_colors(features, "region", _FakeAtlas())
    np.testing.assert_allclose(colors[0], [0.5, 0.5, 0.5, 1.0])


def test_point_colors_by_electrode_index_normalises_across_shanks():
    # shank 0 spans electrodes 0..39, shank 1 only 0..19. Under joint
    # normalisation the same electrode index maps to the same colour in
    # every shank, so shank 1's electrode 19 must match shank 0's electrode
    # 19 -- and must NOT match shank 0's electrode 39 (the top of the ramp,
    # which only joint normalisation withholds from shank 1).
    frames = {0: make_electrode_frame(0, n_rows=20), 1: make_electrode_frame(1, n_rows=10)}
    _, features = electrode_points(frames, COLS)
    colors = point_colors(features, "electrode index", None)

    def row(shank, electrode):
        idx = features.index[(features["shank"] == shank) &
                             (features["electrode"] == electrode)][0]
        return colors[idx]

    shank1_top = row(1, 19)
    shank0_matched = row(0, 19)
    shank0_top = row(0, 39)

    np.testing.assert_allclose(shank1_top, shank0_matched)
    assert not np.allclose(shank1_top, shank0_top)
    assert colors.shape[1] == 4


def test_point_colors_rejects_unknown_mode(np2_frames):
    _, features = electrode_points(np2_frames, COLS)
    with pytest.raises(ValueError, match="colour mode"):
        point_colors(features, "banana", None)


from napari_dmc_segload.np2_scene import region_meshes, root_mesh


def test_root_mesh_returns_vertices_and_faces():
    verts, faces = root_mesh(_FakeAtlas())
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3


def test_region_meshes_skips_structures_without_a_mesh(np2_frames):
    meshes, skipped = region_meshes(np2_frames, _FakeAtlas())
    assert [m["acronym"] for m in meshes] == ["ORBl"]     # MOs is meshless in the fake
    assert skipped == ["MOs"]
    assert meshes[0]["vertices"].shape[1] == 3
    assert meshes[0]["faces"].shape[1] == 3
    np.testing.assert_allclose(meshes[0]["color"][:3],
                               [36 / 255, 138 / 255, 94 / 255], atol=1e-6)


def test_region_meshes_ignores_root_and_rare_structures():
    df = make_electrode_frame(0).copy()
    df.loc[df.index[:4], "acronym"] = "ORBm"             # 4 electrodes only
    df.loc[df.index[4:], "acronym"] = "root"
    meshes, skipped = region_meshes({0: df}, _FakeAtlas(), min_electrodes=10)
    assert meshes == []                                  # ORBm too rare, root excluded
    assert skipped == []


def test_region_meshes_caps_the_number_of_regions():
    # Three meshed structures with distinct electrode counts, all above the
    # default min_electrodes, so ranking is observable and the cap is the
    # only thing that can be reducing the result.
    df = make_electrode_frame(0, n_rows=60).copy()      # 120 electrodes
    acronyms = np.empty(len(df), dtype=object)
    acronyms[:60] = "ORBl"      # most electrodes (60)
    acronyms[60:100] = "ORBm"   # fewer (40)
    acronyms[100:] = "PL"       # fewest (20)
    df["acronym"] = acronyms

    meshes, skipped = region_meshes({0: df}, _FakeAtlas(), min_electrodes=1, max_regions=2)
    assert skipped == []
    assert [m["acronym"] for m in meshes] == ["ORBl", "ORBm"]     # cap size AND ranking


from napari_dmc_segload.np2_scene import atlas_acronyms, roi_mesh, ROI_OPACITY
from conftest import make_electrode_frame
from conftest import NP2_COORD_COLS as COLS


class _RoiAtlas:
    resolution = (10.0, 10.0, 10.0)
    structures = {
        "ORBl": {"acronym": "ORBl", "rgb_triplet": [36, 138, 94]},
        "MOs": {"acronym": "MOs", "rgb_triplet": [31, 157, 90]},
        "LDT": {"acronym": "LDT", "rgb_triplet": [255, 0, 0]},
        "root": {"acronym": "root", "rgb_triplet": [255, 255, 255]},
    }
    meshless = {"MOs"}

    def mesh_from_structure(self, acronym):
        if acronym not in self.structures or acronym in self.meshless:
            raise KeyError(acronym)
        pts = __import__("numpy").arange(12, dtype=float).reshape(4, 3)
        cell = type("C", (), {"data": __import__("numpy").arange(12).reshape(-1, 3)})()
        return type("M", (), {"points": pts, "cells": [cell]})()


def test_atlas_acronyms_sorted_and_complete():
    acs = atlas_acronyms(_RoiAtlas())
    assert acs == sorted(acs)
    assert set(acs) == {"ORBl", "MOs", "LDT", "root"}


def test_roi_mesh_returns_mesh_dict_for_meshed_structure():
    m = roi_mesh(_RoiAtlas(), "LDT")
    assert m["acronym"] == "LDT"
    assert m["vertices"].shape[1] == 3 and m["faces"].shape[1] == 3
    import numpy as np
    np.testing.assert_allclose(m["color"][:3], [1.0, 0.0, 0.0])


def test_roi_mesh_missing_mesh_raises_valueerror():
    import pytest
    with pytest.raises(ValueError, match="MOs"):
        roi_mesh(_RoiAtlas(), "MOs")
    with pytest.raises(ValueError, match="NOPE"):
        roi_mesh(_RoiAtlas(), "NOPE")


def test_electrode_points_max_depth_filters_inclusive():
    frames = {0: make_electrode_frame(0, n_rows=100)}          # depth_um 175..? decreasing with z
    full, _ = electrode_points(frames, COLS)
    dmax = float(frames[0]["depth_um"].median())
    pts, feats = electrode_points(frames, COLS, max_depth=dmax)
    kept = (frames[0]["depth_um"] <= dmax).sum()
    assert len(pts) == kept == len(feats)
    assert len(pts) < len(full)


def test_electrode_points_max_depth_none_is_unchanged():
    frames = {0: make_electrode_frame(0)}
    a, _ = electrode_points(frames, COLS)
    b, _ = electrode_points(frames, COLS, max_depth=None)
    import numpy as np
    np.testing.assert_array_equal(a, b)


def test_electrode_points_max_depth_below_all_gives_empty():
    frames = {0: make_electrode_frame(0)}
    pts, feats = electrode_points(frames, COLS, max_depth=-1e9)
    assert pts.shape == (0, 3) and len(feats) == 0


def test_point_colors_empty_features_all_modes():
    import numpy as np
    _, feats = electrode_points({0: make_electrode_frame(0)}, COLS, max_depth=-1e9)
    for mode in ("shank", "region", "electrode index"):
        c = point_colors(feats, mode, _RoiAtlas())
        assert c.shape == (0, 4)
