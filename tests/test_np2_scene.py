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


def test_region_meshes_caps_the_number_of_regions(np2_frames):
    meshes, _ = region_meshes(np2_frames, _FakeAtlas(), min_electrodes=1, max_regions=1)
    assert len(meshes) == 1
