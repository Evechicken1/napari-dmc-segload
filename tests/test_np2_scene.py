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
    # shank 0 spans electrodes 0..39, shank 1 only 0..19: with joint
    # normalisation shank 1 must not reach the top of the ramp.
    frames = {0: make_electrode_frame(0, n_rows=20), 1: make_electrode_frame(1, n_rows=10)}
    _, features = electrode_points(frames, COLS)
    colors = point_colors(features, "electrode index", None)
    top_overall = colors[features["electrode"].idxmax()]
    shank1 = features.index[features["shank"] == 1]
    assert not np.allclose(colors[shank1].max(axis=0), top_overall)
    assert colors.shape[1] == 4


def test_point_colors_rejects_unknown_mode(np2_frames):
    _, features = electrode_points(np2_frames, COLS)
    with pytest.raises(ValueError, match="colour mode"):
        point_colors(features, "banana", None)
