import numpy as np
import pytest

from napari_dmc_segload import _probe_vis_widget as pv
from napari_dmc_segload._probe_vis_widget import ProbeVisWidget


class _FakeMesh:
    def __init__(self, n_verts=12):
        self.points = np.random.default_rng(1).random((n_verts, 3)) * 1000.0
        cell = type("C", (), {"data": np.arange(n_verts).reshape(-1, 3)})()
        self.cells = [cell]


class _FakeSpace:
    axes_description = ("ap", "si", "rl")


class _FakeAtlas:
    resolution = (10.0, 10.0, 10.0)
    space = _FakeSpace()
    structures = {
        "ORBl": {"rgb_triplet": [36, 138, 94]},
        "MOs": {"rgb_triplet": [31, 157, 90]},
        "root": {"rgb_triplet": [255, 255, 255]},
    }

    def mesh_from_structure(self, acronym):
        return _FakeMesh()


@pytest.fixture
def widget(make_napari_viewer, monkeypatch):
    monkeypatch.setattr(pv, "load_atlas", lambda p: _FakeAtlas())
    viewer = make_napari_viewer()
    return ProbeVisWidget(viewer), viewer


def test_visualize_adds_points_and_surfaces(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    names = [ly.name for ly in viewer.layers]
    assert ProbeVisWidget.POINTS_LAYER in names
    assert ProbeVisWidget.ROOT_LAYER in names
    assert any(n.startswith(ProbeVisWidget.REGION_PREFIX) for n in names)
    pts = viewer.layers[ProbeVisWidget.POINTS_LAYER]
    assert len(pts.data) == 800
    np.testing.assert_allclose(pts.scale, [10.0, 10.0, 10.0])
    assert viewer.dims.ndisplay == 3


def test_visualize_fills_panel_and_writes_qc_file(widget, np2_animal_dir):
    w, _ = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    text = w.report.toPlainText()
    assert "shank 0" in text and "ORBl" in text
    out = np2_animal_dir / "results" / "neuropixels_probe" / "animal01_np2_qc.txt"
    assert out.exists()
    assert out.read_text() == text


def test_colour_mode_change_recolours_without_reloading(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    pts = viewer.layers[ProbeVisWidget.POINTS_LAYER]
    before = np.array(pts.face_color, copy=True)
    w.form.colour_by.value = "electrode index"
    after = np.array(viewer.layers[ProbeVisWidget.POINTS_LAYER].face_color)
    assert after.shape == before.shape
    assert not np.allclose(before, after)
    assert len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data) == 800


def test_second_visualize_does_not_duplicate_layers(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    n_first = len(viewer.layers)
    w._visualize()
    assert len(viewer.layers) == n_first


def test_second_visualize_preserves_unrelated_user_layers(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    user_data = np.zeros((3, 3))
    viewer.add_points(user_data, name="my own points")
    w._visualize()
    names = [ly.name for ly in viewer.layers]
    assert names.count("my own points") == 1
    np.testing.assert_array_equal(viewer.layers["my own points"].data, user_data)
    assert names.count(ProbeVisWidget.POINTS_LAYER) == 1
    assert names.count(ProbeVisWidget.ROOT_LAYER) == 1
    region_names = [n for n in names if n.startswith(ProbeVisWidget.REGION_PREFIX)]
    assert len(region_names) == len(set(region_names)) and region_names


def test_missing_electrode_csvs_aborts_cleanly(widget, tmp_path):
    w, viewer = widget
    base = tmp_path / "animal02"
    (base / "results" / "neuropixels_probe" / "neuropixels_probe_0").mkdir(parents=True)
    w.form.input_path.value = str(base)
    w._visualize()                       # must not raise
    assert len(viewer.layers) == 0
    assert w.report.toPlainText() == ""


def test_invalid_path_aborts_cleanly(widget, tmp_path):
    w, viewer = widget
    w.form.input_path.value = str(tmp_path / "does_not_exist")
    w._visualize()                       # must not raise
    assert len(viewer.layers) == 0


def test_scene_build_failure_reports_and_preserves_prior_layers(
        widget, np2_animal_dir, monkeypatch):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()                       # first run succeeds, adds layers
    n_before = len(viewer.layers)
    names_before = [ly.name for ly in viewer.layers]

    def _boom(atlas):
        raise RuntimeError("mesh file missing")

    monkeypatch.setattr(pv, "root_mesh", _boom)
    w._visualize()                       # must not raise
    assert [ly.name for ly in viewer.layers] == names_before
    assert len(viewer.layers) == n_before


def test_scene_build_failure_on_first_run_leaves_no_partial_layers(
        widget, np2_animal_dir, monkeypatch):
    w, viewer = widget

    def _boom(atlas):
        raise RuntimeError("mesh file missing")

    monkeypatch.setattr(pv, "root_mesh", _boom)
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()                       # must not raise
    assert len(viewer.layers) == 0


def test_qc_write_failure_still_adds_scene_and_fills_panel(
        widget, np2_animal_dir, monkeypatch):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)

    from pathlib import Path

    def _boom(self, text):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "write_text", _boom)
    w._visualize()                       # must not raise

    names = [ly.name for ly in viewer.layers]
    assert ProbeVisWidget.POINTS_LAYER in names
    assert ProbeVisWidget.ROOT_LAYER in names
    assert "shank 0" in w.report.toPlainText()
