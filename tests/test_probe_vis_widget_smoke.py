import numpy as np
import pytest

from napari_dmc_segload import _probe_vis_widget as pv
from napari_dmc_segload._probe_vis_widget import ProbeVisWidget
from napari_dmc_segload.np2_scene import ROI_OPACITY as ROI_OPACITY_EXPECTED
from napari_dmc_segload.np2_scene import MODE_SHANK, point_colors

MESHLESS_ACRONYM = "VISp"


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
        "ORBl": {"acronym": "ORBl", "rgb_triplet": [36, 138, 94]},
        "MOs": {"acronym": "MOs", "rgb_triplet": [31, 157, 90]},
        "root": {"acronym": "root", "rgb_triplet": [255, 255, 255]},
        "LDT": {"acronym": "LDT", "rgb_triplet": [255, 0, 0]},
        MESHLESS_ACRONYM: {"acronym": MESHLESS_ACRONYM, "rgb_triplet": [128, 128, 128]},
    }

    def mesh_from_structure(self, acronym):
        if acronym == MESHLESS_ACRONYM:
            raise KeyError(acronym)
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


def test_visualize_populates_region_dropdown(widget, np2_animal_dir):
    w, _ = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    choices = list(w.form.add_region.choices)
    assert len(choices) > 0
    assert "LDT" in choices


def test_add_region_adds_highlighted_roi_layer(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.add_region.value = "LDT"
    w._add_region()
    name = ProbeVisWidget.ROI_PREFIX + "LDT"
    assert name in [ly.name for ly in viewer.layers]
    assert viewer.layers[name].opacity == pytest.approx(ROI_OPACITY_EXPECTED)


def test_add_region_before_visualize_aborts_cleanly(widget):
    w, viewer = widget
    w._add_region()                      # must not raise
    assert not any(ly.name.startswith(ProbeVisWidget.ROI_PREFIX) for ly in viewer.layers)


def test_add_region_meshless_reports_and_adds_no_layer(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.add_region.value = MESHLESS_ACRONYM     # defined on the fake atlas
    w._add_region()                                 # must not raise
    assert not any(ly.name.startswith(ProbeVisWidget.ROI_PREFIX) for ly in viewer.layers)


def test_second_visualize_clears_roi_layers(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.add_region.value = "LDT"
    w._add_region()
    w._visualize()
    assert not any(ly.name.startswith(ProbeVisWidget.ROI_PREFIX) for ly in viewer.layers)


def test_depth_filter_hides_deep_electrodes_and_restores(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    full = len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data)
    n_region = sum(ly.name.startswith(ProbeVisWidget.REGION_PREFIX) for ly in viewer.layers)
    qc_before = w.report.toPlainText()
    # enable a mid-range max depth
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = 3000.0
    filtered = len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data)
    assert 0 < filtered < full
    # region meshes and QC untouched by a display filter
    assert sum(ly.name.startswith(ProbeVisWidget.REGION_PREFIX) for ly in viewer.layers) == n_region
    assert w.report.toPlainText() == qc_before
    # disable -> restored
    w.form.limit_depth.value = False
    assert len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data) == full


def test_depth_filter_below_all_leaves_empty_points_no_crash(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = -1.0
    assert len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data) == 0


def test_depth_filter_empty_then_restored_repopulates_colour(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = -1.0                # empty: face_color assignment skipped
    assert len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data) == 0
    w.form.max_depth_um.value = 3000.0               # back to a non-empty subset
    layer = viewer.layers[ProbeVisWidget.POINTS_LAYER]
    assert len(layer.data) > 0
    colors = np.array(layer.face_color)
    assert colors.shape == (len(layer.data), 4)
    np.testing.assert_allclose(
        colors, point_colors(layer.features, w.form.colour_by.value, w._atlas))


def test_visualize_adds_tract_layer_coloured_by_shank(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    assert ProbeVisWidget.TRACT_LAYER in [ly.name for ly in viewer.layers]
    tr = viewer.layers[ProbeVisWidget.TRACT_LAYER]
    assert len(tr.data) == 200                       # 4 shanks x 50 samples
    np.testing.assert_allclose(tr.scale, [10.0, 10.0, 10.0])
    np.testing.assert_allclose(
        np.array(tr.face_color), point_colors(tr.features, MODE_SHANK, None))


def test_depth_filter_leaves_tract_layer_untouched(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    tract_n = len(viewer.layers[ProbeVisWidget.TRACT_LAYER].data)
    elec_full = len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data)
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = 3000.0
    assert len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data) < elec_full   # filtered
    assert len(viewer.layers[ProbeVisWidget.TRACT_LAYER].data) == tract_n     # unchanged


def test_second_visualize_does_not_duplicate_tract_layer(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w._visualize()
    names = [ly.name for ly in viewer.layers]
    assert names.count(ProbeVisWidget.TRACT_LAYER) == 1


def test_visualize_without_tract_csvs_still_builds_scene(widget, np2_animal_dir):
    w, viewer = widget
    root = np2_animal_dir / "results" / "neuropixels_probe"
    for c in root.glob("*/*_neuropixels_probe.csv"):
        c.unlink()
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()                                   # must not raise
    names = [ly.name for ly in viewer.layers]
    assert ProbeVisWidget.POINTS_LAYER in names
    assert ProbeVisWidget.TRACT_LAYER not in names
    assert "no tract overlay" in w.report.toPlainText()


def test_colour_switch_still_works_with_depth_active(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    full = len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data)
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = 3000.0
    pts = viewer.layers[ProbeVisWidget.POINTS_LAYER]
    n_filtered = len(pts.data)
    assert 0 < n_filtered < full          # prove the filter is actually active
    before = np.array(pts.face_color, copy=True)
    w.form.colour_by.value = "electrode index"
    after_layer = viewer.layers[ProbeVisWidget.POINTS_LAYER]
    after = np.array(after_layer.face_color)
    assert len(after_layer.data) == n_filtered
    assert after.shape == before.shape and not np.allclose(before, after)
