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


def _fake_clicks(*_):
    return {0: np.array([[900, 300, 560], [901, 310, 561]], dtype=float),
            1: np.array([[905, 320, 540]], dtype=float)}


def test_visualize_populates_seg_dropdown_default_none(widget, np2_animal_dir, monkeypatch):
    w, viewer = widget
    monkeypatch.setattr(pv, "registration_channels", lambda p: ["dapi", "green"])
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    choices = list(w.form.seg_regi_channel.choices)
    assert choices[0] == pv.NONE_CHANNEL and "dapi" in choices and "green" in choices
    assert w.form.seg_regi_channel.value == pv.NONE_CHANNEL
    assert ProbeVisWidget.SEGPOINTS_LAYER not in [ly.name for ly in viewer.layers]


def test_select_channel_adds_segpoints_coloured_by_shank(widget, np2_animal_dir, monkeypatch):
    w, viewer = widget
    monkeypatch.setattr(pv, "registration_channels", lambda p: ["dapi"])
    monkeypatch.setattr(pv, "register_probe_clicks", _fake_clicks)
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.seg_regi_channel.value = "dapi"
    assert ProbeVisWidget.SEGPOINTS_LAYER in [ly.name for ly in viewer.layers]
    sp = viewer.layers[ProbeVisWidget.SEGPOINTS_LAYER]
    assert len(sp.data) == 3
    np.testing.assert_allclose(sp.scale, [10.0, 10.0, 10.0])
    np.testing.assert_allclose(np.array(sp.face_color),
                               point_colors(sp.features, MODE_SHANK, None))


def test_deselect_channel_removes_segpoints(widget, np2_animal_dir, monkeypatch):
    w, viewer = widget
    monkeypatch.setattr(pv, "registration_channels", lambda p: ["dapi"])
    monkeypatch.setattr(pv, "register_probe_clicks", _fake_clicks)
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.seg_regi_channel.value = "dapi"
    assert ProbeVisWidget.SEGPOINTS_LAYER in [ly.name for ly in viewer.layers]
    w.form.seg_regi_channel.value = pv.NONE_CHANNEL
    assert ProbeVisWidget.SEGPOINTS_LAYER not in [ly.name for ly in viewer.layers]


def test_depth_filter_leaves_segpoints_untouched(widget, np2_animal_dir, monkeypatch):
    w, viewer = widget
    monkeypatch.setattr(pv, "registration_channels", lambda p: ["dapi"])
    monkeypatch.setattr(pv, "register_probe_clicks", _fake_clicks)
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.seg_regi_channel.value = "dapi"
    n = len(viewer.layers[ProbeVisWidget.SEGPOINTS_LAYER].data)
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = 3000.0
    assert len(viewer.layers[ProbeVisWidget.SEGPOINTS_LAYER].data) == n


def test_register_failure_reports_and_keeps_scene(widget, np2_animal_dir, monkeypatch):
    w, viewer = widget
    monkeypatch.setattr(pv, "registration_channels", lambda p: ["dapi"])

    def _boom(*_):
        raise RuntimeError("dmc not installed")

    monkeypatch.setattr(pv, "register_probe_clicks", _boom)
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.seg_regi_channel.value = "dapi"                 # must not raise
    assert ProbeVisWidget.SEGPOINTS_LAYER not in [ly.name for ly in viewer.layers]
    assert ProbeVisWidget.POINTS_LAYER in [ly.name for ly in viewer.layers]


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


def _write_staggered_animal(base, inserts):
    import json
    from napari_dmc_segload.np2_results import ELECTRODE_CSV_SUFFIX
    from conftest import make_electrode_frame
    root = base / "results" / "neuropixels_probe"
    for shank, ins in enumerate(inserts):
        sd = root / f"neuropixels_probe_{shank}"
        sd.mkdir(parents=True)
        make_electrode_frame(shank, insert=ins).to_csv(
            sd / f"{base.name}{ELECTRODE_CSV_SUFFIX}", index=False)
    (base / "params.json").write_text(
        json.dumps({"atlas_info": {"atlas": "allen_mouse_10um"}}))
    return base


def test_depth_filter_is_flat_across_staggered_entries(widget, tmp_path):
    from napari_dmc_segload.np2_scene import highest_entry_dv
    w, viewer = widget
    base = _write_staggered_animal(tmp_path / "animalS", [4000.0, 3000.0, 3500.0, 3800.0])
    w.form.input_path.value = str(base)
    w._visualize()
    full = len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data)
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = 3000.0
    layer = viewer.layers[ProbeVisWidget.POINTS_LAYER]
    assert 0 < len(layer.data) < full
    dv_ref = highest_entry_dv(w._frames, "dv_coords")
    dv_cut = dv_ref + 3000.0 / 10.0
    dv_index = w._atlas.space.axes_description.index("si")   # ("ap","si","rl") -> 1
    assert np.all(np.asarray(layer.data)[:, dv_index] <= dv_cut + 1e-6)
