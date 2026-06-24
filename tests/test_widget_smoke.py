import numpy as np
from napari_dmc_segload._widget import SegLoaderWidget


def _configure(w, animal_dir, ref, seg, seg_type="cells", idx=0):
    w.form.input_path.value = str(animal_dir)
    w._scan_channels()
    w.form.reference_channels.value = ref
    w.form.segmentation_channels.value = seg
    w.form.seg_type.value = seg_type
    w.form.image_idx.value = idx


def test_scan_populates_channel_choices(make_napari_viewer, animal_dir):
    w = SegLoaderWidget(make_napari_viewer())
    w.form.input_path.value = str(animal_dir)
    w._scan_channels()
    assert list(w.form.reference_channels.choices) == ["dapi", "cy3"]
    assert list(w.form.segmentation_channels.choices) == ["dapi", "cy3"]


def test_load_creates_image_and_points_layers(make_napari_viewer, animal_dir):
    viewer = make_napari_viewer()
    w = SegLoaderWidget(viewer)
    _configure(w, animal_dir, ref=["dapi"], seg=["cy3"], seg_type="cells")
    w._load_into_viewer()
    names = [ly.name for ly in viewer.layers]
    assert "dapi channel" in names
    assert "cy3 channel" in names
    assert "cy3" in names  # empty points tracing layer
    pts = viewer.layers["cy3"]
    assert pts.__class__.__name__ == "Points"


def test_load_injection_creates_shapes_layer(make_napari_viewer, animal_dir):
    viewer = make_napari_viewer()
    w = SegLoaderWidget(viewer)
    _configure(w, animal_dir, ref=[], seg=["cy3"], seg_type="injection_site")
    w._load_into_viewer()
    assert viewer.layers["cy3"].__class__.__name__ == "Shapes"


def test_load_probe_creates_n_probe_point_layers(make_napari_viewer, animal_dir):
    viewer = make_napari_viewer()
    w = SegLoaderWidget(viewer)
    _configure(w, animal_dir, ref=["dapi"], seg=["cy3"], seg_type="neuropixels_probe")
    w.form.n_probes.value = 2
    w._load_into_viewer()
    names = [ly.name for ly in viewer.layers]
    assert "neuropixels_probe_0" in names and "neuropixels_probe_1" in names


def test_save_writes_csv_for_traced_points(make_napari_viewer, animal_dir):
    viewer = make_napari_viewer()
    w = SegLoaderWidget(viewer)
    _configure(w, animal_dir, ref=["dapi"], seg=["cy3"], seg_type="cells")
    w._load_into_viewer()
    viewer.layers["cy3"].data = np.array([[1.0, 2.0], [3.0, 4.0]])
    w._save_segmentation()
    out = animal_dir / "segmentation" / "cells" / "cy3" / "animal01_obj1_1_stitched_cells.csv"
    assert out.exists()
