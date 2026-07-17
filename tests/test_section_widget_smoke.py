import numpy as np
import tifffile
from napari_dmc_segload._section_widget import SectionWidget


def _write_multichannel_tif(path, n_ch=3, h=40, w=60):
    arr = np.stack(
        [np.full((h, w), (c + 1) * 1000, dtype=np.uint16) for c in range(n_ch)],
        axis=0,
    )
    tifffile.imwrite(str(path), arr)
    return arr


def _rect(y0, x0, y1, x1):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def test_load_adds_image_and_slices_layers(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    _write_multichannel_tif(p, n_ch=3)
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.overview_scale.value = 0.1
    w._load_overview()
    names = [ly.name for ly in viewer.layers]
    assert "green" in names and "cy3" in names and "dapi" in names
    assert "slices" in names


def test_export_writes_dmc_tree(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    _write_multichannel_tif(p, n_ch=3, h=40, w=60)
    out_root = tmp_path / "dmc_input"
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.animal_id.value = "m1"
    w.form.output_root.value = str(out_root)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.order_by.value = "draw"   # avoid manual click-numbering in a headless test
    w.form.overview_scale.value = 0.1
    w._load_overview()
    # one rectangle covering the whole 4x6 overview -> full 40x60 crop
    viewer.layers["slices"].data = [_rect(0, 0, 4, 6)]
    w._export()
    for label in ("green", "cy3", "dapi"):
        f = out_root / "m1" / "stitched" / label / "m1_s001.tif"
        assert f.exists(), f"missing {f}"
    crop = tifffile.imread(str(out_root / "m1" / "stitched" / "cy3" / "m1_s001.tif"))
    assert crop.dtype == np.uint16 and crop.shape == (40, 60)
    # channel 1 constant value 2000 preserved
    assert np.all(crop == 2000)
    # section layers cleared after export
    assert "slices" not in [ly.name for ly in viewer.layers]


def test_export_continues_numbering_from_existing_slices(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    _write_multichannel_tif(p, n_ch=3, h=40, w=60)
    out_root = tmp_path / "dmc_input"
    # pretend a previous slide already wrote s001, s002
    stitched = out_root / "m1" / "stitched"
    for label in ("green", "cy3", "dapi"):
        (stitched / label).mkdir(parents=True, exist_ok=True)
        for n in (1, 2):
            (stitched / label / f"m1_s{n:03d}.tif").touch()

    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.animal_id.value = "m1"
    w.form.output_root.value = str(out_root)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.order_by.value = "draw"
    w.form.overview_scale.value = 0.1
    w._load_overview()
    # two rectangles -> should land at s003, s004, not overwrite s001/s002
    viewer.layers["slices"].data = [_rect(0, 0, 2, 6), _rect(2, 0, 4, 6)]
    w._export()
    green = stitched / "green"
    assert (green / "m1_s003.tif").exists()
    assert (green / "m1_s004.tif").exists()
    # the pre-existing placeholders are untouched (still empty)
    assert (green / "m1_s001.tif").stat().st_size == 0


def test_export_without_load_is_a_noop(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w._export()  # must not raise
    assert len(viewer.layers) == 0
