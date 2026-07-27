"""Per-section horizontal-flip behaviour of the Section widget."""
import numpy as np
import tifffile
from napari_dmc_segload._section_widget import SectionWidget, flip_label


def _rect(y0, x0, y1, x1):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def _write_gradient_tif(path, n_ch=3, h=40, w=60):
    """Per-channel image whose value ramps along x, so a left-right flip is
    detectable (a constant image would be flip-invariant)."""
    row = (np.arange(w, dtype=np.uint16) * 10)
    arr = np.stack([np.broadcast_to(row + c * 1000, (h, w)).astype(np.uint16)
                    for c in range(n_ch)], axis=0)
    tifffile.imwrite(str(path), arr)
    return arr


def test_flip_label():
    assert flip_label(3, False) == "3"
    assert flip_label(3, True) == "3 ⇄"
    assert flip_label(0, True) == "0 ⇄"
    assert flip_label(0, False) == "0"


def test_export_mirrors_only_flagged_section(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    src = _write_gradient_tif(p, n_ch=3, h=40, w=60)
    out_root = tmp_path / "dmc_input"
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.animal_id.value = "m1"
    w.form.output_root.value = str(out_root)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.order_by.value = "draw"
    w.form.overview_scale.value = 0.1
    w._load_overview()
    # top half -> s001 (flagged flip), bottom half -> s002 (untouched)
    viewer.layers["slices"].data = [_rect(0, 0, 2, 6), _rect(2, 0, 4, 6)]
    w._toggle_flip(0)          # flag the first (drawn-order) section

    w._export()

    labels = ("green", "cy3", "dapi")
    for ci, label in enumerate(labels):
        top = tifffile.imread(str(out_root / "m1" / "stitched" / label / "m1_s001.tif"))
        bot = tifffile.imread(str(out_root / "m1" / "stitched" / label / "m1_s002.tif"))
        assert np.array_equal(top, src[ci, 0:20, 0:60][:, ::-1]), f"{label} s001 not mirrored"
        assert np.array_equal(bot, src[ci, 20:40, 0:60]), f"{label} s002 wrongly altered"


def test_flip_survives_renumber(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    _write_gradient_tif(p)
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.order_by.value = "draw"
    w.form.overview_scale.value = 0.1
    w._load_overview()
    viewer.layers["slices"].data = [_rect(0, 0, 2, 6)]
    w._toggle_flip(0)
    assert bool(w._get_flips()[0]) is True

    # adding a rectangle triggers _refresh_numbers, which must NOT wipe flip
    viewer.layers["slices"].data = [_rect(0, 0, 2, 6), _rect(2, 0, 4, 6)]
    flips = w._get_flips()
    assert flips[0] is True or flips[0] == True     # noqa: E712 - numpy bool
    labels = list(viewer.layers["slices"].features["label"])
    assert labels[0].endswith("⇄")


def test_mode_guards(make_napari_viewer, tmp_path):
    p = tmp_path / "slide.tif"
    _write_gradient_tif(p)
    viewer = make_napari_viewer()
    w = SectionWidget(viewer)
    w.form.input_path.value = str(p)
    w.form.channel_labels.value = "green,cy3,dapi"
    w.form.order_by.value = "manual"
    w.form.overview_scale.value = 0.1
    w._load_overview()
    viewer.layers["slices"].data = [_rect(0, 0, 2, 6)]

    w._toggle_flip_mode()
    assert w._flip_mode is True
    assert w._shapes.mode == "select"

    w._number_mode()
    assert w._flip_mode is False

    # toggling flip changes only the flip flag, never the slice number
    before = w._get_numbers()[0]
    w._toggle_flip(0)
    assert w._get_flips()[0] == True     # noqa: E712
    assert w._get_numbers()[0] == before
