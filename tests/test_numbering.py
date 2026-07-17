import numpy as np
from napari_dmc_segload.numbering import compute_slice_numbers, next_slice_number


def _rect(y0, x0, y1, x1):
    """Rectangle as napari stores it: 4 corner rows of (row, col)."""
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def test_empty_returns_empty():
    assert compute_slice_numbers([], ov_h=100, order_by="grid") == []


def test_draw_order_is_input_order():
    rects = [_rect(50, 0, 60, 10), _rect(0, 0, 10, 10)]
    assert compute_slice_numbers(rects, ov_h=100, order_by="draw") == [1, 2]


def test_vertical_orders_by_top_edge():
    # second rect is higher up -> should be numbered 1
    rects = [_rect(50, 0, 60, 10), _rect(0, 0, 10, 10)]
    assert compute_slice_numbers(rects, ov_h=100, order_by="vertical") == [2, 1]


def test_grid_rows_top_to_bottom_left_to_right():
    # two rects on top row (left, right), one on bottom row
    top_left = _rect(0, 0, 10, 10)
    top_right = _rect(0, 50, 10, 60)
    bottom = _rect(80, 0, 90, 10)
    nums = compute_slice_numbers([top_right, top_left, bottom],
                                 ov_h=100, order_by="grid")
    # input order: top_right, top_left, bottom
    assert nums == [2, 1, 3]


def _write_slices(out_base, chan, animal_id, numbers):
    """Touch <out_base>/<chan>/<animal_id>_sNNN.tif for each number."""
    d = out_base / chan
    d.mkdir(parents=True, exist_ok=True)
    for n in numbers:
        (d / f"{animal_id}_s{str(n).zfill(3)}.tif").touch()


def test_next_slice_number_missing_dir_starts_at_one(tmp_path):
    assert next_slice_number(tmp_path / "nope" / "stitched", "B2") == 1


def test_next_slice_number_empty_dir_starts_at_one(tmp_path):
    (tmp_path / "green").mkdir()
    assert next_slice_number(tmp_path, "B2") == 1


def test_next_slice_number_continues_after_existing(tmp_path):
    _write_slices(tmp_path, "green", "B2", range(1, 13))
    _write_slices(tmp_path, "cy3", "B2", range(1, 13))
    assert next_slice_number(tmp_path, "B2") == 13


def test_next_slice_number_takes_max_across_ragged_channels(tmp_path):
    # an interrupted export wrote 12 green but only 10 cy3
    _write_slices(tmp_path, "green", "B2", range(1, 13))
    _write_slices(tmp_path, "cy3", "B2", range(1, 11))
    assert next_slice_number(tmp_path, "B2") == 13


def test_next_slice_number_ignores_unrelated_filenames(tmp_path):
    d = tmp_path / "green"
    d.mkdir(parents=True)
    for name in ["notes.txt", "B2.tif", "B2_overview.tif", "B2_sABC.tif"]:
        (d / name).touch()
    assert next_slice_number(tmp_path, "B2") == 1


def test_next_slice_number_ignores_other_animals(tmp_path):
    _write_slices(tmp_path, "green", "B7", range(1, 21))
    _write_slices(tmp_path, "green", "B2", range(1, 4))
    assert next_slice_number(tmp_path, "B2") == 4


def test_next_slice_number_past_zero_pad_width(tmp_path):
    _write_slices(tmp_path, "green", "B2", [999])
    assert next_slice_number(tmp_path, "B2") == 1000
