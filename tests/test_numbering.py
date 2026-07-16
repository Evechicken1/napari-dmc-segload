import numpy as np
from napari_dmc_segload.numbering import compute_slice_numbers


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
