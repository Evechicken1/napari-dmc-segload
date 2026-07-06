"""Slice-numbering strategies for the section widget.

Pure functions — no napari. `order_by` and `row_tol` were module globals in the
original dmc_section.py script; here they are explicit parameters.
"""

from __future__ import annotations

import numpy as np

ROW_TOL = 0.05   # grid mode: centroids within this fraction of overview height = same row


def compute_slice_numbers(rects, ov_h: int, order_by: str,
                          row_tol: float = ROW_TOL) -> list[int]:
    """Return a slice number per rectangle, in the SAME order as `rects`.

    Numbering follows `order_by` ("draw" / "vertical" / "grid"). Centroids are in
    overview (row, col) coords. `order_by == "manual"` is handled by the widget,
    not here; passing it falls through to the "grid" branch.
    """
    n = len(rects)
    if n == 0:
        return []
    cents = [(v[:, 0].mean(), v[:, 1].mean()) for v in rects]
    idx = list(range(n))

    if order_by == "draw":
        order = idx
    elif order_by == "vertical":
        order = sorted(idx, key=lambda i: cents[i][0])
    else:  # "grid": rows top->bottom, left->right within a row
        tol = row_tol * ov_h
        rows: list[list[int]] = []
        for i in sorted(idx, key=lambda i: cents[i][0]):
            for row in rows:
                if abs(cents[i][0] - cents[row[0]][0]) <= tol:
                    row.append(i)
                    break
            else:
                rows.append([i])
        order = [i for row in rows for i in sorted(row, key=lambda i: cents[i][1])]

    numbers = [0] * n
    for slice_no, i in enumerate(order, start=1):
        numbers[i] = slice_no
    return numbers
