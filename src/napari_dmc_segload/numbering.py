"""Slice-numbering strategies for the section widget.

Pure functions — no napari. `order_by` and `row_tol` were module globals in the
original dmc_section.py script; here they are explicit parameters.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROW_TOL = 0.05   # grid mode: centroids within this fraction of overview height = same row


def next_slice_number(out_base: Path, animal_id: str) -> int:
    """1 + the highest slice number already written under out_base, else 1.

    Scans every channel subdirectory (out_base/<channel>/) for files named
    ``<animal_id>_sNNN.tif`` and returns ``max(NNN) + 1``. A missing/empty tree
    or no matching files yields 1. The maximum is taken across all channels so an
    interrupted export cannot desynchronise them on the next run.
    """
    out_base = Path(out_base)
    if not out_base.is_dir():
        return 1
    pat = re.compile(rf"^{re.escape(animal_id)}_s(\d+)$")
    highest = 0
    for tif in out_base.glob(f"*/{animal_id}_s*.tif"):
        m = pat.match(tif.stem)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


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
