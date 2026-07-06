import numpy as np
import pytest
from napari_dmc_segload.np2_geometry import generate_site_coords


def test_single_column_rows_evenly_spaced_on_straight_track():
    # Straight vertical track: Y 0..100, X constant 0.
    verts = np.array([[0.0, 0.0], [100.0, 0.0]])
    coords, rows, cols = generate_site_coords(verts, n_rows=5, n_columns=1)
    assert coords.shape == (5, 2)
    assert list(rows) == [0, 1, 2, 3, 4]
    assert set(cols) == {0}
    np.testing.assert_allclose(coords[:, 0], [0, 25, 50, 75, 100])
    np.testing.assert_allclose(coords[:, 1], 0.0)


def test_two_columns_symmetric_about_centreline():
    verts = np.array([[0.0, 0.0], [100.0, 0.0]])
    coords, rows, cols = generate_site_coords(
        verts, n_rows=5, n_columns=2,
        row_pitch_um=15.0, column_spacing_um=32.0)
    assert coords.shape == (10, 2)
    assert list(cols) == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # per-row arc = 100/4 = 25 px represents 15 um; full column offset:
    # (32/15)*25 = 53.333..., half = 26.667 to each side (in X).
    half = 0.5 * (32.0 / 15.0) * 25.0
    np.testing.assert_allclose(coords[0, 1], -half)  # row0 col0
    np.testing.assert_allclose(coords[1, 1], +half)  # row0 col1
    # both columns of a row share the same depth (Y).
    np.testing.assert_allclose(coords[0, 0], coords[1, 0])


def test_rows_distributed_by_arc_length_on_bent_track():
    # L-shaped: segment1 Y 0..100 (len 100), segment2 X 0..100 (len 100).
    verts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    coords, rows, cols = generate_site_coords(verts, n_rows=5, n_columns=1)
    # targets by arc length: 0, 50, 100, 150, 200
    np.testing.assert_allclose(coords[0], [0.0, 0.0])
    np.testing.assert_allclose(coords[1], [50.0, 0.0])
    np.testing.assert_allclose(coords[2], [100.0, 0.0])
    np.testing.assert_allclose(coords[3], [100.0, 50.0])
    np.testing.assert_allclose(coords[4], [100.0, 100.0])


def test_single_row_places_one_point_no_offset():
    verts = np.array([[0.0, 0.0], [100.0, 0.0]])
    coords, rows, cols = generate_site_coords(verts, n_rows=1, n_columns=2)
    assert coords.shape == (1, 2)  # degenerate: one point, no column split
    np.testing.assert_allclose(coords[0], [0.0, 0.0])


def test_too_few_vertices_raises():
    with pytest.raises(ValueError):
        generate_site_coords(np.array([[0.0, 0.0]]), n_rows=5)


def test_zero_length_track_raises():
    with pytest.raises(ValueError):
        generate_site_coords(np.array([[1.0, 1.0], [1.0, 1.0]]), n_rows=5)
