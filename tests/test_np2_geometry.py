import numpy as np
import pytest
from napari_dmc_segload.np2_geometry import place_electrodes_on_track


def test_electrodes_land_at_true_depth_fraction():
    verts = np.array([[0.0, 0.0], [705.0, 0.0]])  # straight, L=705 px, X=0
    z = np.array([0.0, 15.0, 705.0])
    x = np.array([43.0, 43.0, 43.0])  # single centre column -> no offset
    coords = place_electrodes_on_track(verts, z, x)
    np.testing.assert_allclose(coords[:, 0], [0.0, 15.0, 705.0])
    np.testing.assert_allclose(coords[:, 1], 0.0)


def test_two_columns_symmetric_about_centreline():
    verts = np.array([[0.0, 0.0], [705.0, 0.0]])  # px_per_um = 705/705 = 1
    # Two depth rows (z 0 and 705) so the depth span defines the pixel scale;
    # each row has both columns at x = 27 and 59 (shank centre 43 -> +/-16 px).
    z = np.array([0.0, 0.0, 705.0, 705.0])
    x = np.array([27.0, 59.0, 27.0, 59.0])
    coords = place_electrodes_on_track(verts, z, x)
    np.testing.assert_allclose(coords[0], [0.0, -16.0])
    np.testing.assert_allclose(coords[1], [0.0, 16.0])
    np.testing.assert_allclose(coords[2], [705.0, -16.0])
    np.testing.assert_allclose(coords[3], [705.0, 16.0])


def test_distributed_by_arc_length_on_bent_track():
    # L-shaped: seg1 Y 0..100 (len 100), seg2 X 0..100 (len 100), total 200 px.
    verts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    z = np.array([0.0, 50.0, 100.0])  # z_span 100 -> fractions 0, .5, 1
    x = np.array([43.0, 43.0, 43.0])  # centre -> no offset
    coords = place_electrodes_on_track(verts, z, x)
    np.testing.assert_allclose(coords[0], [0.0, 0.0])
    np.testing.assert_allclose(coords[1], [100.0, 0.0])
    np.testing.assert_allclose(coords[2], [100.0, 100.0])


def test_too_few_vertices_raises():
    with pytest.raises(ValueError):
        place_electrodes_on_track(np.array([[0.0, 0.0]]),
                                  np.array([0.0]), np.array([27.0]))


def test_zero_length_track_raises():
    with pytest.raises(ValueError):
        place_electrodes_on_track(np.array([[1.0, 1.0], [1.0, 1.0]]),
                                  np.array([0.0, 15.0]), np.array([27.0, 27.0]))


def test_zero_depth_span_raises():
    with pytest.raises(ValueError):
        place_electrodes_on_track(np.array([[0.0, 0.0], [100.0, 0.0]]),
                                  np.array([5.0, 5.0]), np.array([27.0, 59.0]))
