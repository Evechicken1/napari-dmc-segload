import numpy as np
import pytest
from napari_dmc_segload.np2_geometry import place_electrodes_on_track


def test_tip_maps_to_high_Y_endpoint_either_direction():
    # The endpoint lower in the image (larger Y) is the tip (electrode z=0),
    # regardless of the direction the shank was traced.
    z = np.array([0.0, 705.0])
    x = np.array([43.0, 43.0])  # centre column -> no offset
    tip_first = place_electrodes_on_track(np.array([[705.0, 0.0], [0.0, 0.0]]), z, x)
    top_first = place_electrodes_on_track(np.array([[0.0, 0.0], [705.0, 0.0]]), z, x)
    np.testing.assert_allclose(tip_first[0], [705.0, 0.0])  # z=0 tip at high-Y end
    np.testing.assert_allclose(tip_first[1], [0.0, 0.0])    # z=705 top at low-Y end
    np.testing.assert_allclose(top_first, tip_first)        # direction-independent


def test_electrodes_land_at_true_depth_fraction():
    verts = np.array([[705.0, 0.0], [0.0, 0.0]])  # tip (Y=705) first, L=705 px
    z = np.array([0.0, 15.0, 705.0])
    x = np.array([43.0, 43.0, 43.0])  # single centre column -> no offset
    coords = place_electrodes_on_track(verts, z, x)
    np.testing.assert_allclose(coords[:, 0], [705.0, 690.0, 0.0])
    np.testing.assert_allclose(coords[:, 1], 0.0)


def test_two_columns_symmetric_about_centreline():
    verts = np.array([[705.0, 0.0], [0.0, 0.0]])  # tip first; px_per_um = 705/705 = 1
    # Two depth rows (z 0 and 705) so the depth span defines the pixel scale;
    # each row has both columns at x = 27 and 59 (shank centre 43 -> +/-16 px).
    z = np.array([0.0, 0.0, 705.0, 705.0])
    x = np.array([27.0, 59.0, 27.0, 59.0])
    coords = place_electrodes_on_track(verts, z, x)
    # tip row (z=0) at Y=705, top row (z=705) at Y=0
    np.testing.assert_allclose(coords[0, 0], 705.0)
    np.testing.assert_allclose(coords[1, 0], 705.0)
    np.testing.assert_allclose(coords[2, 0], 0.0)
    np.testing.assert_allclose(coords[3, 0], 0.0)
    # columns sit symmetric about the centreline, 16 px to each side
    np.testing.assert_allclose(coords[0, 1], -coords[1, 1])
    np.testing.assert_allclose(abs(coords[0, 1]), 16.0)
    # the same column keeps the same offset across rows
    np.testing.assert_allclose(coords[2, 1], coords[0, 1])


def test_distributed_by_arc_length_on_bent_track():
    # L-shaped, traced top-first: seg1 Y 0..100 (len 100), seg2 X 0..100 (len 100).
    # Auto-orient flips it so the tip (max Y = 100) is the deepest electrode.
    verts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    z = np.array([0.0, 50.0, 100.0])  # z_span 100 -> fractions 0, .5, 1
    x = np.array([43.0, 43.0, 43.0])  # centre -> no offset
    coords = place_electrodes_on_track(verts, z, x)
    np.testing.assert_allclose(coords[0], [100.0, 100.0])  # tip (z=0) at high Y
    np.testing.assert_allclose(coords[1], [100.0, 0.0])    # midway by arc length
    np.testing.assert_allclose(coords[2], [0.0, 0.0])      # top (z=100)


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
