import numpy as np


def place_electrodes_on_track(track_vertices, z_um, x_um):
    """Reconstruct histology coordinates for electrodes along a traced shank track.

    Parameters
    ----------
    track_vertices : (N>=2, 2) array of (Position Y, Position X) polyline
        vertices in image pixels. Trace direction does not matter: the endpoint
        lower in the image (larger Position Y) is taken as the shank tip
        (deepest electrode, z_min), and the vertices are oriented accordingly.
    z_um, x_um : parallel per-electrode on-shank coordinates (micrometres).

    The tip/top polyline endpoints correspond to the electrodes at min(z_um)
    and max(z_um). Each electrode is placed at arc-length fraction
    (z - z_min) / (z_max - z_min). Columns are offset by (x - x_centre), scaled
    by the pixels-per-micron implied by the depth span, along the local
    perpendicular to the track.

    Returns
    -------
    coords : (M, 2) array of (Y, X) histology coordinates, one per electrode,
        in input order.
    """
    verts = np.asarray(track_vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 2:
        raise ValueError("track_vertices must be an (N>=2, 2) array")

    # Auto-orient by depth: the traced endpoint lower in the image (larger
    # Position Y, column 0) is the tip (deepest electrode). Flip the vertex
    # order if needed so the tip endpoint comes first, making the result
    # independent of the direction the shank was traced.
    if verts[0, 0] < verts[-1, 0]:
        verts = verts[::-1]

    z = np.asarray(z_um, dtype=float)
    x = np.asarray(x_um, dtype=float)
    if z.ndim != 1 or z.shape != x.shape:
        raise ValueError("z_um and x_um must be 1-D arrays of equal length")

    seg = np.diff(verts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total == 0:
        raise ValueError("track has zero length")

    z_min, z_max = float(z.min()), float(z.max())
    z_span = z_max - z_min
    if z_span == 0:
        raise ValueError("all electrodes share the same depth (z_span == 0)")
    px_per_um = total / z_span
    x_centre = 0.5 * (float(x.min()) + float(x.max()))

    coords = []
    for zi, xi in zip(z, x):
        target = (zi - z_min) / z_span * total
        j = int(np.searchsorted(cum, target, side="right") - 1)
        j = min(max(j, 0), len(seg_len) - 1)
        local = (target - cum[j]) / seg_len[j] if seg_len[j] > 0 else 0.0
        pos = verts[j] + local * (verts[j + 1] - verts[j])

        tangent = verts[j + 1] - verts[j]
        norm = np.hypot(tangent[0], tangent[1])
        tangent = tangent / norm if norm > 0 else np.array([1.0, 0.0])
        perp = np.array([-tangent[1], tangent[0]])  # rotate 90 deg in (Y, X)

        coords.append(pos + (xi - x_centre) * px_per_um * perp)

    return np.array(coords)
