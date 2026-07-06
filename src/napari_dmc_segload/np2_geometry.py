import numpy as np


def generate_site_coords(track_vertices, n_rows, n_columns=2,
                         row_pitch_um=15.0, column_spacing_um=32.0):
    """Reconstruct NP2.0 recording-site coordinates along a traced shank track.

    Parameters
    ----------
    track_vertices : (N>=2, 2) array of (Position Y, Position X) polyline
        vertices in image pixels, ordered deepest -> shallowest site.
    n_rows : number of electrode rows (depth positions) to place.
    n_columns : 1 (depth ladder) or 2 (NP2.0 two-column grid).
    row_pitch_um, column_spacing_um : NP2.0 geometry; used only to scale the
        horizontal offset between the two columns, self-consistently with the
        arc-length-normalised track.

    Returns
    -------
    coords : (M, 2) array of (Y, X) site coordinates, row-major.
    rows : (M,) int array of row index (0 = deepest endpoint).
    cols : (M,) int array of column index (0 or 1).
    """
    verts = np.asarray(track_vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 2:
        raise ValueError("track_vertices must be an (N>=2, 2) array")

    seg = np.diff(verts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total == 0:
        raise ValueError("track has zero length")

    per_row_arc = total / (n_rows - 1) if n_rows > 1 else 0.0
    offset_px = 0.0
    if n_columns == 2 and n_rows > 1:
        offset_px = (column_spacing_um / row_pitch_um) * per_row_arc

    coords, rows, cols = [], [], []
    for k in range(n_rows):
        target = (k / (n_rows - 1)) * total if n_rows > 1 else 0.0
        j = int(np.searchsorted(cum, target, side="right") - 1)
        j = min(max(j, 0), len(seg_len) - 1)
        local = (target - cum[j]) / seg_len[j] if seg_len[j] > 0 else 0.0
        pos = verts[j] + local * (verts[j + 1] - verts[j])

        tangent = verts[j + 1] - verts[j]
        norm = np.hypot(tangent[0], tangent[1])
        tangent = tangent / norm if norm > 0 else np.array([1.0, 0.0])
        perp = np.array([-tangent[1], tangent[0]])  # rotate 90 deg in (Y, X)

        if n_columns == 2 and n_rows > 1:
            coords.append(pos - 0.5 * offset_px * perp); rows.append(k); cols.append(0)
            coords.append(pos + 0.5 * offset_px * perp); rows.append(k); cols.append(1)
        else:
            coords.append(pos); rows.append(k); cols.append(0)

    return np.array(coords), np.array(rows, dtype=int), np.array(cols, dtype=int)
