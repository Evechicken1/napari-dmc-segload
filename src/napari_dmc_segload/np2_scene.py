"""Scene data for the NP2.0 probe visualization.

Pure numpy/pandas: builds the point array, per-point colours and atlas meshes
that the widget hands to napari. Imports no napari, and takes an atlas-like
object (`.structures`, `.mesh_from_structure`) so tests can inject a fake.
"""

import numpy as np
import pandas as pd

MODE_SHANK = "shank"
MODE_REGION = "region"
MODE_INDEX = "electrode index"
COLOUR_MODES = (MODE_SHANK, MODE_REGION, MODE_INDEX)

# Four visually distinct shank colours (RGBA 0-1).
SHANK_COLORS = np.array([
    [0.90, 0.29, 0.23, 1.0],   # red
    [0.20, 0.60, 0.86, 1.0],   # blue
    [0.95, 0.77, 0.06, 1.0],   # amber
    [0.18, 0.80, 0.44, 1.0],   # green
])
UNKNOWN_REGION_COLOR = np.array([0.5, 0.5, 0.5, 1.0])

# Viridis anchor stops, interpolated per channel — keeps this module dependency-free.
_INDEX_CMAP_STOPS = np.array([
    [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
])


def electrode_points(frames, coord_cols, max_depth=None):
    """Stack every shank's electrodes into one point array + feature table.

    Shanks are concatenated in ascending index order, rows in file order.
    Returns (points (N,3) float voxel coords, features DataFrame with columns
    shank, acronym, electrode).

    If `max_depth` is not None, electrodes with depth_um greater than it are
    dropped (display filter); the tip stays, the shallow end is trimmed.
    """
    points, feats = [], []
    for shank in sorted(frames):
        df = frames[shank]
        if max_depth is not None:
            df = df[df["depth_um"] <= max_depth]
        points.append(df[coord_cols].to_numpy(dtype=float))
        feats.append(pd.DataFrame({
            "shank": df["shank"].to_numpy(dtype=int),
            "acronym": df["acronym"].astype(str).to_numpy(),
            "electrode": df["electrode"].to_numpy(dtype=int),
        }))
    pts = np.concatenate(points, axis=0) if points else np.zeros((0, 3))
    return pts, pd.concat(feats, ignore_index=True)


def tract_points(frames, coord_cols):
    """Stack every shank's registered tract samples into one point array.

    `frames` is {shank: tract_df}. Tract CSVs carry no shank column, so the
    shank id comes from the dict key, broadcast across that shank's rows.
    Shanks are concatenated in ascending index order. Returns (points (N,3)
    float voxel coords, features DataFrame with a single 'shank' column) so the
    result can be coloured through the shared `point_colors(.., MODE_SHANK, ..)`.
    Empty frames give (0,3) points and an empty 'shank' column.
    """
    points, feats = [], []
    for shank in sorted(frames):
        df = frames[shank]
        pts = df[coord_cols].to_numpy(dtype=float)
        points.append(pts)
        feats.append(pd.DataFrame({"shank": np.full(len(pts), shank, dtype=int)}))
    stacked = np.concatenate(points, axis=0) if points else np.zeros((0, 3))
    features = (pd.concat(feats, ignore_index=True) if feats
               else pd.DataFrame({"shank": np.zeros(0, dtype=int)}))
    return stacked, features


def point_colors(features, mode, atlas):
    """RGBA (N,4) for each electrode under the given colour mode.

    `atlas` is only used by MODE_REGION and may be None otherwise.
    """
    if len(features) == 0:
        return np.zeros((0, 4))

    if mode == MODE_SHANK:
        # % len(SHANK_COLORS) assumes a single 4-shank NP2.0 probe (shank
        # indices 0-3); a second probe traced into the same tree (indices 4+)
        # would wrap around and reuse colours.
        idx = features["shank"].to_numpy(dtype=int) % len(SHANK_COLORS)
        return SHANK_COLORS[idx]

    if mode == MODE_REGION:
        structures = atlas.structures
        out = np.tile(UNKNOWN_REGION_COLOR, (len(features), 1))
        for acronym in features["acronym"].unique():
            try:
                rgb = structures[acronym]["rgb_triplet"]
            except (KeyError, TypeError):
                continue
            mask = (features["acronym"] == acronym).to_numpy()
            out[mask, :3] = np.asarray(rgb, dtype=float) / 255.0
        return out

    if mode == MODE_INDEX:
        e = features["electrode"].to_numpy(dtype=float)
        lo, hi = float(e.min()), float(e.max())
        t = np.zeros_like(e) if hi <= lo else (e - lo) / (hi - lo)
        return _index_ramp(t)

    raise ValueError(f"unknown colour mode '{mode}'; expected one of {COLOUR_MODES}")


def _index_ramp(t):
    """Map t in [0,1] to (N,4) RGBA along the viridis anchor stops."""
    x = np.linspace(0.0, 1.0, len(_INDEX_CMAP_STOPS))
    rgb = np.stack([np.interp(t, x, _INDEX_CMAP_STOPS[:, k]) for k in range(3)], axis=1)
    return np.concatenate([rgb, np.ones((len(t), 1))], axis=1)


MIN_MESH_ELECTRODES = 10     # ignore structures a shank merely clips
MAX_MESH_REGIONS = 12        # keep the scene readable
ROOT_OPACITY = 0.08
REGION_OPACITY = 0.25
ROI_OPACITY = 0.5
ROOT_COLOR = np.array([0.6, 0.6, 0.6, 1.0])


def root_mesh(atlas):
    """(vertices, faces) of the whole-brain outline, in um."""
    return _mesh_arrays(atlas.mesh_from_structure("root"))


def region_meshes(frames, atlas, min_electrodes=MIN_MESH_ELECTRODES,
                  max_regions=MAX_MESH_REGIONS):
    """Meshes for the structures the electrodes actually pass through.

    Structures holding fewer than `min_electrodes` electrodes are ignored, as is
    'root' (drawn separately). The rest are taken in descending electrode count
    up to `max_regions`. Returns (meshes, skipped) where each mesh is
    {acronym, vertices, faces, color} and `skipped` lists the acronyms that
    qualified but have no mesh in this atlas.
    """
    counts = {}
    for df in frames.values():
        for acronym, n in df["acronym"].astype(str).value_counts().items():
            counts[acronym] = counts.get(acronym, 0) + int(n)
    counts.pop("root", None)

    wanted = sorted((a for a, n in counts.items() if n >= min_electrodes),
                    key=lambda a: (-counts[a], a))[:max_regions]

    meshes, skipped = [], []
    for acronym in wanted:
        try:
            verts, faces = _mesh_arrays(atlas.mesh_from_structure(acronym))
        except Exception:            # atlas has no mesh file for this structure
            skipped.append(acronym)
            continue
        meshes.append({"acronym": acronym, "vertices": verts, "faces": faces,
                       "color": _structure_color(atlas, acronym)})
    return meshes, skipped


def _mesh_arrays(mesh):
    """(vertices, faces) as plain arrays from a meshio-style mesh."""
    return (np.asarray(mesh.points, dtype=float),
            np.asarray(mesh.cells[0].data, dtype=int))


def _structure_color(atlas, acronym):
    """RGBA for a structure from the atlas palette, grey if unknown."""
    try:
        rgb = atlas.structures[acronym]["rgb_triplet"]
    except (KeyError, TypeError):
        return UNKNOWN_REGION_COLOR.copy()
    return np.array([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0])


def atlas_acronyms(atlas):
    """Sorted list of every structure acronym in the atlas."""
    return sorted({d["acronym"] for d in atlas.structures.values()})


def roi_mesh(atlas, acronym):
    """Mesh dict {acronym, vertices, faces, color} for one structure, drawn as
    a highlighted region-of-interest. Raises ValueError if the atlas has no mesh
    for `acronym` (unknown structure or missing mesh file)."""
    try:
        verts, faces = _mesh_arrays(atlas.mesh_from_structure(acronym))
    except Exception:
        raise ValueError(f"no mesh available for '{acronym}' in this atlas")
    return {"acronym": acronym, "vertices": verts, "faces": faces,
            "color": _structure_color(atlas, acronym)}
