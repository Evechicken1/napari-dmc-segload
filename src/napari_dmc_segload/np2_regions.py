"""Annotate electrode coordinates with brain-atlas regions.

Takes an atlas-like object exposing `.annotation` (3-D int array indexed
[a, b, c] in the atlas axis order) and `.structures` (with a `.data` dict
mapping structure_id -> {'acronym', 'name'}), matching bg_atlasapi's
BrainGlobeAtlas. A fake atlas can be injected for testing.
"""

import numpy as np


def annotate_regions(coords_df, atlas, coord_cols):
    """Return a copy of `coords_df` with structure_id/acronym/name columns from
    the atlas annotation at each row's (a, b, c) voxel (in `coord_cols` order).
    Out-of-bounds voxels are clipped into range; label 0 -> 'root'."""
    annot = atlas.annotation
    a = np.clip(coords_df[coord_cols[0]].to_numpy(int), 0, annot.shape[0] - 1)
    b = np.clip(coords_df[coord_cols[1]].to_numpy(int), 0, annot.shape[1] - 1)
    c = np.clip(coords_df[coord_cols[2]].to_numpy(int), 0, annot.shape[2] - 1)
    sid = np.asarray(annot[a, b, c]).astype(int)

    data = atlas.structures.data if hasattr(atlas.structures, "data") else atlas.structures
    out = coords_df.copy()
    out["structure_id"] = sid
    out["acronym"] = [data[i]["acronym"] if i > 0 and i in data else "root" for i in sid]
    out["name"] = [data[i]["name"] if i > 0 and i in data else "root" for i in sid]
    return out
