# NP2.0 Electrode Reconstruction From Registered Tract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 2D per-section `neuropixels_2.0` feature with a post-registration step that reconstructs every NP2.0 electrode's 3D atlas location by reusing dmc's per-shank consensus line fit and placing the full NP2.0 electrode template along it.

**Architecture:** Trace shanks as dmc's `neuropixels_probe` (4 shanks); dmc registers + fits. A new pure `np2_tract.py` recovers each shank's line from dmc's tract CSV (`coords ≈ a + b·distance_to_tip`) and places the `np2_layout.py` template at `distance_to_tip = z + 175 µm`, offsetting the two columns along the shank-to-shank axis. `np2_regions.py` annotates atlas regions. A new `_electrodes_widget.py` orchestrates and writes one CSV per shank. The obsolete 2D feature (`neuropixels_2.0` seg type, `np2_geometry.py`, `save_sites`, Generate button) is removed.

**Tech Stack:** Python, numpy, pandas, napari, magicgui, bg_atlasapi, pytest.

**Environment:** base `python` has NO numpy/napari — run pytest/git with the env:
`source ~/anaconda3/etc/profile.d/conda.sh && conda activate napari-env`, then `cd` into the worktree
`c:/Users/xaand/Documents/PhD/Analysis/Informatics/Python/napari-dmc-segload-np2tract`.
Branch: `feat/np2-electrodes-from-tract`. Run pytest with
`PYTHONPATH="$(pwd)/src" python -m pytest ...` (the editable install points at the main checkout, not this worktree).

**dmc facts this depends on (its stable on-disk contract):**
- Per-shank tract CSV: `input_path/results/neuropixels_probe/neuropixels_probe_<i>/<animal>_neuropixels_probe.csv`.
- Columns used: `distance_to_tip(um)`, `depth(um)`, and atlas-coord columns named per
  `atlas.space.axes_description` via `{ap→ap_coords, si→dv_coords, rl→ml_coords}`.
- dmc coords are exactly linear in `distance_to_tip`:
  `coords = (probe_insert - dtt)/10 · direction_unit + surface_vox`.
- Atlas: `bg_atlasapi.BrainGlobeAtlas(params['atlas_info']['atlas'])`; `atlas.annotation` (3-D int array,
  indexed `[a,b,c]` in `axes_description` order), `atlas.structures.data[id]['acronym'|'name']`.

---

### Task 1: Remove the obsolete 2D `neuropixels_2.0` feature

Do this first so later template changes don't fight the old 2D tests.

**Files:**
- Modify: `src/napari_dmc_segload/constants.py`
- Modify: `src/napari_dmc_segload/_widget.py`
- Modify: `src/napari_dmc_segload/io.py`
- Modify: `tests/test_io.py`
- Modify: `tests/test_widget_smoke.py`
- Delete: `src/napari_dmc_segload/np2_geometry.py`, `tests/test_np2_geometry.py`

- [ ] **Step 1: constants.py — drop the seg type and SITE group**

Replace:

```python
ALL_SEG_TYPES = ["cells", "injection_site", "optic_fiber", "neuropixels_probe",
                 "projections", "neuropixels_2.0"]

# Neuropixels 2.0 recording-site reconstruction.
SITE_SEG_TYPES = {"neuropixels_2.0"}
```

with:

```python
ALL_SEG_TYPES = ["cells", "injection_site", "optic_fiber", "neuropixels_probe",
                 "projections"]
```

- [ ] **Step 2: _widget.py — imports**

Replace:

```python
from .constants import (
    DISPLAY_CMAP,
    REFERENCE_OPACITY,
    SEGMENTATION_OPACITY,
    POINT_SEG_TYPES,
    SHAPE_SEG_TYPES,
    PROBE_SEG_TYPES,
    SITE_SEG_TYPES,
    ALL_SEG_TYPES,
)
from .channel_scanner import ChannelScanner
from .np2_layout import N_SHANKS, shank_electrodes
from .np2_geometry import place_electrodes_on_track
from .io import (
    resolve_image,
    image_stem,
    load_grayscale,
    save_points_layer,
    save_shapes_layer,
    save_sites,
)
```

with:

```python
from .constants import (
    DISPLAY_CMAP,
    REFERENCE_OPACITY,
    SEGMENTATION_OPACITY,
    POINT_SEG_TYPES,
    SHAPE_SEG_TYPES,
    PROBE_SEG_TYPES,
    ALL_SEG_TYPES,
)
from .channel_scanner import ChannelScanner
from .io import (
    resolve_image,
    image_stem,
    load_grayscale,
    save_points_layer,
    save_shapes_layer,
)
```

`import numpy as np` at the top stays — `_reorder_layers`/others still... (actually it is now unused; remove it too). Replace `import numpy as np\n` line: delete it. Verify with `grep -n "np\." src/napari_dmc_segload/_widget.py` returning nothing before committing; if it returns matches, keep the import.

- [ ] **Step 3: _widget.py — remove the Generate button**

Replace:

```python
        self.btn_generate = QPushButton("Generate NP2.0 sites")
        self.btn_generate.clicked.connect(self._generate_sites)
        self.layout().addWidget(self.btn_generate)

        self.btn_save = QPushButton("Save segmentation")
```

with:

```python
        self.btn_save = QPushButton("Save segmentation")
```

- [ ] **Step 4: _widget.py — remove the SITE branch in `_add_tracing_layers`**

Replace:

```python
    def _add_tracing_layers(self, seg_type, seg_channels, n_probes, point_size):
        if seg_type in SITE_SEG_TYPES:
            for i in range(N_SHANKS):
                self.viewer.add_points(name=f"{seg_type}_track_{i}", size=point_size)
        elif seg_type in PROBE_SEG_TYPES:
```

with:

```python
    def _add_tracing_layers(self, seg_type, seg_channels, n_probes, point_size):
        if seg_type in PROBE_SEG_TYPES:
```

- [ ] **Step 5: _widget.py — delete `_sites_for_shank` and `_generate_sites`**

Delete both whole methods — the block starting at `    def _sites_for_shank(self, i):` up to (but not including) `    def _reorder_layers(self, ref, seg, seg_type, n_probes):`.

- [ ] **Step 6: _widget.py — remove the SITE branch in `_reorder_layers`**

Replace:

```python
        if seg_type in PROBE_SEG_TYPES:
            top_to_bottom += [f"{seg_type}_{i}" for i in range(n_probes)]
        elif seg_type in SITE_SEG_TYPES:
            for i in range(N_SHANKS):
                top_to_bottom.append(f"{seg_type}_track_{i}")
                top_to_bottom.append(f"{seg_type}_sites_{i}")
        else:  # point/shape tracing layers share the channel name
            top_to_bottom += list(seg)
```

with:

```python
        if seg_type in PROBE_SEG_TYPES:
            top_to_bottom += [f"{seg_type}_{i}" for i in range(n_probes)]
        else:  # point/shape tracing layers share the channel name
            top_to_bottom += list(seg)
```

- [ ] **Step 7: _widget.py — remove the SITE branch in `_save_segmentation`**

Replace:

```python
        if seg_type in SITE_SEG_TYPES:
            saved = False
            for i in range(N_SHANKS):
                result = self._sites_for_shank(i)
                if result is None:
                    continue
                coords, records = result
                save_sites(coords, records, input_path, seg_type,
                           f"shank_{i}", stem)
                saved = True
            if not saved:
                show_info("No shank tracks with >=2 points to save.")
                return
            show_info("Segmentation saved.")
            return

        if seg_type in PROBE_SEG_TYPES:
```

with:

```python
        if seg_type in PROBE_SEG_TYPES:
```

- [ ] **Step 8: io.py — delete `save_sites`**

Delete the entire `def save_sites(...):` function (the block from `def save_sites(coords, records, input_path: Path, seg_type: str,` to the end of its `return save_name`).

- [ ] **Step 9: tests — remove obsolete tests and delete files**

In `tests/test_io.py`, delete everything from the line
`from napari_dmc_segload.io import save_sites` to the end of the file.

In `tests/test_widget_smoke.py`, delete everything from the line `import pandas as pd`
(the second import block, near the bottom) to the end of the file — this removes
`test_np2_load_creates_track_layers`, `test_np2_generate_creates_site_layer`,
`test_np2_save_writes_per_shank_csv`, and `test_np2_uses_four_shanks_regardless_of_spinbox`.

Delete the two 2D-geometry files:

```bash
git rm src/napari_dmc_segload/np2_geometry.py tests/test_np2_geometry.py
```

- [ ] **Step 10: Verify the remaining suite is green**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest -q`
Expected: PASS (all remaining tests; no import errors, no `neuropixels_2.0`/`save_sites`/`np2_geometry` references).
Also run: `grep -rn "SITE_SEG_TYPES\|save_sites\|np2_geometry\|neuropixels_2.0\|_generate_sites\|_sites_for_shank" src tests` → no matches.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "refactor: remove obsolete 2D neuropixels_2.0 feature"
```

---

### Task 2: Extend `np2_layout.py` to the full 1280-electrode/shank template

**Files:**
- Modify: `src/napari_dmc_segload/np2_layout.py`
- Modify: `tests/test_np2_layout.py`

- [ ] **Step 1: Update the tests for the full template**

Replace the entire contents of `tests/test_np2_layout.py` with:

```python
from napari_dmc_segload.np2_layout import (
    N_SHANKS, SHANK_Z_UM, SHANK_X_UM, shank_electrodes)


def test_template_dimensions():
    assert N_SHANKS == 4
    assert len(SHANK_Z_UM) == 640                 # full shank
    assert SHANK_X_UM == [27, 59]
    assert SHANK_Z_UM[0] == 0 and SHANK_Z_UM[-1] == 9585   # 15 * 639
    assert all(SHANK_Z_UM[i + 1] - SHANK_Z_UM[i] == 15 for i in range(639))


def test_shank_electrodes_count_and_ids():
    e = shank_electrodes(2)
    assert len(e) == 1280                         # 640 rows x 2 cols
    assert all(r["shank"] == 2 for r in e)
    assert e[0] == {"shank": 2, "electrode": 0, "row": 0, "col": 0, "x_um": 27, "z_um": 0}
    assert e[1] == {"shank": 2, "electrode": 1, "row": 0, "col": 1, "x_um": 59, "z_um": 0}
    assert e[2] == {"shank": 2, "electrode": 2, "row": 1, "col": 0, "x_um": 27, "z_um": 15}
    assert e[-1]["electrode"] == 1279 and e[-1]["z_um"] == 9585
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_np2_layout.py -v`
Expected: FAIL (current template has 48 rows / 96 electrodes).

- [ ] **Step 3: Update the template extent**

In `src/napari_dmc_segload/np2_layout.py`, replace:

```python
N_SHANKS = 4
# 48 depth rows, 15 um pitch, z from the shank tip (deepest electrode = 0 um).
SHANK_Z_UM = [15 * r for r in range(48)]  # 0, 15, ..., 705
# Two columns across the 70 um shank (centre 43 um).
SHANK_X_UM = [27, 59]
```

with:

```python
N_SHANKS = 4
# Full NP2.0 4-shank (NP2010) shank: 640 depth rows, 15 um pitch, z measured from
# the shank tip (deepest electrode = 0 um). The 27/59 um columns and 15 um pitch
# were parsed from the example snsGeomMap; the 640-row (1280 electrode/shank)
# extent is the NP2010 spec (re-verify against a snsGeomMap/imro if it ever changes).
SHANK_Z_UM = [15 * r for r in range(640)]  # 0, 15, ..., 9585
# Two columns across the 70 um shank (centre 43 um).
SHANK_X_UM = [27, 59]
```

Also update the module docstring's "standard across sessions" note if it says 48 rows (it does not — leave as is).

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_np2_layout.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_layout.py tests/test_np2_layout.py
git commit -m "feat: extend NP2.0 template to full 1280 electrodes per shank"
```

---

### Task 3: `np2_tract.py` — recover line + place electrodes (pure)

**Files:**
- Create: `src/napari_dmc_segload/np2_tract.py`
- Test: `tests/test_np2_tract.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_np2_tract.py`:

```python
import numpy as np
import pandas as pd
import pytest
from napari_dmc_segload.np2_tract import (
    recover_line, recover_probe_insert, shank_direction, lateral_axis,
    place_electrodes)

COLS = ["ap_coords", "dv_coords", "ml_coords"]


def _synthetic_tract(a, b, insert, n=192):
    dtt = np.arange(n) * 20 + 175
    coords = a + np.outer(dtt, b)
    return pd.DataFrame({
        "distance_to_tip(um)": dtt,
        "depth(um)": insert - dtt,
        "ap_coords": np.round(coords[:, 0]).astype(int),
        "dv_coords": np.round(coords[:, 1]).astype(int),
        "ml_coords": np.round(coords[:, 2]).astype(int),
    })


def test_recover_line_returns_tip_point_and_slope():
    a = np.array([100.0, 200.0, 50.0])
    b = np.array([0.02, -0.10, 0.0])
    df = _synthetic_tract(a, b, insert=4000)
    point, slope = recover_line(df, COLS)
    np.testing.assert_allclose(point, a, atol=1.0)      # intercept = tip voxel (dtt=0)
    np.testing.assert_allclose(slope, b, atol=0.01)


def test_recover_probe_insert_from_depth_plus_dtt():
    df = _synthetic_tract(np.zeros(3), np.zeros(3), insert=4000)
    assert abs(recover_probe_insert(df) - 4000) < 1e-6


def test_recover_line_missing_column_raises():
    df = _synthetic_tract(np.zeros(3), np.zeros(3), insert=1000).drop(columns=["ml_coords"])
    with pytest.raises(ValueError):
        recover_line(df, COLS)


def test_lateral_axis_is_unit_and_perpendicular_to_shank():
    shank_dir = np.array([0.0, 1.0, 0.0])           # shank runs along DV
    lat = lateral_axis([0.0, 0.0, 0.0], [0.0, 5.0, 750.0], shank_dir)
    np.testing.assert_allclose(lat, [0.0, 0.0, 1.0], atol=1e-9)  # DV component removed
    assert abs(np.linalg.norm(lat) - 1.0) < 1e-9


def test_place_electrodes_depth_columns_and_offset():
    point = np.array([100.0, 200.0, 50.0])
    slope = np.array([0.0, -0.10, 0.0])
    lat = np.array([0.0, 0.0, 1.0])
    electrodes = [
        {"shank": 0, "electrode": 0, "row": 0, "col": 0, "x_um": 27, "z_um": 0},
        {"shank": 0, "electrode": 1, "row": 0, "col": 1, "x_um": 59, "z_um": 0},
        {"shank": 0, "electrode": 2, "row": 1, "col": 0, "x_um": 27, "z_um": 15},
    ]
    df = place_electrodes(point, slope, 4000, lat, electrodes, COLS)
    # z=0 -> distance_to_tip 175; depth = insert - dtt
    assert df.loc[0, "distance_to_tip_um"] == 175
    assert df.loc[0, "depth_um"] == 3825
    assert df.loc[2, "distance_to_tip_um"] == 190   # +15 um
    # both columns of a row share depth-axis coords (ap, dv); differ on lateral (ml)
    assert df.loc[0, "ap_coords"] == df.loc[1, "ap_coords"]
    assert df.loc[0, "dv_coords"] == df.loc[1, "dv_coords"]
    assert df.loc[1, "ml_coords"] > df.loc[0, "ml_coords"]
    # identity columns preserved
    assert list(df.columns[:6]) == ["shank", "electrode", "row", "col", "x_um", "z_um"]


def test_shank_direction_normalises_and_rejects_zero():
    np.testing.assert_allclose(shank_direction(np.array([0.0, -3.0, 0.0])), [0.0, -1.0, 0.0])
    with pytest.raises(ValueError):
        shank_direction(np.zeros(3))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_np2_tract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_dmc_segload.np2_tract'`.

- [ ] **Step 3: Implement `np2_tract.py`**

Create `src/napari_dmc_segload/np2_tract.py`:

```python
"""Recover dmc's per-shank probe line from its tract CSV and place the NP2.0
electrode template along it. Pure numpy/pandas — no atlas, no napari."""

import numpy as np
import pandas as pd

DIST_COL = "distance_to_tip(um)"   # dmc tract CSV column
DEPTH_COL = "depth(um)"            # dmc tract CSV column


def load_tract_csv(path):
    """Read a dmc probe-tract CSV into a DataFrame."""
    return pd.read_csv(path)


def recover_line(tract_df, coord_cols):
    """Least-squares fit of each atlas-coord column against distance_to_tip.

    dmc writes coords = (probe_insert - dtt)/10 * direction + surface, i.e. a
    line in dtt. Returns (point, slope), both length-3 arrays in `coord_cols`
    order, where `point` is the coord at dtt=0 (the tip voxel) and `slope` is
    d(coord)/d(dtt). The many rows average out dmc's integer-voxel rounding.
    """
    if DIST_COL not in tract_df.columns:
        raise ValueError(f"tract CSV missing column '{DIST_COL}'")
    dtt = tract_df[DIST_COL].to_numpy(dtype=float)
    point = np.empty(3)
    slope = np.empty(3)
    for k, c in enumerate(coord_cols):
        if c not in tract_df.columns:
            raise ValueError(f"tract CSV missing column '{c}'")
        s, intercept = np.polyfit(dtt, tract_df[c].to_numpy(dtype=float), 1)
        slope[k] = s
        point[k] = intercept
    return point, slope


def recover_probe_insert(tract_df):
    """probe_insert (um) = median(depth + distance_to_tip) over all rows."""
    for c in (DEPTH_COL, DIST_COL):
        if c not in tract_df.columns:
            raise ValueError(f"tract CSV missing column '{c}'")
    return float(np.median(tract_df[DEPTH_COL].to_numpy(float)
                           + tract_df[DIST_COL].to_numpy(float)))


def shank_direction(slope):
    """Unit vector along a shank, from its line slope."""
    slope = np.asarray(slope, float)
    n = np.linalg.norm(slope)
    if n == 0:
        raise ValueError("degenerate line: zero slope")
    return slope / n


def lateral_axis(tip_min, tip_max, shank_dir):
    """Unit lateral (column) axis: the shank-to-shank vector with its component
    along the shank direction removed, so it is perpendicular to the shank."""
    v = np.asarray(tip_max, float) - np.asarray(tip_min, float)
    shank_dir = np.asarray(shank_dir, float)
    v = v - np.dot(v, shank_dir) * shank_dir
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("cannot determine lateral axis (shank tips coincide)")
    return v / n


def place_electrodes(point, slope, probe_insert, lat_axis, electrodes, coord_cols,
                     tip_offset_um=175.0, shank_centre_um=43.0, voxel_um=10.0):
    """Place template electrodes along the recovered line.

    For each electrode: distance_to_tip = z_um + tip_offset_um;
    coord = point + slope*dtt + ((x_um - shank_centre)/voxel_um) * lat_axis;
    depth = probe_insert - dtt. `electrodes` is a list of dicts with keys
    shank, electrode, row, col, x_um, z_um. Returns a DataFrame with those
    identity columns plus distance_to_tip_um, depth_um and the three int
    `coord_cols`.
    """
    point = np.asarray(point, float)
    slope = np.asarray(slope, float)
    lat_axis = np.asarray(lat_axis, float)
    rows = []
    for e in electrodes:
        dtt = e["z_um"] + tip_offset_um
        col_off = (e["x_um"] - shank_centre_um) / voxel_um
        coord = point + slope * dtt + col_off * lat_axis
        rec = {"shank": e["shank"], "electrode": e["electrode"], "row": e["row"],
               "col": e["col"], "x_um": e["x_um"], "z_um": e["z_um"],
               "distance_to_tip_um": dtt, "depth_um": probe_insert - dtt}
        for k, c in enumerate(coord_cols):
            rec[c] = int(round(coord[k]))
        rows.append(rec)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_np2_tract.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_tract.py tests/test_np2_tract.py
git commit -m "feat: recover shank line from dmc tract and place NP2.0 electrodes"
```

---

### Task 4: `np2_regions.py` — atlas region annotation

**Files:**
- Create: `src/napari_dmc_segload/np2_regions.py`
- Test: `tests/test_np2_regions.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_np2_regions.py`:

```python
import numpy as np
import pandas as pd
from napari_dmc_segload.np2_regions import annotate_regions

COLS = ["ap_coords", "dv_coords", "ml_coords"]


class _FakeStructures:
    def __init__(self, data):
        self.data = data


class _FakeAtlas:
    def __init__(self, annotation, data):
        self.annotation = annotation
        self.structures = _FakeStructures(data)


def _atlas():
    annot = np.zeros((3, 3, 3), dtype=int)
    annot[1, 1, 1] = 997
    return _FakeAtlas(annot, {997: {"acronym": "grey", "name": "grey matter"}})


def test_labels_in_bounds_voxel():
    df = pd.DataFrame({"ap_coords": [1], "dv_coords": [1], "ml_coords": [1]})
    out = annotate_regions(df, _atlas(), COLS)
    assert out.loc[0, "structure_id"] == 997
    assert out.loc[0, "acronym"] == "grey"
    assert out.loc[0, "name"] == "grey matter"


def test_out_of_bounds_is_clipped_and_root():
    df = pd.DataFrame({"ap_coords": [100], "dv_coords": [0], "ml_coords": [0]})
    out = annotate_regions(df, _atlas(), COLS)          # clips to (2,0,0) -> label 0
    assert out.loc[0, "structure_id"] == 0
    assert out.loc[0, "acronym"] == "root"
    assert out.loc[0, "name"] == "root"


def test_does_not_mutate_input():
    df = pd.DataFrame({"ap_coords": [1], "dv_coords": [1], "ml_coords": [1]})
    annotate_regions(df, _atlas(), COLS)
    assert "structure_id" not in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_np2_regions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_dmc_segload.np2_regions'`.

- [ ] **Step 3: Implement `np2_regions.py`**

Create `src/napari_dmc_segload/np2_regions.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_np2_regions.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_regions.py tests/test_np2_regions.py
git commit -m "feat: annotate electrode coords with atlas regions"
```

---

### Task 5: `_electrodes_widget.py` — orchestration + napari registration

**Files:**
- Create: `src/napari_dmc_segload/_electrodes_widget.py`
- Modify: `src/napari_dmc_segload/napari.yaml`
- Test: `tests/test_electrodes_widget_smoke.py`

- [ ] **Step 1: Write the failing test (core orchestration with a fake atlas)**

Create `tests/test_electrodes_widget_smoke.py`:

```python
import numpy as np
import pandas as pd
import pytest
from napari_dmc_segload._electrodes_widget import reconstruct_electrodes


class _FakeStructures:
    def __init__(self, data):
        self.data = data


class _FakeSpace:
    axes_description = ("ap", "si", "rl")   # allen coronal -> ap, dv, ml


class _FakeAtlas:
    def __init__(self):
        self.annotation = np.full((200, 200, 200), 997, dtype=int)
        self.structures = _FakeStructures({997: {"acronym": "grey", "name": "grey matter"}})
        self.space = _FakeSpace()


def _write_tract(shank_dir, tip, insert):
    shank_dir.mkdir(parents=True)
    dtt = np.arange(60) * 20 + 175
    # line: dv decreases as we move away from tip; ap/ml constant
    ap = np.full(60, tip[0])
    dv = tip[1] - 0.05 * dtt
    ml = np.full(60, tip[2])
    pd.DataFrame({
        "distance_to_tip(um)": dtt,
        "depth(um)": insert - dtt,
        "ap_coords": np.round(ap).astype(int),
        "dv_coords": np.round(dv).astype(int),
        "ml_coords": np.round(ml).astype(int),
    }).to_csv(shank_dir / "animal01_neuropixels_probe.csv")


@pytest.fixture
def animal_dir(tmp_path):
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    _write_tract(root / "neuropixels_probe_0", tip=(60, 120, 40), insert=2000)
    _write_tract(root / "neuropixels_probe_1", tip=(60, 120, 65), insert=2000)  # +25 vox = +250 um
    return base


def test_reconstruct_writes_one_csv_per_shank(animal_dir):
    written = reconstruct_electrodes(animal_dir, _FakeAtlas())
    assert len(written) == 2
    for out in written:
        assert out.name == "animal01_np2_electrodes.csv"
        df = pd.read_csv(out)
        expected = {"shank", "electrode", "row", "col", "x_um", "z_um",
                    "distance_to_tip_um", "depth_um",
                    "ap_coords", "dv_coords", "ml_coords",
                    "structure_id", "acronym", "name"}
        assert expected <= set(df.columns)
        # inside-brain filtering removed most of the 1280 template electrodes
        assert 0 < len(df) < 1280
        assert (df["depth_um"] >= -5).all()


def test_reconstruct_requires_two_shanks(tmp_path):
    base = tmp_path / "animal01"
    _write_tract(base / "results" / "neuropixels_probe" / "neuropixels_probe_0",
                 tip=(60, 120, 40), insert=2000)
    with pytest.raises(ValueError):
        reconstruct_electrodes(base, _FakeAtlas())


def test_reconstruct_missing_results_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        reconstruct_electrodes(tmp_path / "nope", _FakeAtlas())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_electrodes_widget_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_dmc_segload._electrodes_widget'`.

- [ ] **Step 3: Implement `_electrodes_widget.py`**

Create `src/napari_dmc_segload/_electrodes_widget.py`:

```python
"""Dock widget: reconstruct NP2.0 electrode locations from dmc's per-shank
registered probe tract, after the animal has been registered in dmc.

Trace shanks as dmc `neuropixels_probe` (4 shanks), run dmc registration +
Results, then run this. Writes one CSV per shank with each electrode's atlas
coordinates and region.
"""

import json
from pathlib import Path

from natsort import natsorted
from napari import Viewer
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton
from magicgui import magicgui
from magicgui.widgets import FunctionGui

from .np2_layout import shank_electrodes
from .np2_tract import (
    load_tract_csv, recover_line, recover_probe_insert, shank_direction,
    lateral_axis, place_electrodes)
from .np2_regions import annotate_regions

# dmc names atlas-coord columns from the atlas axis order via this map.
AXIS_TO_COORD_COL = {"ap": "ap_coords", "si": "dv_coords", "rl": "ml_coords"}
INSIDE_BRAIN_DEPTH_UM = -5   # dmc convention: depth >= -5 um is inside the brain


def reconstruct_electrodes(input_path, atlas):
    """Reconstruct + write per-shank electrode CSVs. Returns list of paths.

    `atlas` is a BrainGlobeAtlas-like object with `.annotation`, `.structures`,
    and `.space.axes_description`. Raises FileNotFoundError if the dmc results
    are missing and ValueError if fewer than two shanks were traced.
    """
    input_path = Path(input_path)
    results_root = input_path / "results" / "neuropixels_probe"
    if not results_root.is_dir():
        raise FileNotFoundError(
            f"no 'results/neuropixels_probe' under {input_path}; run dmc Results first")

    shank_dirs = natsorted([p for p in results_root.iterdir() if p.is_dir()],
                           key=lambda p: p.name)
    if len(shank_dirs) < 2:
        raise ValueError("need >= 2 traced shanks to determine the lateral (column) axis")

    coord_cols = [AXIS_TO_COORD_COL[ax] for ax in atlas.space.axes_description]

    fits = []  # (shank_dir, point, slope, probe_insert)
    for sd in shank_dirs:
        csvs = list(sd.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"no tract CSV in {sd}")
        df = load_tract_csv(csvs[0])
        point, slope = recover_line(df, coord_cols)
        fits.append((sd, point, slope, recover_probe_insert(df)))

    tip_min = fits[0][1]
    tip_max = fits[-1][1]
    animal = input_path.name

    written = []
    for shank_idx, (sd, point, slope, probe_insert) in enumerate(fits):
        lat = lateral_axis(tip_min, tip_max, shank_direction(slope))
        electrodes = shank_electrodes(shank_idx)
        df = place_electrodes(point, slope, probe_insert, lat, electrodes, coord_cols)
        df = annotate_regions(df, atlas, coord_cols)
        df = df[df["depth_um"] >= INSIDE_BRAIN_DEPTH_UM].reset_index(drop=True)
        out = sd / f"{animal}_np2_electrodes.csv"
        df.to_csv(out)
        written.append(out)
    return written


def _load_atlas(input_path):
    """Build the BrainGlobeAtlas named in the animal's params.json."""
    params_fn = Path(input_path) / "params.json"
    if not params_fn.is_file():
        raise FileNotFoundError(f"no params.json under {input_path}")
    with open(params_fn) as f:
        params = json.load(f)
    atlas_name = params["atlas_info"]["atlas"]
    from bg_atlasapi import BrainGlobeAtlas   # imported lazily (heavy, optional in tests)
    return BrainGlobeAtlas(atlas_name)


def _build_form() -> FunctionGui:
    @magicgui(
        layout="vertical",
        input_path=dict(widget_type="FileEdit", mode="d",
                        label="input path (animal_id): "),
        call_button=False,
    )
    def form(input_path):
        pass
    return form


class ElectrodesWidget(QWidget):
    """Dock widget: reconstruct NP2.0 electrodes from the registered probe tract."""

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.form = _build_form()
        self.layout().addWidget(self.form.native)
        self.btn = QPushButton("Reconstruct electrodes")
        self.btn.clicked.connect(self._reconstruct)
        self.layout().addWidget(self.btn)

    def _reconstruct(self):
        p = self.form.input_path.value
        if not p or not Path(p).is_dir():
            show_info("Select a valid input path (animal_id) directory.")
            return
        try:
            atlas = _load_atlas(p)
            written = reconstruct_electrodes(p, atlas)
        except (FileNotFoundError, ValueError, KeyError) as e:
            show_info(str(e))
            return
        show_info(f"Wrote {len(written)} shank electrode file(s).")
```

- [ ] **Step 4: Run the smoke tests to verify they pass**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_electrodes_widget_smoke.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Register the widget in `napari.yaml`**

Replace the `contributions:` block of `src/napari_dmc_segload/napari.yaml` with:

```yaml
contributions:
  commands:
    - id: napari-dmc-segload.SegLoaderWidget
      python_name: napari_dmc_segload._widget:SegLoaderWidget
      title: Open Segmentation Loader
    - id: napari-dmc-segload.SectionWidget
      python_name: napari_dmc_segload._section_widget:SectionWidget
      title: Section slides
    - id: napari-dmc-segload.ElectrodesWidget
      python_name: napari_dmc_segload._electrodes_widget:ElectrodesWidget
      title: NP2.0 electrodes
  widgets:
    - command: napari-dmc-segload.SegLoaderWidget
      display_name: Segmentation loader
    - command: napari-dmc-segload.SectionWidget
      display_name: Section slides
    - command: napari-dmc-segload.ElectrodesWidget
      display_name: NP2.0 electrodes
```

- [ ] **Step 6: Verify import + full suite**

Run: `PYTHONPATH="$(pwd)/src" python -c "from napari_dmc_segload._electrodes_widget import ElectrodesWidget; print('ok')"`
Expected: `ok`
Run: `PYTHONPATH="$(pwd)/src" python -m pytest -q`
Expected: PASS (whole suite green).

- [ ] **Step 7: Commit**

```bash
git add src/napari_dmc_segload/_electrodes_widget.py src/napari_dmc_segload/napari.yaml tests/test_electrodes_widget_smoke.py
git commit -m "feat: add NP2.0 electrodes widget (reconstruct from registered tract)"
```

---

## Self-Review Notes

- **Spec coverage:** workflow-position (Task 5 widget reads dmc results), reuse-dmc's-line via lstsq (Task 3 `recover_line`), probe_insert recovery (Task 3), lateral axis from extreme shanks (Task 3 `lateral_axis` + Task 5 tip_min/tip_max), placement at `z+175` with column offset (Task 3 `place_electrodes`), full 1280 template (Task 2), region annotation (Task 4), inside-brain filter `depth >= -5` (Task 5), distinct `*_np2_electrodes.csv` filename (Task 5), axis-order handling via `axes_description`→`coord_cols` (Task 5), all removals (Task 1). All covered.
- **Type/signature consistency:** `recover_line(df, coord_cols) -> (point, slope)`, `place_electrodes(point, slope, probe_insert, lat_axis, electrodes, coord_cols, ...)`, `lateral_axis(tip_min, tip_max, shank_dir)`, `annotate_regions(coords_df, atlas, coord_cols)`, `reconstruct_electrodes(input_path, atlas)` — all call sites in Task 5 match Task 3/4 definitions. `shank_electrodes(i)` dict keys (`shank, electrode, row, col, x_um, z_um`) match what `place_electrodes` reads.
- **Placeholder scan:** none; every code step is complete.
- **Note for executor:** Task 1 Step 2 asks to remove `import numpy as np` from `_widget.py` only if unused after the edits — the grep check is included so the executor confirms rather than guesses.
