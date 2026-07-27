# NP2.0 Probe Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a napari dock widget "NP2.0 probe visualization" that renders the already-reconstructed NP2.0 electrodes in a 3D atlas scene and reports geometric QC, so a bad reconstruction is visible instead of merely well-formed on disk.

**Architecture:** Three new pure modules — `np2_results.py` (find and load the per-shank electrode CSVs), `np2_qc.py` (geometry metrics and report text), `np2_scene.py` (points, colours, atlas meshes) — plus a thin Qt shell `_probe_vis_widget.py` that pushes their output into the viewer. The pure modules import no napari and take an atlas-like object, so everything except the widget is testable headless. The widget is read-only with respect to the reconstruction: it consumes `*_np2_electrodes.csv` and writes only a QC text file.

**Tech Stack:** Python 3.10, napari 0.7.0, magicgui, qtpy, numpy, pandas, natsort, `bg_atlasapi` (the older API — `from bg_atlasapi import BrainGlobeAtlas`), pytest + pytest-qt.

**Spec:** `docs/superpowers/specs/2026-07-24-np2-probe-visualization-design.md`

---

## Environment and working rules

Every command in this plan runs from the worktree with the conda env active:

```bash
cd /c/Users/xaand/Documents/PhD/Analysis/Informatics/Python/napari-dmc-segload-probe-vis
source ~/anaconda3/etc/profile.d/conda.sh && conda activate napari-env
```

`pytest` **must** be run with `PYTHONPATH` pointed at this worktree's `src`, because the editable install points at the main checkout:

```bash
PYTHONPATH="$PWD/src" pytest tests/ -v
```

Tests need a display — napari layers require a GL context, so `QT_QPA_PLATFORM=offscreen` will fail. This is already documented in the README.

Branch is `feat/probe-vis`. Commit after every task.

## Verified facts this plan depends on

Checked directly in `napari-env`, do not re-derive:

- `atlas.resolution == (10.0, 10.0, 10.0)`, `atlas.space.axes_description == ('ap', 'si', 'rl')`, `atlas.annotation.shape == (1320, 800, 1140)`.
- `atlas.mesh_from_structure(acronym)` returns a meshio mesh: `.points` is `(V, 3)` float64 in **µm**, same axis order as the annotation; `.cells[0].data` is `(F, 3)` int triangles. root = 49324 verts / 98638 tris; cortical structures 1.7k–6.3k verts.
- `atlas.structures[acronym]` is a dict containing `rgb_triplet` (0–255 ints).
- napari 0.7.0: `Points.__init__` accepts `features`, `face_color`, `size`, `scale`, `border_width` (**not** `edge_width`). `Surface.__init__` accepts `colormap`, `opacity`, `shading`, `scale`. `napari.utils.colormaps.Colormap([[r,g,b,a],[r,g,b,a]], name=...)` builds a solid colour.

## File structure

| File | Responsibility |
|---|---|
| `src/napari_dmc_segload/np2_results.py` | **new** — where results live on disk: shank folders, shank index, electrode CSV loading, atlas construction, coord-column naming |
| `src/napari_dmc_segload/np2_qc.py` | **new** — pure geometry metrics, region ladder, warnings, report text |
| `src/napari_dmc_segload/np2_scene.py` | **new** — pure scene data: point array, per-point colours, atlas meshes |
| `src/napari_dmc_segload/_probe_vis_widget.py` | **new** — Qt shell: form, button, colour combo, QC panel |
| `src/napari_dmc_segload/np2_tract.py` | modify — `recover_line` gains a `dist_col` parameter |
| `src/napari_dmc_segload/_electrodes_widget.py` | modify — drop private path/atlas helpers, import from `np2_results` |
| `src/napari_dmc_segload/napari.yaml` | modify — register the widget |
| `tests/conftest.py` | modify — add shared NP2 electrode-frame fixtures |
| `tests/test_np2_results.py` | **new** |
| `tests/test_np2_qc.py` | **new** |
| `tests/test_np2_scene.py` | **new** |
| `tests/test_probe_vis_widget_smoke.py` | **new** |
| `README.md` | modify — document the new widget |

`np2_results.py` also absorbs `AXIS_TO_COORD_COL` and `_load_atlas` from `_electrodes_widget.py`. The spec describes moving the shank-folder discovery; these two move with it for the same reason — the visualization widget needs them and there must be exactly one definition of each.

---

## Task 1: `np2_results.py` — locate and load reconstruction results

**Files:**
- Create: `src/napari_dmc_segload/np2_results.py`
- Test: `tests/test_np2_results.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_np2_results.py`:

```python
import json

import numpy as np
import pandas as pd
import pytest

from napari_dmc_segload.np2_results import (
    ELECTRODE_CSV_SUFFIX, coord_cols, load_electrode_csvs, probe_results_dir,
    shank_dirs, shank_index)

COLS = ["ap_coords", "dv_coords", "ml_coords"]


class _FakeSpace:
    axes_description = ("ap", "si", "rl")


class _FakeAtlas:
    space = _FakeSpace()
    resolution = (10.0, 10.0, 10.0)


def _write_electrodes(shank_dir, shank, n=6):
    """Minimal well-formed electrode CSV."""
    shank_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "shank": np.full(n, shank),
        "electrode": np.arange(n),
        "row": np.arange(n) // 2,
        "col": np.arange(n) % 2,
        "x_um": np.where(np.arange(n) % 2 == 0, 27, 59),
        "z_um": (np.arange(n) // 2) * 15,
        "distance_to_tip_um": (np.arange(n) // 2) * 15 + 175.0,
        "depth_um": 4000.0 - ((np.arange(n) // 2) * 15 + 175.0),
        "ap_coords": np.full(n, 600),
        "dv_coords": np.arange(n) + 400,
        "ml_coords": np.full(n, 500 + shank * 25),
        "structure_id": np.full(n, 723),
        "acronym": ["ORBl"] * n,
        "name": ["Orbital area, lateral part"] * n,
    })
    df.to_csv(shank_dir / f"animal01{ELECTRODE_CSV_SUFFIX}", index=False)


@pytest.fixture
def results_animal(tmp_path):
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    for i in (0, 1, 2, 3):
        _write_electrodes(root / f"neuropixels_probe_{i}", shank=i)
    return base


def test_shank_index_parses_trailing_number():
    assert shank_index("neuropixels_probe_2") == 2
    assert shank_index("neuropixels_probe_10") == 10


def test_shank_index_without_number_raises():
    with pytest.raises(ValueError):
        shank_index("neuropixels_probe")


def test_probe_results_dir_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_results_dir(tmp_path / "nope")


def test_shank_dirs_are_natsorted(results_animal):
    names = [d.name for d in shank_dirs(results_animal)]
    assert names == ["neuropixels_probe_0", "neuropixels_probe_1",
                     "neuropixels_probe_2", "neuropixels_probe_3"]


def test_load_electrode_csvs_keyed_by_shank_index(results_animal):
    frames = load_electrode_csvs(results_animal)
    assert sorted(frames) == [0, 1, 2, 3]
    assert all(isinstance(df, pd.DataFrame) for df in frames.values())
    assert frames[2]["shank"].unique().tolist() == [2]


def test_load_electrode_csvs_uses_folder_index_not_sort_position(tmp_path):
    # only shanks 0 and 2 traced: keys must be 0 and 2, not 0 and 1
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    _write_electrodes(root / "neuropixels_probe_0", shank=0)
    _write_electrodes(root / "neuropixels_probe_2", shank=2)
    assert sorted(load_electrode_csvs(base)) == [0, 2]


def test_load_electrode_csvs_no_files_raises(tmp_path):
    base = tmp_path / "animal01"
    (base / "results" / "neuropixels_probe" / "neuropixels_probe_0").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="NP2.0 electrodes"):
        load_electrode_csvs(base)


def test_load_electrode_csvs_missing_column_raises(results_animal):
    bad = results_animal / "results" / "neuropixels_probe" / "neuropixels_probe_1"
    csv = next(bad.glob(f"*{ELECTRODE_CSV_SUFFIX}"))
    pd.read_csv(csv).drop(columns=["depth_um"]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="depth_um"):
        load_electrode_csvs(results_animal)


def test_coord_cols_follows_atlas_axis_order():
    assert coord_cols(_FakeAtlas()) == COLS
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_results.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'napari_dmc_segload.np2_results'`.

- [ ] **Step 3: Write the implementation**

Create `src/napari_dmc_segload/np2_results.py`:

```python
"""Where an animal's NP2.0 reconstruction lives on disk, and how to load it.

dmc writes its registered probe tracts to
`results/neuropixels_probe/neuropixels_probe_<i>/`; the "NP2.0 electrodes"
widget writes one `<animal>_np2_electrodes.csv` per shank into the same folders.
This module owns those locations so the electrodes widget and the visualization
widget agree on them.
"""

import json
import re
from pathlib import Path

import pandas as pd
from natsort import natsorted

PROBE_RESULTS_SUBDIR = ("results", "neuropixels_probe")
ELECTRODE_CSV_SUFFIX = "_np2_electrodes.csv"

# dmc names atlas-coord columns from the atlas axis order via this map.
AXIS_TO_COORD_COL = {"ap": "ap_coords", "si": "dv_coords", "rl": "ml_coords"}

# Columns every consumer of an electrode CSV relies on (coord columns are
# checked separately, since their names depend on the atlas axis order).
REQUIRED_ELECTRODE_COLS = ("shank", "electrode", "distance_to_tip_um",
                           "depth_um", "acronym")


def probe_results_dir(input_path):
    """Path to `results/neuropixels_probe` under an animal folder.

    Raises FileNotFoundError if it does not exist (dmc Results not run yet)."""
    d = Path(input_path).joinpath(*PROBE_RESULTS_SUBDIR)
    if not d.is_dir():
        raise FileNotFoundError(
            f"no 'results/neuropixels_probe' under {input_path}; run dmc Results first")
    return d


def shank_index(folder_name):
    """Physical shank index from a dmc probe folder name, e.g.
    'neuropixels_probe_2' -> 2. Raises ValueError if no number is present."""
    nums = re.findall(r"\d+", str(folder_name))
    if not nums:
        raise ValueError(f"cannot parse a shank index from folder '{folder_name}'")
    return int(nums[-1])


def shank_dirs(input_path):
    """Per-shank result folders, natsorted by name."""
    return natsorted([p for p in probe_results_dir(input_path).iterdir() if p.is_dir()],
                     key=lambda p: p.name)


def load_electrode_csvs(input_path):
    """Load every `*_np2_electrodes.csv` under the probe results.

    Returns {shank_index: DataFrame}, keyed by the index parsed from the folder
    name (never the sort position). Raises FileNotFoundError if no electrode CSV
    exists anywhere, and ValueError if one is missing a required column.
    """
    frames = {}
    for sd in shank_dirs(input_path):
        csvs = sorted(sd.glob(f"*{ELECTRODE_CSV_SUFFIX}"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        missing = [c for c in REQUIRED_ELECTRODE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{csvs[0]} is missing column(s): {', '.join(missing)}")
        frames[shank_index(sd.name)] = df
    if not frames:
        raise FileNotFoundError(
            f"no '*{ELECTRODE_CSV_SUFFIX}' under {probe_results_dir(input_path)}; "
            "run the 'NP2.0 electrodes' widget first")
    return frames


def coord_cols(atlas):
    """dmc's atlas-coordinate column names, in the atlas's own axis order."""
    return [AXIS_TO_COORD_COL[ax] for ax in atlas.space.axes_description]


def load_atlas(input_path):
    """Build the BrainGlobeAtlas named in the animal's params.json."""
    params_fn = Path(input_path) / "params.json"
    if not params_fn.is_file():
        raise FileNotFoundError(f"no params.json under {input_path}")
    with open(params_fn) as f:
        params = json.load(f)
    atlas_name = params["atlas_info"]["atlas"]
    from bg_atlasapi import BrainGlobeAtlas   # imported lazily (heavy, optional in tests)
    return BrainGlobeAtlas(atlas_name)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_results.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_results.py tests/test_np2_results.py
git commit -m "feat: add np2_results — locate and load per-shank electrode CSVs"
```

---

## Task 2: point `_electrodes_widget` at `np2_results`

Removes the duplicate definitions of shank discovery, shank-index parsing, the coord-column map and atlas loading. Behaviour is unchanged, so the existing tests are the regression check.

**Files:**
- Modify: `src/napari_dmc_segload/_electrodes_widget.py`
- Test: `tests/test_electrodes_widget_smoke.py` (unchanged — must stay green)

- [ ] **Step 1: Run the existing tests to confirm the starting point is green**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_electrodes_widget_smoke.py -v
```

Expected: 4 passed.

- [ ] **Step 2: Rewrite the module head and `reconstruct_electrodes`**

In `src/napari_dmc_segload/_electrodes_widget.py`, replace everything from the imports down to the end of `_load_atlas` (i.e. lines 9–99 of the current file) with:

```python
import re
from pathlib import Path

from napari import Viewer
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton
from magicgui import magicgui
from magicgui.widgets import FunctionGui

from .np2_layout import shank_electrodes
from .np2_results import coord_cols, load_atlas, shank_dirs, shank_index
from .np2_tract import (
    load_tract_csv, recover_line, recover_probe_insert, shank_direction,
    lateral_axis, place_electrodes)
from .np2_regions import annotate_regions

INSIDE_BRAIN_DEPTH_UM = -5   # dmc convention: depth >= -5 um is inside the brain


def reconstruct_electrodes(input_path, atlas):
    """Reconstruct + write per-shank electrode CSVs. Returns list of paths.

    `atlas` is a BrainGlobeAtlas-like object with `.annotation`, `.structures`,
    and `.space.axes_description`. Raises FileNotFoundError if the dmc results
    are missing and ValueError if fewer than two shanks were traced.
    """
    input_path = Path(input_path)
    dirs = shank_dirs(input_path)
    if len(dirs) < 2:
        raise ValueError("need >= 2 traced shanks to determine the lateral (column) axis")

    cols = coord_cols(atlas)

    # (shank_dir, shank_idx, point, slope, probe_insert). The shank index is
    # parsed from the folder name (e.g. "neuropixels_probe_2" -> 2), NOT the
    # sort position, so a missing/untraced shank never re-labels the others.
    fits = []
    for sd in dirs:
        csvs = [p for p in sd.glob("*.csv") if not p.name.endswith("_np2_electrodes.csv")]
        if not csvs:
            raise FileNotFoundError(f"no tract CSV in {sd}")
        df = load_tract_csv(csvs[0])
        point, slope = recover_line(df, cols)
        fits.append((sd, shank_index(sd.name), point, slope, recover_probe_insert(df)))

    # Lateral axis spans the two extreme traced shanks (first/last by name order).
    tip_min = fits[0][2]
    tip_max = fits[-1][2]
    animal = input_path.name

    written = []
    for sd, shank_idx, point, slope, probe_insert in fits:
        lat = lateral_axis(tip_min, tip_max, shank_direction(slope))
        electrodes = shank_electrodes(shank_idx)
        df = place_electrodes(point, slope, probe_insert, lat, electrodes, cols)
        df = annotate_regions(df, atlas, cols)
        df = df[df["depth_um"] >= INSIDE_BRAIN_DEPTH_UM].reset_index(drop=True)
        out = sd / f"{animal}_np2_electrodes.csv"
        df.to_csv(out, index=False)
        written.append(out)
    return written
```

Note the one behaviour fix folded in: the tract-CSV glob now excludes `*_np2_electrodes.csv`, so re-running the reconstruction after a previous run cannot pick its own output as the tract. Delete the now-unused `import json`, `from natsort import natsorted`, `AXIS_TO_COORD_COL`, `_shank_index` and `_load_atlas`.

- [ ] **Step 3: Update the `_reconstruct` slot to use the imported loader**

In the same file, in `ElectrodesWidget._reconstruct`, change:

```python
            atlas = _load_atlas(p)
```

to:

```python
            atlas = load_atlas(p)
```

- [ ] **Step 4: Add a regression test for the self-output glob fix**

Append to `tests/test_electrodes_widget_smoke.py`:

```python
def test_rerun_ignores_previous_electrode_output(animal_dir):
    first = reconstruct_electrodes(animal_dir, _FakeAtlas())
    second = reconstruct_electrodes(animal_dir, _FakeAtlas())
    assert len(second) == len(first) == 2
    for a, b in zip(first, second):
        assert pd.read_csv(a).equals(pd.read_csv(b))
```

- [ ] **Step 5: Run the electrodes tests**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_electrodes_widget_smoke.py tests/test_np2_results.py -v
```

Expected: 14 passed.

- [ ] **Step 6: Run the whole suite to confirm nothing else regressed**

```bash
PYTHONPATH="$PWD/src" pytest tests/ -v
```

Expected: all passed.

- [ ] **Step 7: Commit**

```bash
git add src/napari_dmc_segload/_electrodes_widget.py tests/test_electrodes_widget_smoke.py
git commit -m "refactor: share results-path helpers via np2_results; ignore own output when globbing tracts"
```

---

## Task 3: `recover_line` accepts a distance column name

QC refits each shank's direction from the electrode CSV, whose distance column is `distance_to_tip_um`, not the tract CSV's `distance_to_tip(um)`. One parameter, no second copy of the fit.

**Files:**
- Modify: `src/napari_dmc_segload/np2_tract.py:16-37`
- Test: `tests/test_np2_tract.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_np2_tract.py`:

```python
def test_recover_line_accepts_alternate_distance_column():
    a = np.array([100.0, 200.0, 50.0])
    b = np.array([0.02, -0.10, 0.0])
    df = _synthetic_tract(a, b, insert=4000).rename(
        columns={"distance_to_tip(um)": "distance_to_tip_um"})
    point, slope = recover_line(df, COLS, dist_col="distance_to_tip_um")
    np.testing.assert_allclose(point, a, atol=1.0)
    np.testing.assert_allclose(slope, b, atol=0.01)


def test_recover_line_missing_alternate_distance_column_raises():
    df = _synthetic_tract(np.zeros(3), np.zeros(3), insert=1000)
    with pytest.raises(ValueError, match="distance_to_tip_um"):
        recover_line(df, COLS, dist_col="distance_to_tip_um")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_tract.py -v -k alternate
```

Expected: FAIL — `TypeError: recover_line() got an unexpected keyword argument 'dist_col'`.

- [ ] **Step 3: Add the parameter**

In `src/napari_dmc_segload/np2_tract.py`, change the signature and the two uses of `DIST_COL` inside `recover_line`:

```python
def recover_line(tract_df, coord_cols, dist_col=DIST_COL):
    """Least-squares fit of each atlas-coord column against the distance column.

    dmc writes coords = (probe_insert - dtt)/10 * direction + surface, i.e. a
    line in dtt. Returns (point, slope), both length-3 arrays in `coord_cols`
    order, where `point` is the coord at dtt=0 (the tip voxel) and `slope` is
    d(coord)/d(dtt). The many rows average out dmc's integer-voxel rounding.
    `dist_col` defaults to the dmc tract CSV's column; our electrode CSVs use
    `distance_to_tip_um`.
    """
    if dist_col not in tract_df.columns:
        raise ValueError(f"CSV missing column '{dist_col}'")
    if len(tract_df) < 2:
        raise ValueError("CSV needs >= 2 rows to fit a line")
    dtt = tract_df[dist_col].to_numpy(dtype=float)
```

The rest of the function body is unchanged.

- [ ] **Step 4: Run the tract tests**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_tract.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_tract.py tests/test_np2_tract.py
git commit -m "feat: let recover_line fit against a caller-chosen distance column"
```

---

## Task 4: shared test fixtures for synthetic electrode frames

Tasks 5–8 all need realistic electrode frames. Build them with the already-tested `shank_electrodes` + `place_electrodes` so the fixtures cannot drift from the real placement model.

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add the fixtures**

Append to `tests/conftest.py`:

```python
from napari_dmc_segload.np2_layout import shank_electrodes
from napari_dmc_segload.np2_results import ELECTRODE_CSV_SUFFIX
from napari_dmc_segload.np2_tract import place_electrodes

NP2_COORD_COLS = ["ap_coords", "dv_coords", "ml_coords"]
NP2_LATERAL = np.array([0.0, 0.0, 1.0])          # columns run along ml
NP2_TIP0 = np.array([600.0, 400.0, 500.0])       # shank 0 tip voxel
NP2_SLOPE = np.array([0.0, -0.1, 0.0])           # straight down DV, 1 voxel / 10 um
NP2_INSERT_UM = 4000.0
NP2_SHANK_PITCH_VOX = 25.0                       # 250 um at 10 um voxels


def _np2_acronym(depth_um):
    """Two stacked structures so a region ladder has more than one rung."""
    if depth_um >= 3000:
        return "ORBl"
    return "MOs"


def make_electrode_frame(shank, n_rows=100, tip=None, slope=None,
                         insert=NP2_INSERT_UM):
    """One shank's electrode DataFrame, shaped exactly like the widget's CSV."""
    tip = NP2_TIP0 + np.array([0.0, 0.0, NP2_SHANK_PITCH_VOX * shank]) if tip is None \
        else np.asarray(tip, float)
    slope = NP2_SLOPE if slope is None else np.asarray(slope, float)
    electrodes = [e for e in shank_electrodes(shank) if e["row"] < n_rows]
    df = place_electrodes(tip, slope, insert, NP2_LATERAL, electrodes, NP2_COORD_COLS)
    df["acronym"] = [_np2_acronym(d) for d in df["depth_um"]]
    df["structure_id"] = [723 if a == "ORBl" else 993 for a in df["acronym"]]
    df["name"] = ["Orbital area, lateral part" if a == "ORBl" else "Secondary motor area"
                  for a in df["acronym"]]
    return df


@pytest.fixture
def np2_frames():
    """Four well-formed shanks, 250 um apart, parallel, 200 electrodes each."""
    return {i: make_electrode_frame(i) for i in range(4)}


@pytest.fixture
def np2_animal_dir(tmp_path, np2_frames):
    """Animal folder holding those four shanks' electrode CSVs + params.json."""
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    for shank, df in np2_frames.items():
        sd = root / f"neuropixels_probe_{shank}"
        sd.mkdir(parents=True)
        df.to_csv(sd / f"animal01{ELECTRODE_CSV_SUFFIX}", index=False)
    (base / "params.json").write_text(
        json.dumps({"atlas_info": {"atlas": "allen_mouse_10um"}}))
    return base
```

`json`, `numpy as np` and `pytest` are already imported at the top of `conftest.py`, so no new stdlib imports are needed.

Test modules import these helpers with a plain `from conftest import ...`. That works because `tests/` has no `__init__.py`, so pytest's default prepend import mode puts the test directory on `sys.path`; a relative `from .conftest import ...` would fail.

- [ ] **Step 2: Verify the fixtures import cleanly**

```bash
PYTHONPATH="$PWD/src" pytest tests/ -v --collect-only -q
```

Expected: collection succeeds, no errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add synthetic NP2.0 electrode frame fixtures"
```

---

## Task 5: `np2_qc.py` — per-shank metrics and the region ladder

**Files:**
- Create: `src/napari_dmc_segload/np2_qc.py`
- Test: `tests/test_np2_qc.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_np2_qc.py`:

```python
import numpy as np
import pytest

from napari_dmc_segload.np2_qc import region_ladder, shank_metrics
from conftest import NP2_COORD_COLS as COLS
from conftest import make_electrode_frame


def test_shank_metrics_basic_geometry(np2_frames):
    m = shank_metrics(np2_frames[0], COLS)
    assert m["shank"] == 0
    assert m["n_electrodes"] == 200                      # 100 rows x 2 columns
    assert m["insertion_depth_um"] == pytest.approx(3825.0)   # 4000 - (0 + 175)
    deepest = np2_frames[0].loc[np2_frames[0]["distance_to_tip_um"].idxmin()]
    assert m["tip_voxel"] == tuple(int(deepest[c]) for c in COLS)
    assert m["angle_from_dv_deg"] == pytest.approx(0.0, abs=0.5)
    np.testing.assert_allclose(np.abs(m["direction"]), [0.0, 1.0, 0.0], atol=1e-2)


def test_shank_metrics_angle_for_tilted_shank():
    tilt = np.deg2rad(15.0)
    slope = 0.1 * np.array([np.sin(tilt), -np.cos(tilt), 0.0])
    m = shank_metrics(make_electrode_frame(0, slope=slope), COLS)
    assert m["angle_from_dv_deg"] == pytest.approx(15.0, abs=1.0)


def test_region_ladder_collapses_contiguous_runs(np2_frames):
    ladder = region_ladder(np2_frames[0])
    assert [r["acronym"] for r in ladder] == ["ORBl", "MOs"]
    assert ladder[0]["depth_from_um"] > ladder[0]["depth_to_um"]      # deep -> superficial
    assert ladder[0]["depth_to_um"] > ladder[1]["depth_from_um"]
    assert sum(r["n_electrodes"] for r in ladder) == len(np2_frames[0])


def test_region_ladder_single_region():
    df = make_electrode_frame(0, n_rows=10)              # all deep -> all ORBl
    ladder = region_ladder(df)
    assert [r["acronym"] for r in ladder] == ["ORBl"]
    assert ladder[0]["n_electrodes"] == 20


def test_region_ladder_empty_frame(np2_frames):
    assert region_ladder(np2_frames[0].iloc[0:0]) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_qc.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'napari_dmc_segload.np2_qc'`.

- [ ] **Step 3: Write the implementation**

Create `src/napari_dmc_segload/np2_qc.py`:

```python
"""Geometric QC for a reconstructed NP2.0 probe.

Pure numpy/pandas over the per-shank electrode DataFrames: no atlas, no napari,
no file IO. Everything is refitted from the electrode CSVs themselves, so QC
validates the files that were actually written rather than the tract they came
from.
"""

import numpy as np

from .np2_tract import recover_line, shank_direction

DIST_COL = "distance_to_tip_um"      # electrode CSV distance column
VOXEL_UM = 10.0                      # atlas voxel size assumed by the CSVs
DV_AXIS = 1                          # index of the 'si' (dorsoventral) axis
SHANK_PITCH_UM = 250.0               # NP2.0 4-shank inter-shank spacing

# Warning thresholds.
SPACING_TOL_UM = 50.0
ANGLE_TOL_DEG = 5.0
MIN_ELECTRODES = 50


def shank_metrics(df, coord_cols):
    """Geometry summary for one shank's electrode DataFrame.

    Keys: shank, n_electrodes, insertion_depth_um (deepest electrode still
    inside the brain), tip_voxel (coords of the smallest distance_to_tip_um),
    fit_tip (fitted intercept at distance_to_tip = 0, float voxels — the anchor
    used for inter-shank distances), direction (unit vector), angle_from_dv_deg,
    ladder.
    """
    point, slope = recover_line(df, coord_cols, dist_col=DIST_COL)
    direction = shank_direction(slope)
    tip_row = df.loc[df[DIST_COL].idxmin()]
    cos_dv = abs(float(direction[DV_AXIS]))
    return {
        "shank": int(df["shank"].iloc[0]),
        "n_electrodes": int(len(df)),
        "insertion_depth_um": float(df["depth_um"].max()),
        "tip_voxel": tuple(int(tip_row[c]) for c in coord_cols),
        "fit_tip": point,
        "direction": direction,
        "angle_from_dv_deg": float(np.degrees(np.arccos(np.clip(cos_dv, -1.0, 1.0)))),
        "ladder": region_ladder(df),
    }


def region_ladder(df):
    """Contiguous runs of acronym from the deepest electrode to the shallowest.

    Returns a list of {acronym, depth_from_um, depth_to_um, n_electrodes}.
    """
    if len(df) == 0:
        return []
    d = df.sort_values("depth_um", ascending=False)
    acronyms = d["acronym"].to_numpy()
    depths = d["depth_um"].to_numpy(dtype=float)
    runs = []
    start = 0
    for i in range(1, len(acronyms) + 1):
        if i == len(acronyms) or acronyms[i] != acronyms[start]:
            runs.append({
                "acronym": str(acronyms[start]),
                "depth_from_um": float(depths[start]),
                "depth_to_um": float(depths[i - 1]),
                "n_electrodes": int(i - start),
            })
            start = i
    return runs
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_qc.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_qc.py tests/test_np2_qc.py
git commit -m "feat: add np2_qc per-shank metrics and region ladder"
```

---

## Task 6: `np2_qc.py` — cross-shank metrics, warnings, report text

**Files:**
- Modify: `src/napari_dmc_segload/np2_qc.py`
- Test: `tests/test_np2_qc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_np2_qc.py`:

```python
from napari_dmc_segload.np2_qc import (
    build_report, cross_shank_metrics, format_report)
from conftest import NP2_SHANK_PITCH_VOX, NP2_TIP0


def test_cross_shank_metrics_recovers_pitch_and_parallelism(np2_frames):
    pairs = cross_shank_metrics(np2_frames, COLS)
    assert len(pairs) == 6                               # 4 shanks -> 6 pairs
    by_key = {(p["shank_a"], p["shank_b"]): p for p in pairs}
    assert by_key[(0, 1)]["distance_um"] == pytest.approx(250.0, abs=1.0)
    assert by_key[(0, 1)]["expected_um"] == pytest.approx(250.0)
    assert by_key[(0, 3)]["distance_um"] == pytest.approx(750.0, abs=1.0)
    assert by_key[(0, 3)]["expected_um"] == pytest.approx(750.0)
    assert all(p["angle_deg"] == pytest.approx(0.0, abs=0.5) for p in pairs)


def test_build_report_clean_input_has_no_warnings(np2_frames):
    report = build_report(np2_frames, COLS)
    assert len(report["shanks"]) == 4
    assert len(report["pairs"]) == 6
    assert report["warnings"] == []


def test_build_report_warns_on_bad_spacing(np2_frames):
    frames = dict(np2_frames)
    frames[1] = make_electrode_frame(
        1, tip=NP2_TIP0 + np.array([0.0, 0.0, 40.0]))     # 400 um, not 250
    warnings = build_report(frames, COLS)["warnings"]
    assert any("spacing" in w for w in warnings)
    assert any("shank 0-1" in w or "shanks 0-1" in w for w in warnings)


def test_build_report_warns_on_tilted_shank(np2_frames):
    tilt = np.deg2rad(15.0)
    frames = dict(np2_frames)
    frames[2] = make_electrode_frame(
        2, tip=NP2_TIP0 + np.array([0.0, 0.0, 2 * NP2_SHANK_PITCH_VOX]),
        slope=0.1 * np.array([np.sin(tilt), -np.cos(tilt), 0.0]))
    warnings = build_report(frames, COLS)["warnings"]
    assert any("angle" in w for w in warnings)


def test_build_report_warns_on_too_few_electrodes(np2_frames):
    frames = dict(np2_frames)
    frames[3] = make_electrode_frame(3, n_rows=10)        # 20 electrodes
    warnings = build_report(frames, COLS)["warnings"]
    assert any("electrodes inside the brain" in w for w in warnings)


def test_build_report_warns_when_all_root(np2_frames):
    frames = dict(np2_frames)
    rooted = make_electrode_frame(0).copy()
    rooted["acronym"] = "root"
    frames[0] = rooted
    warnings = build_report(frames, COLS)["warnings"]
    assert any("root" in w for w in warnings)


def test_format_report_is_readable_text(np2_frames):
    text = format_report(build_report(np2_frames, COLS), title="QC — animal01")
    assert "QC — animal01" in text
    assert "shank 0" in text
    assert "ORBl" in text
    assert "no warnings" in text
    assert text.endswith("\n")


def test_format_report_lists_warnings(np2_frames):
    frames = dict(np2_frames)
    frames[3] = make_electrode_frame(3, n_rows=10)
    text = format_report(build_report(frames, COLS))
    assert "warnings" in text
    assert "no warnings" not in text
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_qc.py -v
```

Expected: FAIL — `ImportError: cannot import name 'build_report'`.

- [ ] **Step 3: Write the implementation**

Append to `src/napari_dmc_segload/np2_qc.py`:

```python
def cross_shank_metrics(frames, coord_cols):
    """Pairwise geometry between shanks: tip distance vs the expected
    250 um x index span, and the angle between fitted directions.

    Returns a list of {shank_a, shank_b, distance_um, expected_um, angle_deg},
    one per unordered pair, in ascending shank order.
    """
    metrics = {s: shank_metrics(df, coord_cols) for s, df in frames.items()}
    shanks = sorted(metrics)
    pairs = []
    for i, a in enumerate(shanks):
        for b in shanks[i + 1:]:
            ma, mb = metrics[a], metrics[b]
            dist = float(np.linalg.norm(mb["fit_tip"] - ma["fit_tip"]) * VOXEL_UM)
            cos = abs(float(np.dot(ma["direction"], mb["direction"])))
            pairs.append({
                "shank_a": a,
                "shank_b": b,
                "distance_um": dist,
                "expected_um": SHANK_PITCH_UM * abs(b - a),
                "angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
            })
    return pairs


def build_report(frames, coord_cols):
    """Full QC report: {'shanks': [...], 'pairs': [...], 'warnings': [...]}."""
    shanks = [shank_metrics(frames[s], coord_cols) for s in sorted(frames)]
    pairs = cross_shank_metrics(frames, coord_cols)
    warnings = []

    for m in shanks:
        if m["n_electrodes"] < MIN_ELECTRODES:
            warnings.append(
                f"shank {m['shank']}: only {m['n_electrodes']} electrodes inside the "
                f"brain (expected >= {MIN_ELECTRODES})")
        if {r["acronym"] for r in m["ladder"]} <= {"root"}:
            warnings.append(
                f"shank {m['shank']}: every electrode is in 'root' — the tract may "
                "lie outside the annotated brain")

    for p in pairs:
        off = abs(p["distance_um"] - p["expected_um"])
        if off > SPACING_TOL_UM:
            warnings.append(
                f"shanks {p['shank_a']}-{p['shank_b']}: tip spacing {p['distance_um']:.0f} um, "
                f"expected {p['expected_um']:.0f} um (off by {off:.0f} um)")
        if p["angle_deg"] > ANGLE_TOL_DEG:
            warnings.append(
                f"shanks {p['shank_a']}-{p['shank_b']}: angle between shanks "
                f"{p['angle_deg']:.1f} deg, expected < {ANGLE_TOL_DEG:.0f} deg")

    return {"shanks": shanks, "pairs": pairs, "warnings": warnings}


def format_report(report, title="NP2.0 probe QC"):
    """Fixed-width text rendering of `build_report`'s output."""
    lines = [title, "=" * max(len(title), 60), ""]
    for m in report["shanks"]:
        lines.append(f"shank {m['shank']}")
        lines.append(f"  electrodes inside brain : {m['n_electrodes']}")
        lines.append(f"  insertion depth (um)    : {m['insertion_depth_um']:.0f}")
        lines.append(f"  tip voxel (a,b,c)       : {m['tip_voxel']}")
        lines.append(f"  angle from DV axis      : {m['angle_from_dv_deg']:.1f} deg")
        lines.append("  region ladder (deep -> superficial):")
        for r in m["ladder"]:
            lines.append(
                f"    {r['acronym']:<10} {r['depth_from_um']:>7.0f} -> "
                f"{r['depth_to_um']:>7.0f} um   n={r['n_electrodes']}")
        lines.append("")

    lines.append("shank pairs")
    for p in report["pairs"]:
        lines.append(
            f"  {p['shank_a']}-{p['shank_b']}  tip distance {p['distance_um']:>6.0f} um "
            f"(expected {p['expected_um']:>6.0f})  angle {p['angle_deg']:>5.1f} deg")
    lines.append("")

    if report["warnings"]:
        lines.append("warnings")
        lines.extend(f"  ! {w}" for w in report["warnings"])
    else:
        lines.append("no warnings")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_qc.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_qc.py tests/test_np2_qc.py
git commit -m "feat: add cross-shank QC metrics, warnings and report formatting"
```

---

## Task 7: `np2_scene.py` — points and colours

**Files:**
- Create: `src/napari_dmc_segload/np2_scene.py`
- Test: `tests/test_np2_scene.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_np2_scene.py`:

```python
import numpy as np
import pytest

from napari_dmc_segload.np2_scene import electrode_points, point_colors
from conftest import NP2_COORD_COLS as COLS
from conftest import make_electrode_frame


class _FakeMesh:
    def __init__(self, n_verts=12):
        self.points = np.random.default_rng(0).random((n_verts, 3)) * 1000.0
        cell = type("C", (), {"data": np.arange(n_verts).reshape(-1, 3)})()
        self.cells = [cell]


class _FakeAtlas:
    """Atlas-like: structures with rgb_triplet, meshes for some acronyms only."""

    resolution = (10.0, 10.0, 10.0)
    structures = {
        "ORBl": {"rgb_triplet": [36, 138, 94]},
        "MOs": {"rgb_triplet": [31, 157, 90]},
        "root": {"rgb_triplet": [255, 255, 255]},
    }
    meshless = {"MOs"}

    def mesh_from_structure(self, acronym):
        if acronym in self.meshless:
            raise KeyError(acronym)
        return _FakeMesh()


def test_electrode_points_stacks_shanks_in_ascending_order(np2_frames):
    points, features = electrode_points(np2_frames, COLS)
    assert points.shape == (800, 3)                       # 4 shanks x 200
    assert list(features.columns) == ["shank", "acronym", "electrode"]
    assert features["shank"].tolist()[:200] == [0] * 200
    assert features["shank"].tolist()[-200:] == [3] * 200
    # first row equals the first frame's first coord triple
    np.testing.assert_allclose(
        points[0], np2_frames[0].iloc[0][COLS].to_numpy(dtype=float))


def test_point_colors_by_shank_gives_one_colour_per_shank(np2_frames):
    _, features = electrode_points(np2_frames, COLS)
    colors = point_colors(features, "shank", None)
    assert colors.shape == (800, 4)
    assert len({tuple(c) for c in colors}) == 4


def test_point_colors_by_region_uses_atlas_rgb(np2_frames):
    _, features = electrode_points(np2_frames, COLS)
    colors = point_colors(features, "region", _FakeAtlas())
    orbl = features.index[features["acronym"] == "ORBl"][0]
    np.testing.assert_allclose(colors[orbl], [36 / 255, 138 / 255, 94 / 255, 1.0],
                               atol=1e-6)


def test_point_colors_by_region_unknown_acronym_is_grey(np2_frames):
    _, features = electrode_points(np2_frames, COLS)
    features.loc[0, "acronym"] = "NOT_A_REGION"
    colors = point_colors(features, "region", _FakeAtlas())
    np.testing.assert_allclose(colors[0], [0.5, 0.5, 0.5, 1.0])


def test_point_colors_by_electrode_index_normalises_across_shanks():
    # shank 0 spans electrodes 0..39, shank 1 only 0..19: with joint
    # normalisation shank 1 must not reach the top of the ramp.
    frames = {0: make_electrode_frame(0, n_rows=20), 1: make_electrode_frame(1, n_rows=10)}
    _, features = electrode_points(frames, COLS)
    colors = point_colors(features, "electrode index", None)
    top_overall = colors[features["electrode"].idxmax()]
    shank1 = features.index[features["shank"] == 1]
    assert not np.allclose(colors[shank1].max(axis=0), top_overall)
    assert colors.shape[1] == 4


def test_point_colors_rejects_unknown_mode(np2_frames):
    _, features = electrode_points(np2_frames, COLS)
    with pytest.raises(ValueError, match="colour mode"):
        point_colors(features, "banana", None)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_scene.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'napari_dmc_segload.np2_scene'`.

- [ ] **Step 3: Write the implementation**

Create `src/napari_dmc_segload/np2_scene.py`:

```python
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


def electrode_points(frames, coord_cols):
    """Stack every shank's electrodes into one point array + feature table.

    Shanks are concatenated in ascending index order, rows in file order.
    Returns (points (N,3) float voxel coords, features DataFrame with columns
    shank, acronym, electrode).
    """
    points, feats = [], []
    for shank in sorted(frames):
        df = frames[shank]
        points.append(df[coord_cols].to_numpy(dtype=float))
        feats.append(pd.DataFrame({
            "shank": df["shank"].to_numpy(dtype=int),
            "acronym": df["acronym"].astype(str).to_numpy(),
            "electrode": df["electrode"].to_numpy(dtype=int),
        }))
    return (np.concatenate(points, axis=0),
            pd.concat(feats, ignore_index=True))


def point_colors(features, mode, atlas):
    """RGBA (N,4) for each electrode under the given colour mode.

    `atlas` is only used by MODE_REGION and may be None otherwise.
    """
    if mode == MODE_SHANK:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_scene.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_scene.py tests/test_np2_scene.py
git commit -m "feat: add np2_scene electrode points and colour modes"
```

---

## Task 8: `np2_scene.py` — atlas meshes

**Files:**
- Modify: `src/napari_dmc_segload/np2_scene.py`
- Test: `tests/test_np2_scene.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_np2_scene.py`:

```python
from napari_dmc_segload.np2_scene import region_meshes, root_mesh


def test_root_mesh_returns_vertices_and_faces():
    verts, faces = root_mesh(_FakeAtlas())
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3


def test_region_meshes_skips_structures_without_a_mesh(np2_frames):
    meshes, skipped = region_meshes(np2_frames, _FakeAtlas())
    assert [m["acronym"] for m in meshes] == ["ORBl"]     # MOs is meshless in the fake
    assert skipped == ["MOs"]
    assert meshes[0]["vertices"].shape[1] == 3
    assert meshes[0]["faces"].shape[1] == 3
    np.testing.assert_allclose(meshes[0]["color"][:3],
                               [36 / 255, 138 / 255, 94 / 255], atol=1e-6)


def test_region_meshes_ignores_root_and_rare_structures():
    df = make_electrode_frame(0).copy()
    df.loc[df.index[:4], "acronym"] = "ORBm"             # 4 electrodes only
    df.loc[df.index[4:], "acronym"] = "root"
    meshes, skipped = region_meshes({0: df}, _FakeAtlas(), min_electrodes=10)
    assert meshes == []                                  # ORBm too rare, root excluded
    assert skipped == []


def test_region_meshes_caps_the_number_of_regions(np2_frames):
    meshes, _ = region_meshes(np2_frames, _FakeAtlas(), min_electrodes=1, max_regions=1)
    assert len(meshes) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_scene.py -v -k mesh
```

Expected: FAIL — `ImportError: cannot import name 'region_meshes'`.

- [ ] **Step 3: Write the implementation**

Append to `src/napari_dmc_segload/np2_scene.py`:

```python
MIN_MESH_ELECTRODES = 10     # ignore structures a shank merely clips
MAX_MESH_REGIONS = 12        # keep the scene readable
ROOT_OPACITY = 0.08
REGION_OPACITY = 0.25
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_np2_scene.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_scene.py tests/test_np2_scene.py
git commit -m "feat: add atlas mesh selection to np2_scene"
```

---

## Task 9: `_probe_vis_widget.py` — the dock widget

**Files:**
- Create: `src/napari_dmc_segload/_probe_vis_widget.py`
- Test: `tests/test_probe_vis_widget_smoke.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_probe_vis_widget_smoke.py`:

```python
import numpy as np
import pytest

from napari_dmc_segload import _probe_vis_widget as pv
from napari_dmc_segload._probe_vis_widget import ProbeVisWidget


class _FakeMesh:
    def __init__(self, n_verts=12):
        self.points = np.random.default_rng(1).random((n_verts, 3)) * 1000.0
        cell = type("C", (), {"data": np.arange(n_verts).reshape(-1, 3)})()
        self.cells = [cell]


class _FakeSpace:
    axes_description = ("ap", "si", "rl")


class _FakeAtlas:
    resolution = (10.0, 10.0, 10.0)
    space = _FakeSpace()
    structures = {
        "ORBl": {"rgb_triplet": [36, 138, 94]},
        "MOs": {"rgb_triplet": [31, 157, 90]},
        "root": {"rgb_triplet": [255, 255, 255]},
    }

    def mesh_from_structure(self, acronym):
        return _FakeMesh()


@pytest.fixture
def widget(make_napari_viewer, monkeypatch):
    monkeypatch.setattr(pv, "load_atlas", lambda p: _FakeAtlas())
    viewer = make_napari_viewer()
    return ProbeVisWidget(viewer), viewer


def test_visualize_adds_points_and_surfaces(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    names = [ly.name for ly in viewer.layers]
    assert ProbeVisWidget.POINTS_LAYER in names
    assert ProbeVisWidget.ROOT_LAYER in names
    assert any(n.startswith(ProbeVisWidget.REGION_PREFIX) for n in names)
    pts = viewer.layers[ProbeVisWidget.POINTS_LAYER]
    assert len(pts.data) == 800
    np.testing.assert_allclose(pts.scale, [10.0, 10.0, 10.0])
    assert viewer.dims.ndisplay == 3


def test_visualize_fills_panel_and_writes_qc_file(widget, np2_animal_dir):
    w, _ = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    text = w.report.toPlainText()
    assert "shank 0" in text and "ORBl" in text
    out = np2_animal_dir / "results" / "neuropixels_probe" / "animal01_np2_qc.txt"
    assert out.exists()
    assert out.read_text() == text


def test_colour_mode_change_recolours_without_reloading(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    pts = viewer.layers[ProbeVisWidget.POINTS_LAYER]
    before = np.array(pts.face_color, copy=True)
    w.form.colour_by.value = "electrode index"
    after = np.array(viewer.layers[ProbeVisWidget.POINTS_LAYER].face_color)
    assert after.shape == before.shape
    assert not np.allclose(before, after)
    assert len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data) == 800


def test_second_visualize_does_not_duplicate_layers(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    n_first = len(viewer.layers)
    w._visualize()
    assert len(viewer.layers) == n_first


def test_missing_electrode_csvs_aborts_cleanly(widget, tmp_path):
    w, viewer = widget
    base = tmp_path / "animal02"
    (base / "results" / "neuropixels_probe" / "neuropixels_probe_0").mkdir(parents=True)
    w.form.input_path.value = str(base)
    w._visualize()                       # must not raise
    assert len(viewer.layers) == 0
    assert w.report.toPlainText() == ""


def test_invalid_path_aborts_cleanly(widget, tmp_path):
    w, viewer = widget
    w.form.input_path.value = str(tmp_path / "does_not_exist")
    w._visualize()                       # must not raise
    assert len(viewer.layers) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_probe_vis_widget_smoke.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'napari_dmc_segload._probe_vis_widget'`.

- [ ] **Step 3: Write the implementation**

Create `src/napari_dmc_segload/_probe_vis_widget.py`:

```python
"""Dock widget: show a reconstructed NP2.0 probe in 3D and report its QC.

Read-only with respect to the reconstruction — it loads the per-shank
`*_np2_electrodes.csv` written by the "NP2.0 electrodes" widget, renders them
against atlas meshes, and writes one QC text file.
"""

from pathlib import Path

from napari import Viewer
from napari.utils.colormaps import Colormap
from napari.utils.notifications import show_info
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit

from .np2_qc import build_report, format_report
from .np2_results import coord_cols, load_atlas, load_electrode_csvs, probe_results_dir
from .np2_scene import (
    COLOUR_MODES, REGION_OPACITY, ROOT_COLOR, ROOT_OPACITY, electrode_points,
    point_colors, region_meshes, root_mesh)

POINT_SIZE_VOX = 2.0     # 2 voxels = 20 um across, visible without hiding anatomy


def _build_form() -> FunctionGui:
    @magicgui(
        layout="vertical",
        input_path=dict(widget_type="FileEdit", mode="d",
                        label="input path (animal_id): "),
        colour_by=dict(widget_type="ComboBox", label="colour by",
                       choices=list(COLOUR_MODES), value=COLOUR_MODES[0]),
        call_button=False,
    )
    def form(input_path, colour_by):
        pass
    return form


def _solid_colormap(rgba):
    """A napari Colormap that paints a surface one flat colour."""
    colors = [list(rgba[:3]) + [1.0]] * 2
    return Colormap(colors, name=f"solid_{rgba[0]:.3f}_{rgba[1]:.3f}_{rgba[2]:.3f}")


class ProbeVisWidget(QWidget):
    """Dock widget: 3D view + QC of a reconstructed NP2.0 probe."""

    POINTS_LAYER = "np2 electrodes"
    ROOT_LAYER = "np2 root"
    REGION_PREFIX = "np2 region: "

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        self.form = _build_form()
        self.layout().addWidget(self.form.native)

        self.btn = QPushButton("Visualize probe")
        self.btn.clicked.connect(self._visualize)
        self.layout().addWidget(self.btn)

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setLineWrapMode(QTextEdit.NoWrap)
        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        self.report.setFont(font)
        self.layout().addWidget(self.report)

        # loaded state, so recolouring never re-reads the CSVs
        self._features = None
        self._atlas = None

        self.form.colour_by.changed.connect(self._recolour)

    # --- helpers -----------------------------------------------------------
    def _clear_layers(self):
        """Remove only the layers this widget added."""
        for layer in list(self.viewer.layers):
            if (layer.name in (self.POINTS_LAYER, self.ROOT_LAYER)
                    or layer.name.startswith(self.REGION_PREFIX)):
                self.viewer.layers.remove(layer)

    # --- actions -----------------------------------------------------------
    def _visualize(self):
        p = self.form.input_path.value
        if not p or not Path(p).is_dir():
            show_info("Select a valid input path (animal_id) directory.")
            return
        input_path = Path(p)
        try:
            atlas = load_atlas(input_path)
            frames = load_electrode_csvs(input_path)
        except (FileNotFoundError, ValueError, KeyError) as e:
            show_info(str(e))
            return

        cols = coord_cols(atlas)
        points, features = electrode_points(frames, cols)
        self._atlas = atlas
        self._features = features

        self._clear_layers()

        verts, faces = root_mesh(atlas)
        self.viewer.add_surface((verts, faces), name=self.ROOT_LAYER,
                                colormap=_solid_colormap(ROOT_COLOR),
                                opacity=ROOT_OPACITY, shading="smooth")

        meshes, skipped = region_meshes(frames, atlas)
        for m in meshes:
            self.viewer.add_surface(
                (m["vertices"], m["faces"]), name=self.REGION_PREFIX + m["acronym"],
                colormap=_solid_colormap(m["color"]), opacity=REGION_OPACITY,
                shading="smooth")

        self.viewer.add_points(
            points, name=self.POINTS_LAYER, size=POINT_SIZE_VOX,
            scale=tuple(atlas.resolution), features=features,
            face_color=point_colors(features, self.form.colour_by.value, atlas))
        self.viewer.dims.ndisplay = 3

        text = format_report(build_report(frames, cols),
                             title=f"NP2.0 probe QC — {input_path.name}")
        if skipped:
            text += "note: this atlas has no mesh for: " + ", ".join(skipped) + "\n"
        self.report.setPlainText(text)

        out = probe_results_dir(input_path) / f"{input_path.name}_np2_qc.txt"
        out.write_text(text)
        show_info(f"Visualized {len(frames)} shank(s); QC written to {out.name}")

    def _recolour(self, *_):
        if self._features is None or self.POINTS_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[self.POINTS_LAYER]
        layer.face_color = point_colors(self._features, self.form.colour_by.value,
                                        self._atlas)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH="$PWD/src" pytest tests/test_probe_vis_widget_smoke.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/_probe_vis_widget.py tests/test_probe_vis_widget_smoke.py
git commit -m "feat: add NP2.0 probe visualization dock widget"
```

---

## Task 10: register the widget and document it

**Files:**
- Modify: `src/napari_dmc_segload/napari.yaml`
- Modify: `README.md`

- [ ] **Step 1: Register the widget**

In `src/napari_dmc_segload/napari.yaml`, add the command under `commands:` (after the `ElectrodesWidget` entry):

```yaml
    - id: napari-dmc-segload.ProbeVisWidget
      python_name: napari_dmc_segload._probe_vis_widget:ProbeVisWidget
      title: NP2.0 probe visualization
```

and the widget under `widgets:` (after the `NP2.0 electrodes` entry):

```yaml
    - command: napari-dmc-segload.ProbeVisWidget
      display_name: NP2.0 probe visualization
```

- [ ] **Step 2: Verify the manifest parses and the widget class is importable**

```bash
PYTHONPATH="$PWD/src" python -c "
import yaml
m = yaml.safe_load(open('src/napari_dmc_segload/napari.yaml'))
ids = [c['id'] for c in m['contributions']['commands']]
assert 'napari-dmc-segload.ProbeVisWidget' in ids, ids
import napari_dmc_segload._probe_vis_widget as w
assert hasattr(w, 'ProbeVisWidget')
print('manifest ok:', ids)
"
```

Expected: `manifest ok: [...]` listing four command ids.

- [ ] **Step 3: Document the widget in the README**

Append to `README.md`:

```markdown
## NP2.0 probe visualization

After running **NP2.0 electrodes**, open `Plugins → DMC Segmentation Loader →
NP2.0 probe visualization` to check the reconstruction.

1. Set **input path** to the same animal folder.
2. Click **Visualize probe**. The viewer switches to 3D and gains:
   - `np2 root` — the translucent brain outline,
   - `np2 region: <acronym>` — one surface per structure the electrodes pass
     through (the largest 12, ignoring structures with fewer than 10
     electrodes),
   - `np2 electrodes` — every reconstructed electrode from every shank.
3. Use **colour by** to switch between:
   - `shank` — one colour per shank,
   - `region` — the atlas colour of each electrode's structure,
   - `electrode index` — a viridis ramp over the physical electrode index,
     normalised across all shanks. A correctly placed shank shows a smooth
     tip→top gradient; a reversed gradient means the fit ran the wrong way.

The QC panel below the button reports, per shank, the electrode count inside the
brain, insertion depth, tip voxel, angle from the DV axis and the region ladder;
then every shank pair's tip distance against the expected 250 µm × index span
and the angle between them. Anything outside tolerance is listed under
`warnings`. The same text is written to
`results/neuropixels_probe/<animal>_np2_qc.txt`.

This widget only reads the electrode CSVs — it never recomputes the
reconstruction.
```

- [ ] **Step 4: Run the full suite**

```bash
PYTHONPATH="$PWD/src" pytest tests/ -v
```

Expected: all tests pass, including the pre-existing ones.

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/napari.yaml README.md
git commit -m "feat: register NP2.0 probe visualization widget; document it"
```

---

## Task 11: verify against the real animal folder

Synthetic fixtures cannot catch a wrong atlas axis assumption or a mesh that lands in the wrong place. This task is manual and needs the user's real reconstructed animal.

**Files:** none (verification only)

- [ ] **Step 1: Confirm the reconstruction exists**

Ask the user for the animal folder path, then:

```bash
ls "<animal_folder>/results/neuropixels_probe"/*/*_np2_electrodes.csv
```

Expected: one CSV per traced shank. If absent, the "NP2.0 electrodes" widget has to be run first.

- [ ] **Step 2: Check the QC numbers headlessly before touching the GUI**

```bash
PYTHONPATH="$PWD/src" python -c "
import sys
from napari_dmc_segload.np2_results import coord_cols, load_atlas, load_electrode_csvs
from napari_dmc_segload.np2_qc import build_report, format_report
p = sys.argv[1]
atlas = load_atlas(p)
frames = load_electrode_csvs(p)
print(format_report(build_report(frames, coord_cols(atlas)), title='NP2.0 probe QC'))
" "<animal_folder>"
```

Expected: a report naming plausible structures, inter-shank distances near 250 µm × span, and pairwise angles near 0°. Report the warnings section to the user verbatim — real warnings here are a finding about the reconstruction, not a bug in this widget.

- [ ] **Step 3: Launch napari and inspect the scene**

```bash
python -m napari
```

Open `Plugins → DMC Segmentation Loader → NP2.0 probe visualization`, set the input path, click **Visualize probe**. Confirm with the user:
- the four shanks sit inside the brain outline, parallel, ~250 µm apart,
- the region surfaces enclose the points they are coloured to match,
- `electrode index` shows a smooth tip→top gradient on every shank.

- [ ] **Step 4: Report findings**

Summarise for the user what the scene and the QC text showed. If the reconstruction itself looks wrong, that is a result — say so plainly rather than adjusting tolerances to hide it.

---

## Completion

After Task 11, use `superpowers:finishing-a-development-branch` to decide how to integrate `feat/probe-vis`.
