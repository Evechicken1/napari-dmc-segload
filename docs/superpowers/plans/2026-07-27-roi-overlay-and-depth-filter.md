# ROI overlay + global max-depth filter — Implementation Plan

> **For agentic workers:** implement task-by-task with TDD (test first, watch it fail, then implement). Steps use checkbox syntax.

**Goal:** Extend the "NP2.0 probe visualization" widget with (A) an add-region dropdown that renders any atlas structure's mesh as a highlighted ROI overlay (so a target nucleus like LDT can be shown even when the probe misses it), and (B) a global max-depth filter that live-hides electrodes deeper than a chosen depth.

**Architecture:** Pure additions to `np2_scene.py` (`atlas_acronyms`, `roi_mesh`, a `max_depth` arg on `electrode_points`, an empty-input guard on `point_colors`); the Qt widget `_probe_vis_widget.py` gains four controls and routes both colour and depth through one `_render_points()`. No change to `np2_qc` or `np2_results`. The depth filter is display-only — region meshes and QC stay computed on the full reconstruction.

**Tech stack:** Python 3.10, napari 0.7.0, magicgui, qtpy, numpy/pandas, `bg_atlasapi`, pytest + pytest-qt.

## Environment

```bash
cd /c/Users/xaand/Documents/PhD/Analysis/Informatics/Python/napari-dmc-segload-roi-depth
source ~/anaconda3/etc/profile.d/conda.sh && conda activate napari-env
PYTHONPATH="$PWD/src" python -m pytest tests/ -q     # baseline: 114 passed
```

Do NOT set `QT_QPA_PLATFORM=offscreen` (GL context needed). Branch `feat/probe-vis-roi-depth`. Commit per task.

## Verified facts

- `atlas.structures` is bg_atlasapi's `StructuresDict`: iterating `.values()` yields dicts each carrying an `'acronym'` key and `'rgb_triplet'`; `atlas.structures[acronym]` also works (acronym lookup). `allen_mouse_10um` has ~1327 structures including `LDT` ("Laterodorsal tegmental nucleus"), and `mesh_from_structure("LDT")` returns a 323-vertex mesh.
- `mesh_from_structure` raises for an unknown/mesh-less structure.
- napari `Points` accepts an empty `(0,3)` array; a layer's `.data`, `.features`, `.face_color` can be reassigned after creation (set `.data` before `.face_color` so lengths agree).

---

## Task 1: `np2_scene` — ROI mesh, atlas acronym list, depth filter, empty guard

**Files:** Modify `src/napari_dmc_segload/np2_scene.py`; Test `tests/test_np2_scene.py`

- [ ] **Step 1 — failing tests.** Append to `tests/test_np2_scene.py`. The existing `_FakeAtlas` in that file has `structures` keyed by acronym; extend/369 add a fake whose structure *values* carry an `"acronym"` key so `atlas_acronyms` is exercised faithfully, and whose `mesh_from_structure` raises for a designated meshless acronym.

```python
from napari_dmc_segload.np2_scene import atlas_acronyms, roi_mesh, ROI_OPACITY
from conftest import make_electrode_frame
from conftest import NP2_COORD_COLS as COLS


class _RoiAtlas:
    resolution = (10.0, 10.0, 10.0)
    structures = {
        "ORBl": {"acronym": "ORBl", "rgb_triplet": [36, 138, 94]},
        "MOs": {"acronym": "MOs", "rgb_triplet": [31, 157, 90]},
        "LDT": {"acronym": "LDT", "rgb_triplet": [255, 0, 0]},
        "root": {"acronym": "root", "rgb_triplet": [255, 255, 255]},
    }
    meshless = {"MOs"}

    def mesh_from_structure(self, acronym):
        if acronym not in self.structures or acronym in self.meshless:
            raise KeyError(acronym)
        pts = __import__("numpy").arange(12, dtype=float).reshape(4, 3)
        cell = type("C", (), {"data": __import__("numpy").arange(12).reshape(-1, 3)})()
        return type("M", (), {"points": pts, "cells": [cell]})()


def test_atlas_acronyms_sorted_and_complete():
    acs = atlas_acronyms(_RoiAtlas())
    assert acs == sorted(acs)
    assert set(acs) == {"ORBl", "MOs", "LDT", "root"}


def test_roi_mesh_returns_mesh_dict_for_meshed_structure():
    m = roi_mesh(_RoiAtlas(), "LDT")
    assert m["acronym"] == "LDT"
    assert m["vertices"].shape[1] == 3 and m["faces"].shape[1] == 3
    import numpy as np
    np.testing.assert_allclose(m["color"][:3], [1.0, 0.0, 0.0])


def test_roi_mesh_missing_mesh_raises_valueerror():
    import pytest
    with pytest.raises(ValueError, match="MOs"):
        roi_mesh(_RoiAtlas(), "MOs")
    with pytest.raises(ValueError, match="NOPE"):
        roi_mesh(_RoiAtlas(), "NOPE")


def test_electrode_points_max_depth_filters_inclusive():
    frames = {0: make_electrode_frame(0, n_rows=100)}          # depth_um 175..? decreasing with z
    full, _ = electrode_points(frames, COLS)
    dmax = float(frames[0]["depth_um"].median())
    pts, feats = electrode_points(frames, COLS, max_depth=dmax)
    kept = (frames[0]["depth_um"] <= dmax).sum()
    assert len(pts) == kept == len(feats)
    assert len(pts) < len(full)


def test_electrode_points_max_depth_none_is_unchanged():
    frames = {0: make_electrode_frame(0)}
    a, _ = electrode_points(frames, COLS)
    b, _ = electrode_points(frames, COLS, max_depth=None)
    import numpy as np
    np.testing.assert_array_equal(a, b)


def test_electrode_points_max_depth_below_all_gives_empty():
    frames = {0: make_electrode_frame(0)}
    pts, feats = electrode_points(frames, COLS, max_depth=-1e9)
    assert pts.shape == (0, 3) and len(feats) == 0


def test_point_colors_empty_features_all_modes():
    import numpy as np
    _, feats = electrode_points({0: make_electrode_frame(0)}, COLS, max_depth=-1e9)
    for mode in ("shank", "region", "electrode index"):
        c = point_colors(feats, mode, _RoiAtlas())
        assert c.shape == (0, 4)
```

- [ ] **Step 2 — run, see them fail.** `PYTHONPATH="$PWD/src" python -m pytest tests/test_np2_scene.py -q` → import/attribute errors.

- [ ] **Step 3 — implement.** In `np2_scene.py`:

Add `ROI_OPACITY = 0.5` next to `REGION_OPACITY`.

Add the empty guard as the first line of `point_colors`:
```python
    if len(features) == 0:
        return np.zeros((0, 4))
```

Add a `max_depth=None` parameter to `electrode_points`, filtering each frame before stacking:
```python
def electrode_points(frames, coord_cols, max_depth=None):
    """... (existing docstring) ...

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
```
(Note: `max_depth <= depth_um` keeps deep electrodes; the filter drops the *shallow* excess. "max depth" here means "don't show anything deeper than X" per the spec — so keep `depth_um <= max_depth`. Confirm the test's `kept` count matches this direction; if the fixture's depths make the assertion trivially true, the median split still gives a strict subset, which is enough.)

Add the two new functions near the mesh helpers:
```python
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
```

- [ ] **Step 4 — run, all green.** `PYTHONPATH="$PWD/src" python -m pytest tests/test_np2_scene.py -q`.

- [ ] **Step 5 — real-atlas sanity (report, don't commit as a test).** Run once and paste the result:
```bash
PYTHONPATH="$PWD/src" python -c "
from bg_atlasapi import BrainGlobeAtlas
from napari_dmc_segload.np2_scene import atlas_acronyms, roi_mesh
a=BrainGlobeAtlas('allen_mouse_10um')
acs=atlas_acronyms(a); print(len(acs),'acronyms; LDT in list:','LDT' in acs)
m=roi_mesh(a,'LDT'); print('LDT roi mesh',m['vertices'].shape, m['color'].tolist())
"
```

- [ ] **Step 6 — commit.** `git add -A && git commit -m "feat: np2_scene ROI mesh, atlas acronym list, and max-depth point filter"`

---

## Task 2: widget — add-region dropdown, ROI layers, live depth filter

**Files:** Modify `src/napari_dmc_segload/_probe_vis_widget.py`; Test `tests/test_probe_vis_widget_smoke.py`

- [ ] **Step 1 — failing tests.** Append to `tests/test_probe_vis_widget_smoke.py`. Extend the existing `_FakeAtlas` used there so its `structures` values carry an `"acronym"` key (needed by `atlas_acronyms`) and it can mesh `LDT` plus a designated meshless acronym; keep the existing tests green. Then:

```python
def test_visualize_populates_region_dropdown(widget, np2_animal_dir):
    w, _ = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    choices = list(w.form.add_region.choices)
    assert len(choices) > 0
    assert "LDT" in choices


def test_add_region_adds_highlighted_roi_layer(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.add_region.value = "LDT"
    w._add_region()
    name = ProbeVisWidget.ROI_PREFIX + "LDT"
    assert name in [ly.name for ly in viewer.layers]
    assert viewer.layers[name].opacity == pytest.approx(ROI_OPACITY_EXPECTED)


def test_add_region_before_visualize_aborts_cleanly(widget):
    w, viewer = widget
    w._add_region()                      # must not raise
    assert not any(ly.name.startswith(ProbeVisWidget.ROI_PREFIX) for ly in viewer.layers)


def test_add_region_meshless_reports_and_adds_no_layer(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.add_region.value = MESHLESS_ACRONYM     # defined on the fake atlas
    w._add_region()                                 # must not raise
    assert not any(ly.name.startswith(ProbeVisWidget.ROI_PREFIX) for ly in viewer.layers)


def test_second_visualize_clears_roi_layers(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.add_region.value = "LDT"
    w._add_region()
    w._visualize()
    assert not any(ly.name.startswith(ProbeVisWidget.ROI_PREFIX) for ly in viewer.layers)


def test_depth_filter_hides_deep_electrodes_and_restores(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    full = len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data)
    n_region = sum(ly.name.startswith(ProbeVisWidget.REGION_PREFIX) for ly in viewer.layers)
    qc_before = w.report.toPlainText()
    # enable a mid-range max depth
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = 1500.0
    filtered = len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data)
    assert 0 < filtered < full
    # region meshes and QC untouched by a display filter
    assert sum(ly.name.startswith(ProbeVisWidget.REGION_PREFIX) for ly in viewer.layers) == n_region
    assert w.report.toPlainText() == qc_before
    # disable -> restored
    w.form.limit_depth.value = False
    assert len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data) == full


def test_depth_filter_below_all_leaves_empty_points_no_crash(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = -1.0
    assert len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data) == 0


def test_colour_switch_still_works_with_depth_active(widget, np2_animal_dir):
    w, viewer = widget
    w.form.input_path.value = str(np2_animal_dir)
    w._visualize()
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = 1500.0
    pts = viewer.layers[ProbeVisWidget.POINTS_LAYER]
    before = np.array(pts.face_color, copy=True)
    w.form.colour_by.value = "electrode index"
    after = np.array(viewer.layers[ProbeVisWidget.POINTS_LAYER].face_color)
    assert after.shape == before.shape and not np.allclose(before, after)
```

Add the imports/constants these tests reference at the top of the test file: `from napari_dmc_segload.np2_scene import ROI_OPACITY as ROI_OPACITY_EXPECTED`, and a `MESHLESS_ACRONYM = "..."` matching the fake atlas.

- [ ] **Step 2 — run, see them fail.**

- [ ] **Step 3 — implement.** In `_probe_vis_widget.py`:

1. Extend imports from `.np2_scene` to add `ROI_OPACITY, atlas_acronyms, roi_mesh`.
2. In `_build_form`, add controls (initial empty choices for the combo):
```python
        add_region=dict(widget_type="ComboBox", label="add region", choices=[]),
        limit_depth=dict(widget_type="CheckBox", label="limit depth", value=False),
        max_depth_um=dict(widget_type="FloatSpinBox", label="max depth (um)",
                          value=2000.0, min=-1.0, max=15000.0, step=50.0),
```
and add them to the inner `def form(...)` signature.
3. Add class constant `ROI_PREFIX = "np2 ROI: "`. Include it in `_clear_layers` (remove layers whose name starts with it).
4. Add a **Add region** `QPushButton` under the Visualize button, connected to `self._add_region`.
5. Make the combo searchable — after `self.form = _build_form()`:
```python
        combo = self.form.add_region.native      # QComboBox
        combo.setEditable(True)
        combo.setInsertPolicy(combo.NoInsert)
        combo.completer().setCompletionMode(QCompleter.PopupCompletion)
        combo.completer().setFilterMode(Qt.MatchContains)
```
(import `QCompleter` from `qtpy.QtWidgets`, `Qt` from `qtpy.QtCore`.)
6. Store `self._frames = None` and `self._cols = None` in `__init__`. Connect `limit_depth.changed` and `max_depth_um.changed` to `self._render_points`; keep `colour_by.changed` connected but point it at `self._render_points` too (drop `_recolour`, or make `_recolour` delegate).
7. In `_visualize`, after `atlas`/`frames` load and meshes build succeed: store `self._frames = frames`, `self._cols = cols`, `self._atlas = atlas`; set `self.form.add_region.choices = atlas_acronyms(atlas)`. Replace the direct `add_points(...)` call with `self._render_points()`. Region-mesh and QC code unchanged.
8. Add `_render_points`:
```python
    def _render_points(self, *_):
        if self._frames is None:
            return
        max_depth = self.form.max_depth_um.value if self.form.limit_depth.value else None
        points, features = electrode_points(self._frames, self._cols, max_depth)
        self._features = features
        colors = point_colors(features, self.form.colour_by.value, self._atlas)
        if self.POINTS_LAYER in self.viewer.layers:
            layer = self.viewer.layers[self.POINTS_LAYER]
            layer.data = points          # set data before face_color so lengths agree
            layer.features = features
            layer.face_color = colors
        else:
            self.viewer.add_points(
                points, name=self.POINTS_LAYER, size=POINT_SIZE_VOX,
                scale=tuple(self._atlas.resolution), features=features,
                face_color=colors)
            self.viewer.dims.ndisplay = 3
```
9. Add `_add_region`:
```python
    def _add_region(self):
        if self._atlas is None:
            show_info("Visualize a probe first.")
            return
        acronym = self.form.add_region.native.currentText().strip()
        if not acronym:
            return
        try:
            m = roi_mesh(self._atlas, acronym)
        except ValueError as e:
            show_info(str(e))
            return
        name = self.ROI_PREFIX + acronym
        if name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[name])
        self.viewer.add_surface(
            (m["vertices"], m["faces"]), name=name,
            colormap=_solid_colormap(m["color"]), opacity=ROI_OPACITY,
            shading="smooth")
```

- [ ] **Step 4 — run widget tests, then full suite.** `PYTHONPATH="$PWD/src" python -m pytest tests/test_probe_vis_widget_smoke.py -q` then `tests/ -q`.

- [ ] **Step 5 — README.** Add a short paragraph under the NP2.0 probe visualization section: the **add region** dropdown overlays any atlas structure (e.g. `LDT`) as a highlighted mesh even if the probe misses it; **limit depth** + **max depth (um)** hide electrodes deeper than the value (display only — QC unaffected).

- [ ] **Step 6 — commit.** `git add -A && git commit -m "feat: add-region ROI overlay and live global max-depth filter to probe vis widget"`

---

## Task 3: verify on XT027

- [ ] Headless: load XT027, add the `LDT` ROI mesh, confirm it builds; confirm `electrode_points(..., max_depth=1500)` reduces the point count. Report whether the LDT mesh sits away from the electrode cloud (expected — XT027 has zero electrodes in LDT).
- [ ] Optionally relaunch napari for the user.

## Notes / risks

- Editable-combobox `.value` may not reflect free-typed text under magicgui; `_add_region` reads `native.currentText()` and validates via `roi_mesh`, sidestepping that.
- Depth filter is display-only by design; QC and region meshes intentionally stay on the full reconstruction.
- An empty points layer (max depth below all electrodes) is valid; `point_colors` guards the empty case so index-mode `min()` can't blow up.
