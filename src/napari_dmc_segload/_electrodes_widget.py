"""Dock widget: reconstruct NP2.0 electrode locations from dmc's per-shank
registered probe tract, after the animal has been registered in dmc.

Trace shanks as dmc `neuropixels_probe` (4 shanks), run dmc registration +
Results, then run this. Writes one CSV per shank with each electrode's atlas
coordinates and region.
"""

import json
import re
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

    # (shank_dir, shank_idx, point, slope, probe_insert). The shank index is
    # parsed from the folder name (e.g. "neuropixels_probe_2" -> 2), NOT the
    # sort position, so a missing/untraced shank never re-labels the others.
    fits = []
    for sd in shank_dirs:
        csvs = list(sd.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"no tract CSV in {sd}")
        df = load_tract_csv(csvs[0])
        point, slope = recover_line(df, coord_cols)
        fits.append((sd, _shank_index(sd.name), point, slope, recover_probe_insert(df)))

    # Lateral axis spans the two extreme traced shanks (first/last by name order).
    tip_min = fits[0][2]
    tip_max = fits[-1][2]
    animal = input_path.name

    written = []
    for sd, shank_idx, point, slope, probe_insert in fits:
        lat = lateral_axis(tip_min, tip_max, shank_direction(slope))
        electrodes = shank_electrodes(shank_idx)
        df = place_electrodes(point, slope, probe_insert, lat, electrodes, coord_cols)
        df = annotate_regions(df, atlas, coord_cols)
        df = df[df["depth_um"] >= INSIDE_BRAIN_DEPTH_UM].reset_index(drop=True)
        out = sd / f"{animal}_np2_electrodes.csv"
        df.to_csv(out, index=False)
        written.append(out)
    return written


def _shank_index(folder_name):
    """Physical shank index from a dmc probe folder name, e.g.
    'neuropixels_probe_2' -> 2. Raises ValueError if no number is present."""
    nums = re.findall(r"\d+", folder_name)
    if not nums:
        raise ValueError(f"cannot parse a shank index from folder '{folder_name}'")
    return int(nums[-1])


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
