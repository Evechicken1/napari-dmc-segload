"""Dock widget: reconstruct NP2.0 electrode locations from dmc's per-shank
registered probe tract, after the animal has been registered in dmc.

Trace shanks as dmc `neuropixels_probe` (4 shanks), run dmc registration +
Results, then run this. Writes one CSV per shank with each electrode's atlas
coordinates and region.
"""

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
            atlas = load_atlas(p)
            written = reconstruct_electrodes(p, atlas)
        except (FileNotFoundError, ValueError, KeyError) as e:
            show_info(str(e))
            return
        show_info(f"Wrote {len(written)} shank electrode file(s).")
