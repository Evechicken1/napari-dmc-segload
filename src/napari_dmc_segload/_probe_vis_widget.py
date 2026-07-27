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

        # Build everything that can fail from the atlas *before* touching the
        # viewer, so a bad mesh file (or any other scene-build error) reports
        # via show_info and leaves the user's existing layers untouched
        # instead of wiping them and then dying.
        try:
            verts, faces = root_mesh(atlas)
            meshes, skipped = region_meshes(frames, atlas)
        except Exception as e:
            show_info(f"Could not build the probe scene: {e}")
            return

        self._atlas = atlas
        self._features = features

        self._clear_layers()

        self.viewer.add_surface((verts, faces), name=self.ROOT_LAYER,
                                colormap=_solid_colormap(ROOT_COLOR),
                                opacity=ROOT_OPACITY, shading="smooth")

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
        try:
            out.write_text(text)
        except OSError as e:
            show_info(f"Visualized {len(frames)} shank(s); could not write QC file: {e}")
            return
        show_info(f"Visualized {len(frames)} shank(s); QC written to {out.name}")

    def _recolour(self, *_):
        if self._features is None or self.POINTS_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[self.POINTS_LAYER]
        layer.face_color = point_colors(self._features, self.form.colour_by.value,
                                        self._atlas)
