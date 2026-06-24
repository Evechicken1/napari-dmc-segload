from pathlib import Path

from napari import Viewer
from napari.utils.notifications import show_info
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

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


def _build_form() -> FunctionGui:
    @magicgui(
        layout="vertical",
        input_path=dict(widget_type="FileEdit", mode="d",
                        label="input path (animal_id): "),
        reference_channels=dict(widget_type="Select", label="reference channels",
                                choices=[], value=[]),
        segmentation_channels=dict(widget_type="Select", label="segmentation channels",
                                   choices=[], value=[]),
        seg_type=dict(widget_type="ComboBox", label="segmentation type",
                      choices=ALL_SEG_TYPES, value="cells"),
        n_probes=dict(widget_type="SpinBox", label="number of fibers/probes",
                      value=1, min=1),
        point_size=dict(widget_type="SpinBox", label="point size", value=5, min=1),
        image_idx=dict(widget_type="SpinBox", label="image index", value=0, min=0),
        call_button=False,
    )
    def form(input_path, reference_channels, segmentation_channels, seg_type,
             n_probes, point_size, image_idx):
        pass
    return form


class SegLoaderWidget(QWidget):
    """Dock widget: scan channels, load per-channel images + tracing layers,
    and save traced layers in dmc's on-disk format."""

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        self.form = _build_form()
        self.layout().addWidget(self.form.native)

        self.btn_scan = QPushButton("Scan channels")
        self.btn_scan.clicked.connect(self._scan_channels)
        self.layout().addWidget(self.btn_scan)

        self.btn_load = QPushButton("Load into viewer")
        self.btn_load.clicked.connect(self._load_into_viewer)
        self.layout().addWidget(self.btn_load)

        self.btn_save = QPushButton("Save segmentation")
        self.btn_save.clicked.connect(self._save_segmentation)
        self.layout().addWidget(self.btn_save)

    # --- helpers -----------------------------------------------------------
    def _input_path(self):
        p = self.form.input_path.value
        if not p or not Path(p).is_dir():
            show_info("Select a valid input path (animal_id) directory.")
            return None
        return Path(p)

    def _selected_channels(self):
        ref = list(self.form.reference_channels.value)
        seg = list(self.form.segmentation_channels.value)
        overlap = set(ref) & set(seg)
        if overlap:
            show_info(f"Channel(s) {sorted(overlap)} assigned to both roles; "
                      "put each channel in only one list.")
            return None, None
        if not ref and not seg:
            show_info("No channels selected.")
            return None, None
        return ref, seg

    # --- actions -----------------------------------------------------------
    def _scan_channels(self):
        input_path = self._input_path()
        if input_path is None:
            return
        chans = ChannelScanner(input_path).available_channels()
        if not chans:
            show_info("No channels with images found under single_channel/.")
        self.form.reference_channels.choices = chans
        self.form.segmentation_channels.choices = chans

    def _add_image_layer(self, input_path, chan, idx, opacity):
        try:
            arr = load_grayscale(resolve_image(input_path, chan, idx))
        except IndexError as e:
            show_info(str(e))
            return False
        self.viewer.add_image(
            arr, name=f"{chan} channel",
            colormap=DISPLAY_CMAP.get(chan, "gray"), opacity=opacity,
        )
        return True

    def _add_tracing_layers(self, seg_type, seg_channels, n_probes, point_size):
        if seg_type in PROBE_SEG_TYPES:
            for i in range(n_probes):
                self.viewer.add_points(name=f"{seg_type}_{i}", size=point_size)
        elif seg_type in SHAPE_SEG_TYPES:
            for chan in seg_channels:
                self.viewer.add_shapes(name=chan)
        else:  # POINT_SEG_TYPES
            for chan in seg_channels:
                self.viewer.add_points(name=chan, size=point_size)

    def _load_into_viewer(self):
        input_path = self._input_path()
        if input_path is None:
            return
        ref, seg = self._selected_channels()
        if ref is None:
            return
        idx = int(self.form.image_idx.value)
        point_size = int(self.form.point_size.value)
        for chan in ref:
            self._add_image_layer(input_path, chan, idx, REFERENCE_OPACITY)
        for chan in seg:
            self._add_image_layer(input_path, chan, idx, SEGMENTATION_OPACITY)
        self._add_tracing_layers(
            self.form.seg_type.value, seg, int(self.form.n_probes.value), point_size
        )

    def _save_segmentation(self):
        input_path = self._input_path()
        if input_path is None:
            return
        ref, seg = self._selected_channels()
        if ref is None:
            return
        seg_type = self.form.seg_type.value
        idx = int(self.form.image_idx.value)
        ref_for_stem = (seg + ref)
        if not ref_for_stem:
            show_info("No channels selected.")
            return
        stem = image_stem(input_path, ref_for_stem[0], idx)

        if seg_type in PROBE_SEG_TYPES:
            ids = [f"{seg_type}_{i}" for i in range(int(self.form.n_probes.value))]
            for pid in ids:
                if pid in self.viewer.layers:
                    save_points_layer(self.viewer.layers[pid].data,
                                      input_path, seg_type, pid, stem)
        elif seg_type in SHAPE_SEG_TYPES:
            for chan in seg:
                if chan in self.viewer.layers:
                    save_shapes_layer(list(self.viewer.layers[chan].data),
                                      input_path, seg_type, chan, stem)
        else:
            if not seg:
                show_info("No segmentation channels selected to save.")
                return
            for chan in seg:
                if chan in self.viewer.layers:
                    save_points_layer(self.viewer.layers[chan].data,
                                      input_path, seg_type, chan, stem)
        show_info("Segmentation saved.")
