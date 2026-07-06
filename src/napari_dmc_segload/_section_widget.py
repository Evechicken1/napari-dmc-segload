"""Dock widget: section a whole-slide image into per-slice 16-bit TIFFs.

Ported from Histology/dmc_section.py. napari is already running and hands this
widget a viewer; the widget builds a downsampled overview, lets the user draw +
number one rectangle per section, and writes full-res crops into the tree
DMC-BrainMap expects:

    <output_root>/<animal_id>/stitched/<channel>/<animal_id>_sNNN.tif
"""

from pathlib import Path

import numpy as np
import tifffile
from napari import Viewer
from napari.utils.notifications import show_info
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .readers import make_reader, to_16bit, ZERO_PAD
from .numbering import compute_slice_numbers

# napari display colour per DMC label (purely cosmetic).
DISPLAY_CMAP = {"dapi": "blue", "green": "green", "n3": "cyan",
                "cy3": "red", "cy5": "magenta"}


def _build_form() -> FunctionGui:
    @magicgui(
        layout="vertical",
        input_path=dict(widget_type="FileEdit", mode="r",
                        filter="*.czi *.tif *.tiff", label="input image"),
        animal_id=dict(widget_type="LineEdit", label="animal_id"),
        output_root=dict(widget_type="FileEdit", mode="d", label="output root"),
        channel_labels=dict(widget_type="LineEdit", label="channel labels",
                            value="green,cy3,dapi"),
        order_by=dict(widget_type="ComboBox", label="order by",
                      choices=["manual", "grid", "draw", "vertical"],
                      value="manual"),
        overview_scale=dict(widget_type="FloatSpinBox", label="overview scale",
                            value=0.1, min=0.001, max=1.0, step=0.01),
        call_button=False,
    )
    def form(input_path, animal_id, output_root, channel_labels,
             order_by, overview_scale):
        pass

    # auto-fill animal_id from the chosen file's stem
    @form.input_path.changed.connect
    def _fill_animal_id(value):
        if value and not form.animal_id.value:
            form.animal_id.value = Path(str(value)).stem

    return form


class SectionWidget(QWidget):
    """Dock widget for sectioning whole-slide images into DMC input TIFFs."""

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        self.form = _build_form()
        self.layout().addWidget(self.form.native)

        for text, slot in [
            ("Load overview", self._load_overview),
            ("Draw rectangles", self._draw_mode),
            ("Number sections (click in order)", self._number_mode),
            ("Reset numbering", self._reset_numbering),
            ("Export crops", self._export),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            self.layout().addWidget(btn)

        # per-load state (None until a slide is loaded)
        self.reader = None
        self.labels = []
        self.H = self.W = self.ov_h = 0
        self.fy = self.fx = 1.0
        self._order_by = "manual"
        self._manual = False
        self._counter = 1

    # --- helpers -----------------------------------------------------------
    @property
    def _shapes(self):
        return self.viewer.layers["slices"] if "slices" in self.viewer.layers else None

    def _get_numbers(self):
        shapes = self._shapes
        if shapes is None:
            return []
        n = len(shapes.data)
        if self._manual:
            cur = [int(x) for x in list(shapes.features.get("slice", []))]
            return (cur + [0] * n)[:n]             # unassigned shapes show 0
        return compute_slice_numbers([np.asarray(s) for s in shapes.data],
                                     self.ov_h, self._order_by)

    def _set_numbers(self, nums):
        shapes = self._shapes
        if shapes is None:
            return
        shapes.features = {"slice": np.asarray(nums, dtype=int)}
        shapes.refresh_text()

    def _refresh_numbers(self, *_):
        self._set_numbers(self._get_numbers())

    # --- actions -----------------------------------------------------------
    def _load_overview(self):
        in_path = self.form.input_path.value
        if not in_path or not Path(str(in_path)).is_file():
            show_info("Select a valid input image (.czi/.tif/.tiff).")
            return
        in_path = Path(str(in_path))

        self.reader = make_reader(in_path)
        n_ch, self.H, self.W = self.reader.shape
        labels = [s.strip() for s in str(self.form.channel_labels.value).split(",")
                  if s.strip()]
        if len(labels) != n_ch:
            show_info(f"image has {n_ch} channels but {len(labels)} labels; "
                      f"using {labels[:n_ch]}")
        self.labels = labels[:n_ch]

        scale = float(self.form.overview_scale.value)
        ov = self.reader.overview(scale)          # (C, y, x)
        self.ov_h, ov_w = ov.shape[1], ov.shape[2]
        self.fy, self.fx = self.H / self.ov_h, self.W / ov_w

        # clear any previous section layers, then add fresh ones
        self._remove_section_layers()
        for ci, label in enumerate(self.labels):
            lo, hi = float(ov[ci].min()), float(ov[ci].max())
            kwargs = dict(
                name=label, colormap=DISPLAY_CMAP.get(label, "gray"),
                blending="additive",
            )
            if hi > lo:  # napari rejects non-increasing contrast limits (constant image)
                kwargs["contrast_limits"] = (lo, hi)
            self.viewer.add_image(ov[ci], **kwargs)
        shapes = self.viewer.add_shapes(
            name="slices", shape_type="rectangle",
            edge_color="yellow", edge_width=4, face_color="transparent",
            features={"slice": np.zeros(0, dtype=int)},
            text={"string": "{slice}", "size": 14, "color": "white",
                  "anchor": "center"},
        )
        shapes.mode = "add_rectangle"

        self._order_by = str(self.form.order_by.value)
        self._manual = self._order_by == "manual"
        self._counter = 1
        shapes.events.data.connect(self._refresh_numbers)
        shapes.mouse_drag_callbacks.append(self._click_to_number)
        show_info(f"loaded {in_path.name}: {n_ch}ch {self.H}x{self.W}")

    def _click_to_number(self, layer, event):
        # only act on a plain click while in manual-numbering (select) mode
        if not self._manual or layer.mode != "select":
            return
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged:                                # ignore box moves
            return
        val = layer.get_value(event.position, world=True)
        idx = val[0] if isinstance(val, (tuple, list)) else val
        if idx is None:
            return
        nums = self._get_numbers()
        nums[int(idx)] = self._counter
        self._counter += 1
        self._set_numbers(nums)

    def _draw_mode(self):
        if self._shapes is None:
            show_info("Load an overview first.")
            return
        self._shapes.mode = "add_rectangle"

    def _number_mode(self):
        if self._shapes is None:
            show_info("Load an overview first.")
            return
        self._manual = True
        self._counter = 1
        self._set_numbers([0] * len(self._shapes.data))
        self._shapes.mode = "select"

    def _reset_numbering(self):
        if self._shapes is None:
            return
        self._counter = 1
        self._set_numbers([0] * len(self._shapes.data))

    def _remove_section_layers(self):
        for name in list(self.labels) + ["slices"]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])

    def _export(self):
        if self.reader is None or self._shapes is None:
            show_info("Load an overview first.")
            return
        rects = [np.asarray(s) for s in self._shapes.data]
        if not rects:
            show_info("No rectangles drawn.")
            return

        numbers = self._get_numbers()
        if self._manual:
            if any(x <= 0 for x in numbers):
                show_info("Some sections are unnumbered (showing 0); number them all.")
                return
            if sorted(numbers) != list(range(1, len(numbers) + 1)):
                show_info(f"Numbers are not a clean 1..N sequence: {sorted(numbers)}.")
                return

        boxes = []
        for v, num in zip(rects, numbers):
            ys, xs = v[:, 0], v[:, 1]              # overview coords (row, col)
            r0 = int(np.clip(round(ys.min() * self.fy), 0, self.H))
            r1 = int(np.clip(round(ys.max() * self.fy), 0, self.H))
            c0 = int(np.clip(round(xs.min() * self.fx), 0, self.W))
            c1 = int(np.clip(round(xs.max() * self.fx), 0, self.W))
            boxes.append((num, r0, r1, c0, c1))
        boxes.sort(key=lambda b: b[0])

        animal_id = str(self.form.animal_id.value).strip() or "animal"
        out_root = Path(str(self.form.output_root.value)) if self.form.output_root.value \
            else Path.cwd() / "dmc_input"
        out_base = out_root / animal_id / "stitched"
        for label in self.labels:
            (out_base / label).mkdir(parents=True, exist_ok=True)

        n = 0
        for num, r0, r1, c0, c1 in boxes:
            if r1 <= r0 or c1 <= c0:
                continue
            fname = f"{animal_id}_s{str(num).zfill(ZERO_PAD)}.tif"
            for ci, label in enumerate(self.labels):
                crop = to_16bit(self.reader.crop_fullres(ci, r0, r1, c0, c1))
                tifffile.imwrite(out_base / label / fname, crop)
            n += 1

        show_info(f"wrote {n} slices -> {out_base}")
        if n > 0:
            self._remove_section_layers()          # clear for the next slide
            self.reader = None
