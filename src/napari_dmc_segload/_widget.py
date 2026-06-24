from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel


class SegLoaderWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Segmentation loader (scaffold)"))
