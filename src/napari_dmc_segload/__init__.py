from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napari-dmc-segload")
except PackageNotFoundError:
    __version__ = "unknown"
