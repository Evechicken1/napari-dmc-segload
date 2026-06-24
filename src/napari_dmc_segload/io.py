from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from natsort import natsorted


def _channel_tifs(input_path: Path, chan: str):
    chan_dir = Path(input_path) / "single_channel" / chan
    return natsorted([p for p in chan_dir.glob("*.tif")])


def resolve_image(input_path: Path, chan: str, image_idx: int) -> Path:
    """Path to the natsorted image at `image_idx` in single_channel/<chan>/."""
    tifs = _channel_tifs(input_path, chan)
    if image_idx < 0 or image_idx >= len(tifs):
        raise IndexError(
            f"image_idx {image_idx} out of range for channel '{chan}' "
            f"({len(tifs)} images)"
        )
    return tifs[image_idx]


def image_stem(input_path: Path, chan: str, image_idx: int) -> str:
    """Image file name without its .tif extension (dmc's save base name)."""
    return resolve_image(input_path, chan, image_idx).with_suffix("").name


def load_grayscale(path: Path) -> np.ndarray:
    """Load a single-channel image as a 2-D grayscale array."""
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _seg_dir(input_path: Path, seg_type: str, chan: str) -> Path:
    d = Path(input_path) / "segmentation" / seg_type / chan
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_name(seg_dir: Path, stem: str, seg_type: str) -> Path:
    return seg_dir / f"{stem}_{seg_type}.csv"


def save_points_layer(data, input_path: Path, seg_type: str, chan: str, stem: str):
    """Write a points layer to dmc's CSV format. Empty data removes a stale file
    and returns None; otherwise returns the written path."""
    seg_dir = _seg_dir(input_path, seg_type, chan)
    save_name = _save_name(seg_dir, stem, seg_type)
    arr = np.asarray(data)
    if len(arr) == 0:
        if save_name.exists():
            save_name.unlink()
        return None
    df = pd.DataFrame(arr, columns=["Position Y", "Position X"])
    df.to_csv(save_name)  # default index column, matching dmc
    return save_name


def save_shapes_layer(shapes, input_path: Path, seg_type: str, chan: str, stem: str):
    """Write a shapes (polygon) layer to dmc's CSV format with idx_shape."""
    seg_dir = _seg_dir(input_path, seg_type, chan)
    save_name = _save_name(seg_dir, stem, seg_type)
    if len(shapes) == 0:
        if save_name.exists():
            save_name.unlink()
        return None
    frames = []
    for i, poly in enumerate(shapes):
        df = pd.DataFrame(np.asarray(poly), columns=["Position Y", "Position X"])
        df["idx_shape"] = i
        frames.append(df)
    pd.concat(frames).to_csv(save_name)
    return save_name
