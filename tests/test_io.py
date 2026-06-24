import numpy as np
import pytest
from napari_dmc_segload.io import resolve_image, image_stem, load_grayscale


def test_resolve_image_uses_natsort_order(animal_dir):
    # natsorted: _1, _2, _10  -> index 2 is the _10 image
    p = resolve_image(animal_dir, "dapi", 2)
    assert p.name == "animal01_obj1_10_stitched.tif"


def test_resolve_image_out_of_range_raises(animal_dir):
    with pytest.raises(IndexError):
        resolve_image(animal_dir, "dapi", 99)


def test_image_stem_strips_extension_only(animal_dir):
    assert image_stem(animal_dir, "dapi", 0) == "animal01_obj1_1_stitched"


def test_load_grayscale_returns_2d_array(animal_dir):
    arr = load_grayscale(resolve_image(animal_dir, "dapi", 0))
    assert isinstance(arr, np.ndarray) and arr.ndim == 2


import pandas as pd
from napari_dmc_segload.io import save_points_layer, save_shapes_layer


def test_save_points_writes_dmc_csv(animal_dir):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])  # [Position Y, Position X]
    out = save_points_layer(data, animal_dir, "cells", "cy3", "img1")
    assert out == animal_dir / "segmentation" / "cells" / "cy3" / "img1_cells.csv"
    df = pd.read_csv(out)
    assert list(df.columns)[-2:] == ["Position Y", "Position X"]
    assert df[["Position Y", "Position X"]].values.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_save_points_empty_removes_stale_file(animal_dir):
    out_dir = animal_dir / "segmentation" / "cells" / "cy3"
    out_dir.mkdir(parents=True)
    stale = out_dir / "img1_cells.csv"
    stale.write_text("old")
    result = save_points_layer(np.empty((0, 2)), animal_dir, "cells", "cy3", "img1")
    assert result is None
    assert not stale.exists()


def test_save_shapes_writes_idx_shape_per_polygon(animal_dir):
    poly_a = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    poly_b = np.array([[5.0, 5.0], [5.0, 6.0], [6.0, 6.0]])
    out = save_shapes_layer([poly_a, poly_b], animal_dir, "injection_site", "cy3", "img1")
    df = pd.read_csv(out)
    assert set(df["idx_shape"]) == {0, 1}
    assert list(df["idx_shape"]).count(0) == 3
    assert {"Position Y", "Position X", "idx_shape"} <= set(df.columns)
