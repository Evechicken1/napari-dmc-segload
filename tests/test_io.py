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
