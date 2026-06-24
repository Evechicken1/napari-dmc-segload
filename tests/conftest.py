import json
import numpy as np
import cv2
import pytest


def _write_tif(path, value=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((8, 8), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture
def animal_dir(tmp_path):
    """A fake DMC animal folder: single_channel/{dapi,cy3} with 3 images each,
    an empty 'n3' folder, and params.json listing dapi/green/cy3 as imaged."""
    base = tmp_path / "animal01"
    for chan in ("dapi", "cy3"):
        for name in ("animal01_obj1_1_stitched.tif",
                     "animal01_obj1_2_stitched.tif",
                     "animal01_obj1_10_stitched.tif"):
            _write_tif(base / "single_channel" / chan / name)
    (base / "single_channel" / "n3").mkdir(parents=True, exist_ok=True)  # empty
    params = {"general": {"chans_imaged": ["dapi", "green", "cy3"]}}
    (base / "params.json").write_text(json.dumps(params))
    return base
