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


from napari_dmc_segload.np2_layout import shank_electrodes
from napari_dmc_segload.np2_results import ELECTRODE_CSV_SUFFIX
from napari_dmc_segload.np2_tract import place_electrodes

NP2_COORD_COLS = ["ap_coords", "dv_coords", "ml_coords"]
NP2_LATERAL = np.array([0.0, 0.0, 1.0])          # columns run along ml
NP2_TIP0 = np.array([600.0, 400.0, 500.0])       # shank 0 tip voxel
NP2_SLOPE = np.array([0.0, -0.1, 0.0])           # straight down DV, 1 voxel / 10 um
NP2_INSERT_UM = 4000.0
NP2_SHANK_PITCH_VOX = 25.0                       # 250 um at 10 um voxels


def _np2_acronym(depth_um):
    """Two stacked structures so a region ladder has more than one rung."""
    if depth_um >= 3000:
        return "ORBl"
    return "MOs"


def make_electrode_frame(shank, n_rows=100, tip=None, slope=None,
                         insert=NP2_INSERT_UM):
    """One shank's electrode DataFrame, shaped exactly like the widget's CSV."""
    tip = NP2_TIP0 + np.array([0.0, 0.0, NP2_SHANK_PITCH_VOX * shank]) if tip is None \
        else np.asarray(tip, float)
    slope = NP2_SLOPE if slope is None else np.asarray(slope, float)
    electrodes = [e for e in shank_electrodes(shank) if e["row"] < n_rows]
    df = place_electrodes(tip, slope, insert, NP2_LATERAL, electrodes, NP2_COORD_COLS)
    df["acronym"] = [_np2_acronym(d) for d in df["depth_um"]]
    df["structure_id"] = [723 if a == "ORBl" else 993 for a in df["acronym"]]
    df["name"] = ["Orbital area, lateral part" if a == "ORBl" else "Secondary motor area"
                  for a in df["acronym"]]
    return df


@pytest.fixture
def np2_frames():
    """Four well-formed shanks, 250 um apart, parallel, 200 electrodes each."""
    return {i: make_electrode_frame(i) for i in range(4)}


@pytest.fixture
def np2_animal_dir(tmp_path, np2_frames):
    """Animal folder holding those four shanks' electrode CSVs + params.json."""
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    for shank, df in np2_frames.items():
        sd = root / f"neuropixels_probe_{shank}"
        sd.mkdir(parents=True)
        df.to_csv(sd / f"animal01{ELECTRODE_CSV_SUFFIX}", index=False)
    (base / "params.json").write_text(
        json.dumps({"atlas_info": {"atlas": "allen_mouse_10um"}}))
    return base
