import numpy as np
import pandas as pd
from napari_dmc_segload.np2_regions import annotate_regions

COLS = ["ap_coords", "dv_coords", "ml_coords"]


class _FakeStructures:
    def __init__(self, data):
        self.data = data


class _FakeAtlas:
    def __init__(self, annotation, data):
        self.annotation = annotation
        self.structures = _FakeStructures(data)


def _atlas():
    annot = np.zeros((3, 3, 3), dtype=int)
    annot[1, 1, 1] = 997
    return _FakeAtlas(annot, {997: {"acronym": "grey", "name": "grey matter"}})


def test_labels_in_bounds_voxel():
    df = pd.DataFrame({"ap_coords": [1], "dv_coords": [1], "ml_coords": [1]})
    out = annotate_regions(df, _atlas(), COLS)
    assert out.loc[0, "structure_id"] == 997
    assert out.loc[0, "acronym"] == "grey"
    assert out.loc[0, "name"] == "grey matter"


def test_out_of_bounds_is_clipped_and_root():
    df = pd.DataFrame({"ap_coords": [100], "dv_coords": [0], "ml_coords": [0]})
    out = annotate_regions(df, _atlas(), COLS)          # clips to (2,0,0) -> label 0
    assert out.loc[0, "structure_id"] == 0
    assert out.loc[0, "acronym"] == "root"
    assert out.loc[0, "name"] == "root"


def test_does_not_mutate_input():
    df = pd.DataFrame({"ap_coords": [1], "dv_coords": [1], "ml_coords": [1]})
    annotate_regions(df, _atlas(), COLS)
    assert "structure_id" not in df.columns
