"""Discovery, rescale and orchestration for the raw segmentation-clicks overlay.

The dmc-dependent warp is injected here, so these tests need neither
napari-dmc-brainmap nor an atlas.
"""
import json
import cv2
import numpy as np
import pandas as pd
import pytest

from napari_dmc_segload.np2_segpoints import (
    SEG_CSV_SUFFIX, REGI_SUFFIX, register_probe_clicks, registration_channels,
    rescale_clicks, shank_click_csvs, stack_shank_points)


def _img(path, h, w):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w), dtype=np.uint8))


def test_rescale_clicks_scales_by_ratio_and_drops_extents():
    df = pd.DataFrame({"Position Y": [0, 50, 100], "Position X": [0, 60, 120]})
    out = rescale_clicks(df, (100, 120), (10, 12))     # full max -> low max
    assert out.shape == (3, 2)                          # (x, y), appended extents gone
    np.testing.assert_array_equal(out[:, 0], [0, 6, 12])   # x scaled by 12/120
    np.testing.assert_array_equal(out[:, 1], [0, 5, 10])   # y scaled by 10/100


def test_registration_channels_only_those_with_json(tmp_path):
    base = tmp_path / "animal"
    (base / "sharpy_track" / "dapi").mkdir(parents=True)
    (base / "sharpy_track" / "dapi" / "registration.json").write_text("{}")
    (base / "sharpy_track" / "green").mkdir(parents=True)      # no registration.json
    assert registration_channels(base) == ["dapi"]


def test_registration_channels_missing_dir(tmp_path):
    assert registration_channels(tmp_path / "nope") == []


def test_shank_click_csvs_keyed_by_folder_index(tmp_path):
    base = tmp_path / "animal"
    seg = base / "segmentation" / "neuropixels_probe"
    for i in (0, 2):
        d = seg / f"neuropixels_probe_{i}"
        d.mkdir(parents=True)
        (d / f"animal_s00{i}{SEG_CSV_SUFFIX}").write_text("x")
    (seg / "neuropixels_probe_1").mkdir()                       # empty -> skipped
    out = shank_click_csvs(base)
    assert sorted(out) == [0, 2]
    assert len(out[0]) == 1


def test_shank_click_csvs_missing_tree(tmp_path):
    assert shank_click_csvs(tmp_path / "nope") == {}


def test_stack_shank_points_orders_ascending_and_labels():
    by = {2: np.array([[1, 2, 3], [4, 5, 6]]), 0: np.array([[7, 8, 9]])}
    points, features = stack_shank_points(by)
    assert points.shape == (3, 3)
    assert features["shank"].tolist() == [0, 2, 2]             # shank 0 first
    np.testing.assert_array_equal(points[0], [7, 8, 9])


def test_stack_shank_points_empty():
    points, features = stack_shank_points({})
    assert points.shape == (0, 3)
    assert list(features.columns) == ["shank"] and len(features) == 0


@pytest.fixture
def seg_tree(tmp_path):
    """Tiny animal tree: 2 shanks of click CSVs + the images register_probe_clicks
    reads for dimensions + a registration.json with the imgName mapping."""
    base = tmp_path / "animal"
    chan = "dapi"
    stems = {0: ["animal_s001", "animal_s002"], 2: ["animal_s003"]}
    img_names, idx = {}, 0
    for shank, ss in stems.items():
        d = base / "segmentation" / "neuropixels_probe" / f"neuropixels_probe_{shank}"
        d.mkdir(parents=True)
        for st in ss:
            pd.DataFrame({"Position Y": [10, 20, 30],
                          "Position X": [10, 20, 30]}).to_csv(d / f"{st}{SEG_CSV_SUFFIX}")
            _img(base / "sharpy_track" / chan / f"{st}{REGI_SUFFIX}", 10, 12)
            _img(base / "single_channel" / chan / f"{st}_single.tif", 100, 120)
            img_names[str(idx)] = f"{st}{REGI_SUFFIX}"
            idx += 1
    st_dir = base / "sharpy_track" / chan
    (st_dir / "registration.json").write_text(json.dumps({"imgName": img_names}))
    return base


def test_register_probe_clicks_with_injected_warp(seg_tree):
    seen = []

    def fake_warp(coords_xy, key):
        seen.append((key, np.asarray(coords_xy).copy()))
        return [[c[0] + 1000, c[1] + 2000, 700] for c in coords_xy]

    out = register_probe_clicks(seg_tree, "dapi", atlas=None, warp=fake_warp)
    assert sorted(out) == [0, 2]
    assert out[0].shape == (6, 3)                 # 2 sections x 3 clicks
    assert out[2].shape == (3, 3)
    # the warp received registration-space coords (x<=12, y<=10), not full-res
    _, coords = seen[0]
    assert coords.shape == (3, 2)
    assert coords[:, 0].max() <= 12 and coords[:, 1].max() <= 10


def test_register_probe_clicks_no_segmentation_returns_empty(tmp_path):
    assert register_probe_clicks(tmp_path / "empty", "dapi", atlas=None,
                                 warp=lambda c, k: []) == {}
