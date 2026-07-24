import json

import numpy as np
import pandas as pd
import pytest

from napari_dmc_segload.np2_results import (
    ELECTRODE_CSV_SUFFIX, coord_cols, load_electrode_csvs, probe_results_dir,
    shank_dirs, shank_index)

COLS = ["ap_coords", "dv_coords", "ml_coords"]


class _FakeSpace:
    axes_description = ("ap", "si", "rl")


class _FakeAtlas:
    space = _FakeSpace()
    resolution = (10.0, 10.0, 10.0)


def _write_electrodes(shank_dir, shank, n=6):
    """Minimal well-formed electrode CSV."""
    shank_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "shank": np.full(n, shank),
        "electrode": np.arange(n),
        "row": np.arange(n) // 2,
        "col": np.arange(n) % 2,
        "x_um": np.where(np.arange(n) % 2 == 0, 27, 59),
        "z_um": (np.arange(n) // 2) * 15,
        "distance_to_tip_um": (np.arange(n) // 2) * 15 + 175.0,
        "depth_um": 4000.0 - ((np.arange(n) // 2) * 15 + 175.0),
        "ap_coords": np.full(n, 600),
        "dv_coords": np.arange(n) + 400,
        "ml_coords": np.full(n, 500 + shank * 25),
        "structure_id": np.full(n, 723),
        "acronym": ["ORBl"] * n,
        "name": ["Orbital area, lateral part"] * n,
    })
    df.to_csv(shank_dir / f"animal01{ELECTRODE_CSV_SUFFIX}", index=False)


@pytest.fixture
def results_animal(tmp_path):
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    for i in (0, 1, 2, 3):
        _write_electrodes(root / f"neuropixels_probe_{i}", shank=i)
    return base


def test_shank_index_parses_trailing_number():
    assert shank_index("neuropixels_probe_2") == 2
    assert shank_index("neuropixels_probe_10") == 10


def test_shank_index_without_number_raises():
    with pytest.raises(ValueError):
        shank_index("neuropixels_probe")


def test_probe_results_dir_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_results_dir(tmp_path / "nope")


def test_shank_dirs_are_natsorted(results_animal):
    names = [d.name for d in shank_dirs(results_animal)]
    assert names == ["neuropixels_probe_0", "neuropixels_probe_1",
                     "neuropixels_probe_2", "neuropixels_probe_3"]


def test_load_electrode_csvs_keyed_by_shank_index(results_animal):
    frames = load_electrode_csvs(results_animal)
    assert sorted(frames) == [0, 1, 2, 3]
    assert all(isinstance(df, pd.DataFrame) for df in frames.values())
    assert frames[2]["shank"].unique().tolist() == [2]


def test_load_electrode_csvs_uses_folder_index_not_sort_position(tmp_path):
    # only shanks 0 and 2 traced: keys must be 0 and 2, not 0 and 1
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    _write_electrodes(root / "neuropixels_probe_0", shank=0)
    _write_electrodes(root / "neuropixels_probe_2", shank=2)
    assert sorted(load_electrode_csvs(base)) == [0, 2]


def test_load_electrode_csvs_no_files_raises(tmp_path):
    base = tmp_path / "animal01"
    (base / "results" / "neuropixels_probe" / "neuropixels_probe_0").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="NP2.0 electrodes"):
        load_electrode_csvs(base)


def test_load_electrode_csvs_missing_column_raises(results_animal):
    bad = results_animal / "results" / "neuropixels_probe" / "neuropixels_probe_1"
    csv = next(bad.glob(f"*{ELECTRODE_CSV_SUFFIX}"))
    pd.read_csv(csv).drop(columns=["depth_um"]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="depth_um"):
        load_electrode_csvs(results_animal)


def test_coord_cols_follows_atlas_axis_order():
    assert coord_cols(_FakeAtlas()) == COLS
