import numpy as np
import pandas as pd
import pytest
from napari_dmc_segload._electrodes_widget import reconstruct_electrodes


class _FakeStructures:
    def __init__(self, data):
        self.data = data


class _FakeSpace:
    axes_description = ("ap", "si", "rl")   # allen coronal -> ap, dv, ml


class _FakeAtlas:
    def __init__(self):
        self.annotation = np.full((200, 200, 200), 997, dtype=int)
        self.structures = _FakeStructures({997: {"acronym": "grey", "name": "grey matter"}})
        self.space = _FakeSpace()


def _write_tract(shank_dir, tip, insert, filename="animal01_neuropixels_probe.csv"):
    shank_dir.mkdir(parents=True)
    dtt = np.arange(60) * 20 + 175
    # line: dv decreases as we move away from tip; ap/ml constant
    ap = np.full(60, tip[0])
    dv = tip[1] - 0.05 * dtt
    ml = np.full(60, tip[2])
    pd.DataFrame({
        "distance_to_tip(um)": dtt,
        "depth(um)": insert - dtt,
        "ap_coords": np.round(ap).astype(int),
        "dv_coords": np.round(dv).astype(int),
        "ml_coords": np.round(ml).astype(int),
    }).to_csv(shank_dir / filename)


@pytest.fixture
def animal_dir(tmp_path):
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    _write_tract(root / "neuropixels_probe_0", tip=(60, 120, 40), insert=2000)
    _write_tract(root / "neuropixels_probe_1", tip=(60, 120, 65), insert=2000)  # +25 vox = +250 um
    return base


def test_reconstruct_writes_one_csv_per_shank(animal_dir):
    written = reconstruct_electrodes(animal_dir, _FakeAtlas())
    assert len(written) == 2
    for out in written:
        assert out.name == "animal01_np2_electrodes.csv"
        df = pd.read_csv(out)
        expected = {"shank", "electrode", "row", "col", "x_um", "z_um",
                    "distance_to_tip_um", "depth_um",
                    "ap_coords", "dv_coords", "ml_coords",
                    "structure_id", "acronym", "name"}
        assert expected <= set(df.columns)
        # inside-brain filtering removed most of the 1280 template electrodes
        assert 0 < len(df) < 1280
        assert (df["depth_um"] >= -5).all()


def test_reconstruct_requires_two_shanks(tmp_path):
    base = tmp_path / "animal01"
    _write_tract(base / "results" / "neuropixels_probe" / "neuropixels_probe_0",
                 tip=(60, 120, 40), insert=2000)
    with pytest.raises(ValueError):
        reconstruct_electrodes(base, _FakeAtlas())


def test_reconstruct_missing_results_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        reconstruct_electrodes(tmp_path / "nope", _FakeAtlas())


def test_shank_index_comes_from_folder_name_not_position(tmp_path):
    # Only shanks 0 and 2 were traced. Each output must carry its true physical
    # shank number (0 and 2), not the sort positions (0 and 1).
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    _write_tract(root / "neuropixels_probe_0", tip=(60, 120, 40), insert=2000)
    _write_tract(root / "neuropixels_probe_2", tip=(60, 120, 90), insert=2000)
    written = reconstruct_electrodes(base, _FakeAtlas())
    shanks = {out.parent.name: pd.read_csv(out)["shank"].unique().tolist()
              for out in written}
    assert shanks["neuropixels_probe_0"] == [0]
    assert shanks["neuropixels_probe_2"] == [2]


def test_electrode_csv_alone_is_never_treated_as_a_tract(tmp_path):
    # Each shank folder contains ONLY a previous run's electrode-output CSV --
    # no tract CSV at all. This must fail with "no tract CSV", independent of
    # glob ordering: there is nothing else in the folder a buggy glob could
    # accidentally pick correctly.
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    electrode_cols = ["shank", "electrode", "row", "col", "x_um", "z_um",
                       "distance_to_tip_um", "depth_um", "ap_coords",
                       "dv_coords", "ml_coords", "structure_id", "acronym", "name"]
    for i in (0, 1):
        d = root / f"neuropixels_probe_{i}"
        d.mkdir(parents=True)
        pd.DataFrame([[i, 0, 0, 0, 43.0, 0.0, 175.0, 2000.0, 60, 120, 40, 997,
                        "grey", "grey matter"]],
                     columns=electrode_cols).to_csv(d / "animal01_np2_electrodes.csv",
                                                     index=False)
    with pytest.raises(FileNotFoundError, match="no tract CSV"):
        reconstruct_electrodes(base, _FakeAtlas())


def test_rerun_ignores_previous_electrode_output(tmp_path):
    # Name the tract file so that "<animal>_np2_electrodes.csv" (the previous
    # run's own output, present by the second call) sorts BEFORE it -- the
    # adversarial ordering the unfiltered `list(sd.glob("*.csv"))` glob would
    # pick first, so a bug in tract selection cannot hide behind favourable
    # filesystem ordering.
    base = tmp_path / "animal01"
    root = base / "results" / "neuropixels_probe"
    _write_tract(root / "neuropixels_probe_0", tip=(60, 120, 40), insert=2000,
                 filename="zzz_tract.csv")
    _write_tract(root / "neuropixels_probe_1", tip=(60, 120, 65), insert=2000,
                 filename="zzz_tract.csv")
    first = reconstruct_electrodes(base, _FakeAtlas())
    second = reconstruct_electrodes(base, _FakeAtlas())
    assert len(second) == len(first) == 2
    for a, b in zip(first, second):
        assert pd.read_csv(a).equals(pd.read_csv(b))
