"""Where an animal's NP2.0 reconstruction lives on disk, and how to load it.

dmc writes its registered probe tracts to
`results/neuropixels_probe/neuropixels_probe_<i>/`; the "NP2.0 electrodes"
widget writes one `<animal>_np2_electrodes.csv` per shank into the same folders.
This module owns those locations so the electrodes widget and the visualization
widget agree on them.
"""

import json
import re
from pathlib import Path

import pandas as pd
from natsort import natsorted

PROBE_RESULTS_SUBDIR = ("results", "neuropixels_probe")
ELECTRODE_CSV_SUFFIX = "_np2_electrodes.csv"

# dmc names atlas-coord columns from the atlas axis order via this map.
AXIS_TO_COORD_COL = {"ap": "ap_coords", "si": "dv_coords", "rl": "ml_coords"}

# Columns every consumer of an electrode CSV relies on. The atlas-coordinate
# columns' *names* (ap_coords/dv_coords/ml_coords) are fixed and included here;
# only their *order* depends on the atlas (see coord_cols).
REQUIRED_ELECTRODE_COLS = ("shank", "electrode", "distance_to_tip_um",
                           "depth_um", "acronym", "ap_coords", "dv_coords",
                           "ml_coords")


def probe_results_dir(input_path):
    """Path to `results/neuropixels_probe` under an animal folder.

    Raises FileNotFoundError if it does not exist (dmc Results not run yet)."""
    d = Path(input_path).joinpath(*PROBE_RESULTS_SUBDIR)
    if not d.is_dir():
        raise FileNotFoundError(
            f"no 'results/neuropixels_probe' under {input_path}; run dmc Results first")
    return d


def shank_index(folder_name):
    """Physical shank index from a dmc probe folder's name, e.g.
    'neuropixels_probe_2' -> 2. Raises ValueError if no number is present."""
    nums = re.findall(r"\d+", folder_name)
    if not nums:
        raise ValueError(f"cannot parse a shank index from folder '{folder_name}'")
    return int(nums[-1])


def shank_dirs(input_path):
    """All subdirectories of the animal's probe results folder, natsorted by name."""
    return natsorted([p for p in probe_results_dir(input_path).iterdir() if p.is_dir()],
                     key=lambda p: p.name)


def load_electrode_csvs(input_path):
    """Load each shank folder's single `*_np2_electrodes.csv`.

    Returns {shank_index: DataFrame}, keyed by the index parsed from the folder
    name (never the sort position). Raises FileNotFoundError if no electrode CSV
    exists anywhere, and ValueError if: a folder contains more than one candidate
    CSV, a CSV is missing a required column or has no rows, a CSV's own 'shank'
    column disagrees with the index parsed from its folder, or two folders map
    to the same shank index.
    """
    frames = {}
    seen = {}
    for sd in shank_dirs(input_path):
        csvs = sorted(p for p in sd.glob(f"*{ELECTRODE_CSV_SUFFIX}") if p.is_file())
        if not csvs:
            continue
        if len(csvs) > 1:
            raise ValueError(
                f"{sd} contains {len(csvs)} electrode CSVs "
                f"({', '.join(c.name for c in csvs)}); keep only the current animal's")
        csv = csvs[0]
        df = pd.read_csv(csv)
        missing = [c for c in REQUIRED_ELECTRODE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{csv} is missing column(s): {', '.join(missing)}")
        if df.empty:
            raise ValueError(f"{csv} has no rows; re-run the 'NP2.0 electrodes' widget")
        idx = shank_index(sd.name)
        col_shanks = sorted(df["shank"].unique().tolist())
        if col_shanks != [idx]:
            raise ValueError(
                f"{csv}: folder '{sd.name}' implies shank {idx}, but its 'shank' "
                f"column contains {col_shanks}")
        if idx in seen:
            raise ValueError(
                f"two probe folders map to shank {idx}: {seen[idx].name} and {sd.name}")
        seen[idx] = sd
        frames[idx] = df
    if not frames:
        raise FileNotFoundError(
            f"no '*{ELECTRODE_CSV_SUFFIX}' under {probe_results_dir(input_path)}; "
            "run the 'NP2.0 electrodes' widget first")
    return frames


def coord_cols(atlas):
    """dmc's atlas-coordinate column names, in the atlas's own axis order."""
    cols = []
    for ax in atlas.space.axes_description:
        if ax not in AXIS_TO_COORD_COL:
            raise ValueError(
                f"unsupported atlas axis '{ax}'; expected an ap/si/rl-oriented atlas")
        cols.append(AXIS_TO_COORD_COL[ax])
    return cols


def load_atlas(input_path):
    """Build the BrainGlobeAtlas named in the animal's params.json."""
    params_fn = Path(input_path) / "params.json"
    if not params_fn.is_file():
        raise FileNotFoundError(f"no params.json under {input_path}")
    with open(params_fn) as f:
        params = json.load(f)
    try:
        atlas_name = params["atlas_info"]["atlas"]
    except KeyError:
        raise KeyError(f"{params_fn} has no ['atlas_info']['atlas'] entry")
    from bg_atlasapi import BrainGlobeAtlas   # imported lazily (heavy, optional in tests)
    return BrainGlobeAtlas(atlas_name)
