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

# Columns every consumer of an electrode CSV relies on (coord columns are
# checked separately, since their names depend on the atlas axis order).
REQUIRED_ELECTRODE_COLS = ("shank", "electrode", "distance_to_tip_um",
                           "depth_um", "acronym")


def probe_results_dir(input_path):
    """Path to `results/neuropixels_probe` under an animal folder.

    Raises FileNotFoundError if it does not exist (dmc Results not run yet)."""
    d = Path(input_path).joinpath(*PROBE_RESULTS_SUBDIR)
    if not d.is_dir():
        raise FileNotFoundError(
            f"no 'results/neuropixels_probe' under {input_path}; run dmc Results first")
    return d


def shank_index(folder_name):
    """Physical shank index from a dmc probe folder name, e.g.
    'neuropixels_probe_2' -> 2. Raises ValueError if no number is present."""
    nums = re.findall(r"\d+", str(folder_name))
    if not nums:
        raise ValueError(f"cannot parse a shank index from folder '{folder_name}'")
    return int(nums[-1])


def shank_dirs(input_path):
    """Per-shank result folders, natsorted by name."""
    return natsorted([p for p in probe_results_dir(input_path).iterdir() if p.is_dir()],
                     key=lambda p: p.name)


def load_electrode_csvs(input_path):
    """Load every `*_np2_electrodes.csv` under the probe results.

    Returns {shank_index: DataFrame}, keyed by the index parsed from the folder
    name (never the sort position). Raises FileNotFoundError if no electrode CSV
    exists anywhere, and ValueError if one is missing a required column.
    """
    frames = {}
    for sd in shank_dirs(input_path):
        csvs = sorted(sd.glob(f"*{ELECTRODE_CSV_SUFFIX}"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        missing = [c for c in REQUIRED_ELECTRODE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{csvs[0]} is missing column(s): {', '.join(missing)}")
        frames[shank_index(sd.name)] = df
    if not frames:
        raise FileNotFoundError(
            f"no '*{ELECTRODE_CSV_SUFFIX}' under {probe_results_dir(input_path)}; "
            "run the 'NP2.0 electrodes' widget first")
    return frames


def coord_cols(atlas):
    """dmc's atlas-coordinate column names, in the atlas's own axis order."""
    return [AXIS_TO_COORD_COL[ax] for ax in atlas.space.axes_description]


def load_atlas(input_path):
    """Build the BrainGlobeAtlas named in the animal's params.json."""
    params_fn = Path(input_path) / "params.json"
    if not params_fn.is_file():
        raise FileNotFoundError(f"no params.json under {input_path}")
    with open(params_fn) as f:
        params = json.load(f)
    atlas_name = params["atlas_info"]["atlas"]
    from bg_atlasapi import BrainGlobeAtlas   # imported lazily (heavy, optional in tests)
    return BrainGlobeAtlas(atlas_name)
