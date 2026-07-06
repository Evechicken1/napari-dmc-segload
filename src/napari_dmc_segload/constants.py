# Canonical channel ordering used everywhere channels are sorted/displayed.
CANONICAL_CHANNELS = ["dapi", "green", "n3", "cy3", "cy5"]

# Default per-channel display colormaps (mirrors dmc's get_cmap('display')).
DISPLAY_CMAP = {
    "dapi": "blue",
    "green": "green",
    "n3": "orange",
    "cy3": "red",
    "cy5": "pink",
}

# Role opacities for loaded image layers.
REFERENCE_OPACITY = 0.5
SEGMENTATION_OPACITY = 1.0

# Segmentation-type groupings that decide the tracing-layer kind.
POINT_SEG_TYPES = {"cells", "projections"}
SHAPE_SEG_TYPES = {"injection_site"}
PROBE_SEG_TYPES = {"optic_fiber", "neuropixels_probe"}
ALL_SEG_TYPES = ["cells", "injection_site", "optic_fiber", "neuropixels_probe",
                 "projections", "neuropixels_2.0"]

# Neuropixels 2.0 recording-site reconstruction.
SITE_SEG_TYPES = {"neuropixels_2.0"}
NP2_DEFAULT_ROWS = 48
NP2_DEFAULT_COLUMNS = 2
NP2_ROW_PITCH_UM = 15.0
NP2_COLUMN_SPACING_UM = 32.0
