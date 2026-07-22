"""Baked Neuropixels 2.0 4-shank electrode template (index -> physical location).

Generated from the standard NP2.0 (NP2010) geometry recorded in
`260228_RDP282_g0_t0.imec0.ap.meta` (snsGeomMap). This is the *physical*
electrode layout, which is standard across sessions and channelmap-independent.
The mapping of recording channels to electrodes is session-specific and is
handled downstream, NOT here. To regenerate if the standard layout ever changes,
re-parse a snsGeomMap and update SHANK_Z_UM / SHANK_X_UM below.
"""

N_SHANKS = 4
# Full NP2.0 4-shank (NP2010) shank: 640 depth rows, 15 um pitch, z measured from
# the shank tip (deepest electrode = 0 um). The 27/59 um columns and 15 um pitch
# were parsed from the example snsGeomMap; the 640-row (1280 electrode/shank)
# extent is the NP2010 spec (re-verify against a snsGeomMap/imro if it ever changes).
SHANK_Z_UM = [15 * r for r in range(640)]  # 0, 15, ..., 9585
# Two columns across the 70 um shank (centre 43 um).
SHANK_X_UM = [27, 59]


def shank_electrodes(shank):
    """Template electrodes for one shank, row-major (row outer, col inner).

    Returns a list of dicts {shank, electrode, row, col, x_um, z_um}, where
    `electrode = row * 2 + col` is the physical electrode id within the shank.
    """
    electrodes = []
    for row, z in enumerate(SHANK_Z_UM):
        for col, x in enumerate(SHANK_X_UM):
            electrodes.append({
                "shank": shank,
                "electrode": row * len(SHANK_X_UM) + col,
                "row": row,
                "col": col,
                "x_um": x,
                "z_um": z,
            })
    return electrodes
