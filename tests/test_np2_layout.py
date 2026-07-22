from napari_dmc_segload.np2_layout import (
    N_SHANKS, SHANK_Z_UM, SHANK_X_UM, shank_electrodes)


def test_template_dimensions():
    assert N_SHANKS == 4
    assert len(SHANK_Z_UM) == 640                 # full shank
    assert SHANK_X_UM == [27, 59]
    assert SHANK_Z_UM[0] == 0 and SHANK_Z_UM[-1] == 9585   # 15 * 639
    assert all(SHANK_Z_UM[i + 1] - SHANK_Z_UM[i] == 15 for i in range(639))


def test_shank_electrodes_count_and_ids():
    e = shank_electrodes(2)
    assert len(e) == 1280                         # 640 rows x 2 cols
    assert all(r["shank"] == 2 for r in e)
    assert e[0] == {"shank": 2, "electrode": 0, "row": 0, "col": 0, "x_um": 27, "z_um": 0}
    assert e[1] == {"shank": 2, "electrode": 1, "row": 0, "col": 1, "x_um": 59, "z_um": 0}
    assert e[2] == {"shank": 2, "electrode": 2, "row": 1, "col": 0, "x_um": 27, "z_um": 15}
    assert e[-1]["electrode"] == 1279 and e[-1]["z_um"] == 9585
