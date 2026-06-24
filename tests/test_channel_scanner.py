import json
from napari_dmc_segload.channel_scanner import ChannelScanner


def test_available_channels_intersects_folders_and_params(animal_dir):
    chans = ChannelScanner(animal_dir).available_channels()
    # dapi + cy3 have tifs and are in chans_imaged; green is imaged but has no
    # folder; n3 folder is empty -> excluded. Canonical order preserved.
    assert chans == ["dapi", "cy3"]


def test_empty_channel_folder_excluded(animal_dir):
    assert "n3" not in ChannelScanner(animal_dir).available_channels()


def test_params_missing_falls_back_to_folder_scan(animal_dir):
    (animal_dir / "params.json").unlink()
    chans = ChannelScanner(animal_dir).available_channels()
    assert chans == ["dapi", "cy3"]


def test_params_found_flag_reflects_params_presence(animal_dir):
    scanner = ChannelScanner(animal_dir)
    scanner.available_channels()
    assert scanner.params_found is True
    (animal_dir / "params.json").unlink()
    scanner_no_params = ChannelScanner(animal_dir)
    scanner_no_params.available_channels()
    assert scanner_no_params.params_found is False


def test_no_single_channel_dir_returns_empty(tmp_path):
    assert ChannelScanner(tmp_path).available_channels() == []
