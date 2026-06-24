import json
from pathlib import Path
from typing import List, Optional

from .constants import CANONICAL_CHANNELS


class ChannelScanner:
    """Detect which channels are available for an animal folder."""

    def __init__(self, input_path: Path):
        self.input_path = Path(input_path)
        # Set by available_channels(): False when params.json was missing/invalid
        # and the result fell back to a folder-only scan.
        self.params_found = False

    def _folders_with_tifs(self) -> List[str]:
        sc_dir = self.input_path / "single_channel"
        if not sc_dir.is_dir():
            return []
        found = []
        for sub in sc_dir.iterdir():
            if sub.is_dir() and any(sub.glob("*.tif")):
                found.append(sub.name)
        return found

    def _imaged_channels(self) -> Optional[List[str]]:
        params_fn = self.input_path / "params.json"
        if not params_fn.exists():
            return None
        try:
            data = json.loads(params_fn.read_text())
            return data["general"]["chans_imaged"]
        except (json.JSONDecodeError, KeyError):
            return None

    def available_channels(self) -> List[str]:
        folders = set(self._folders_with_tifs())
        imaged = self._imaged_channels()
        self.params_found = imaged is not None
        if imaged is not None:
            folders &= set(imaged)
        return [c for c in CANONICAL_CHANNELS if c in folders]
