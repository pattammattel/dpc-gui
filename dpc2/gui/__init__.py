from pathlib import Path

# Base directory of this package: .../dpc2/gui
_GUI_DIR = Path(__file__).resolve().parent

# Where all the .ui files live:
UI_DIR = _GUI_DIR / "layout"

# (Optionally) where resources (images, css) live:
RESOURCES_DIR = _GUI_DIR / "resources"

# Known mapping of metadata detector names → data keys
DETECTOR_DATA_KEY_MAP = {
    "eiger2": "eiger2_image",
    "merlin1": "merlin1", 
     "merlin2": "merlin2", # same in both
    "xspress3": "xspress3",      # same in both
    # add more as needed
}
