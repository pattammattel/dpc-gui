from pathlib import Path

# Base directory of this package: .../dpc2/gui
_GUI_DIR = Path(__file__).resolve().parent

# Where all the .ui files live:
UI_DIR = _GUI_DIR / "layout"

# (Optionally) where resources (images, css) live:
RESOURCES_DIR = _GUI_DIR / "resources"
