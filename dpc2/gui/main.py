import sys
import logging
from PyQt6 import QtWidgets
from .windows.dpc2_window import dpc2Window
from . import UI_DIR

def start_dpc2():
    # configure logging…
    app = QtWidgets.QApplication(sys.argv)

    # Apply stylesheet if exists
    qss_file = UI_DIR / "css" / "uswds.qss"
    if qss_file.exists():
        with open(qss_file, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    win = dpc2Window()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    start_dpc2()
