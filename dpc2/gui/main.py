import sys
import logging
from PyQt5 import QtWidgets
<<<<<<< HEAD
from dpc2.gui.windows.dpc_gui import DiffViewWindow
=======
from dpc2.gui.windows.dpc2_window import dpc2Window
>>>>>>> b821ec8 (minor changes)
from dpc2.gui import UI_DIR

def start_dpc2():
    # configure logging…
    app = QtWidgets.QApplication(sys.argv)

    # Apply stylesheet if exists
    qss_file = UI_DIR / "css" / "uswds.qss"
    if qss_file.exists():
        with open(qss_file, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    win = DiffViewWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    start_dpc2()
