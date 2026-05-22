import pytest
from qtpy.QtWidgets import QApplication
from dpc2.gui.windows.dpc2_window import dpc2Window

@pytest.fixture(scope='module')
def app():
    import sys
    app = QApplication(sys.argv)
    yield app
    app.quit()

def test_window_starts(app):
    window = dpc2Window()
    assert window is not None
    assert window.isVisible() is False  # should not be visible until shown
