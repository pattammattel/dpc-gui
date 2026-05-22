from functools import wraps
from qtpy.QtCore import QObject, QThread, pyqtSignal, Qt
from qtpy.QtWidgets import QProgressDialog, QMessageBox, QApplication


class Worker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(Exception)

    def __init__(self, func, args, kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(e)


def with_busy_popup_threaded(
    title="Processing",
    message="Please wait while the operation completes...",
    success_msg=None,
    error_msg="An error occurred."
):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create and show progress dialog
            progress = QProgressDialog(message, None, 0, 0, self)
            progress.setWindowTitle(title)
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            # Create thread and worker
            self._thread = QThread()
            self._worker = Worker(func, (self, *args), kwargs)
            self._worker.moveToThread(self._thread)

            # Handlers
            def on_success(result):
                progress.close()
                self._thread.quit()
                self._thread.wait()
                if success_msg:
                    QMessageBox.information(self, title, success_msg)
                self._thread = None
                self._worker = None
                wrapper._result = result

            def on_error(error):
                progress.close()
                self._thread.quit()
                self._thread.wait()
                QMessageBox.critical(
                    self, title, f"{error_msg}\n\nError: {type(error).__name__}: {str(error)}"
                )
                self._thread = None
                self._worker = None

            # Connect signals
            self._worker.finished.connect(on_success)
            self._worker.error.connect(on_error)
            self._thread.started.connect(self._worker.run)
            self._thread.finished.connect(self._thread.deleteLater)

            # Start thread
            self._thread.start()

        return wrapper
    return decorator



def with_busy_popup(
    title="Processing",
    message="Please wait while the operation completes...",
    success_msg=None,
    error_msg="An error occurred during processing."
):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            progress = QProgressDialog(message, None, 0, 0, self)
            progress.setWindowTitle(title)
            progress.setWindowModality(Qt.NonModal)
            progress.setCancelButton(None)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()

            QApplication.processEvents()  # Allow the dialog to display

            try:
                result = func(self, *args, **kwargs)
                if success_msg:
                    QMessageBox.information(self, title, success_msg)
                return result
            except Exception as e:
                QMessageBox.critical(self, title, f"{error_msg}\n\n{str(e)}")
                raise
            finally:
                progress.close()
                QApplication.processEvents()
        return wrapper
    return decorator
