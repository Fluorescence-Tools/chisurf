"""
Progress dialog widgets for chisurf.

This module provides enhanced progress dialog widgets that can be used
throughout the chisurf application for displaying progress information
to the user during long-running operations.
"""

import traceback
import textwrap
from chisurf.gui import QtWidgets, QtCore


class WorkerSignals(QtCore.QObject):
    """
    Defines the signals available from a running worker thread.
    
    Signals:
    --------
    finished: No data
        Signal emitted when the worker has completed its task
    error: tuple (exctype, value, traceback.format_exc())
        Signal emitted when an exception was raised in the worker
    result: object
        Signal emitted with the result of the task
    progress: int
        Signal emitted to indicate task progress (0-100)
    """
    finished = QtCore.Signal()
    error = QtCore.Signal(object)
    result = QtCore.Signal(object)
    progress = QtCore.Signal(int)


class Worker(QtCore.QRunnable):
    """
    Worker thread for running background tasks.
    
    Inherits from QRunnable to handle worker thread setup, signals and wrap-up.
    
    Parameters:
    -----------
    fn : callable
        The function to run on this worker thread. Supplied args and kwargs will be passed
        through to the function.
    *args : list
        Arguments to pass to the function
    **kwargs : dict
        Keywords to pass to the function
    """
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        
    def run(self):
        """
        Initialize the runner function with passed args, kwargs.
        """
        try:
            # Retrieve the function return value
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            # Print the exception information
            traceback_str = traceback.format_exc()
            print(traceback_str)
            # Emit the error signal
            self.signals.error.emit(traceback_str)
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            # Done
            self.signals.finished.emit()


class EnhancedProgressDialog(QtWidgets.QProgressDialog):
    """
    An enhanced progress dialog that can update its label text without user interaction.
    This is used to replace message boxes with progress bar updates.
    """
    def __init__(self, title, label_text, min_value, max_value, parent=None):
        super().__init__(label_text, "Cancel", min_value, max_value, parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModal)
        try:
            # Ensure the widget is destroyed on close to avoid stray windows
            # lingering due to extra references.
            self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        except Exception:
            pass
        self.setMinimumDuration(0)
        self.setAutoClose(False)
        self.setAutoReset(False)

        # QProgressDialog's default label does not wrap long unbroken strings well,
        # which can make the dialog extremely wide (e.g. very long fit names).
        try:
            self._label = QtWidgets.QLabel(self)
            self._label.setWordWrap(True)
            self._label.setTextFormat(QtCore.Qt.PlainText)
            self.setLabel(self._label)
            self._label.setText(str(label_text))
        except Exception:
            self._label = None
        # Internal state to support deferred finalization
        self._pending_auto_close = True
        self._auto_timer = None
        
    def update_text(self, text):
        """Update the label text without closing the dialog"""
        try:
            if getattr(self, "_label", None) is not None:
                self._label.setText(str(text))
            else:
                self.setLabelText(str(text))
        except Exception:
            pass
        QtWidgets.QApplication.processEvents()
        
    def update_progress(self, value, text=None):
        """Update both progress value and optionally the text"""
        if text is not None:
            self.update_text(text)
        self.setValue(value)
        QtWidgets.QApplication.processEvents()

    def finish(self, final_text=None, auto_close=True, wait_for_user=False, close_delay_ms=1500):
        """Finish the progress operation and close/hide the dialog.

        Parameters
        ----------
        final_text : str, optional
            Final text to display before closing.
        auto_close : bool
            True closes the dialog; False hides it.
        wait_for_user : bool
            If True, do not auto-finalize; caller must call :meth:`finalize`.
        close_delay_ms : int
            Delay in milliseconds before finalizing. If 0, finalizes immediately.
            If < 0, does not auto-finalize.
        """

        if final_text is not None:
            self.update_text(final_text)

        try:
            self.setValue(self.maximum())
        except Exception:
            pass
        QtWidgets.QApplication.processEvents()

        # Stop any previous timer.
        try:
            if getattr(self, "_auto_timer", None) is not None:
                self._auto_timer.stop()
                self._auto_timer.deleteLater()
        except Exception:
            pass
        self._auto_timer = None

        self._pending_auto_close = bool(auto_close)

        def _finalize():
            try:
                if self._pending_auto_close:
                    self.close()
                    try:
                        self.deleteLater()
                    except Exception:
                        pass
                else:
                    self.hide()
            except RuntimeError:
                pass
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass

        if wait_for_user or (isinstance(close_delay_ms, int) and close_delay_ms < 0):
            return

        if isinstance(close_delay_ms, int) and close_delay_ms == 0:
            _finalize()
            return

        try:
            self._auto_timer = QtCore.QTimer(self)
            self._auto_timer.setSingleShot(True)
            self._auto_timer.timeout.connect(_finalize)
            self._auto_timer.start(int(close_delay_ms) if close_delay_ms is not None else 1500)
        except Exception:
            _finalize()

    def finalize(self, force_auto_close=None):
        """Finalize immediately by closing or hiding the dialog."""

        try:
            if getattr(self, "_auto_timer", None) is not None:
                self._auto_timer.stop()
                self._auto_timer.deleteLater()
        except Exception:
            pass
        self._auto_timer = None

        auto = self._pending_auto_close if force_auto_close is None else bool(force_auto_close)
        try:
            if auto:
                self.close()
            else:
                self.hide()
        except RuntimeError:
            pass


def wrap_text(text: str, width: int = 48, max_lines: int = 3) -> str:
    """Wrap long text into a small number of lines.

    Intended for UI labels where long strings (e.g. fit names) would otherwise
    expand dialogs horizontally.
    """

    s = "" if text is None else str(text)
    if width <= 0:
        return s
    if max_lines is not None and max_lines <= 0:
        return ""

    lines = textwrap.wrap(
        s,
        width=int(width),
        break_long_words=True,
        break_on_hyphens=True,
    )
    if not lines:
        return ""

    if max_lines is not None and len(lines) > int(max_lines):
        lines = lines[: int(max_lines)]
        if lines:
            last = lines[-1]
            if len(last) >= 3:
                lines[-1] = last[:-3] + "..."
            else:
                lines[-1] = "..."

    return "\n".join(lines)


class ProgressDialog:
    """
    A context manager for progress dialogs.
    
    This class provides a convenient way to use progress dialogs in a with statement.
    It automatically creates and shows the dialog when entering the context,
    and closes it when exiting the context.
    
    Example:
    --------
    with ProgressDialog("Processing", "Processing files...", 0, 100) as progress:
        for i in range(100):
            # Do some work
            progress.update_progress(i, f"Processing file {i}")
            
    # Or with a worker thread:
    with ProgressDialog("Processing", "Processing files...") as progress:
        worker = Worker(my_function, arg1, arg2)
        progress.start_worker(worker)
    """
    def __init__(self, title, label_text, min_value=0, max_value=100, parent=None):
        self.dialog = EnhancedProgressDialog(title, label_text, min_value, max_value, parent)
        self.thread_pool = QtCore.QThreadPool()
        
    def __enter__(self):
        self.dialog.show()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dialog.finish()
        return False  # Don't suppress exceptions
        
    def update_progress(self, value, text=None):
        """Update the progress dialog"""
        self.dialog.update_progress(value, text)
        
    def update_text(self, text):
        """Update the dialog text"""
        self.dialog.update_text(text)
        
    def start_worker(self, worker):
        """
        Start a worker in a background thread.
        
        Parameters:
        -----------
        worker : Worker
            The worker to start
        """
        # Connect worker signals to dialog updates
        worker.signals.progress.connect(self.dialog.setValue)
        
        # Start the worker
        self.thread_pool.start(worker)
