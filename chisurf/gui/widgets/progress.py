"""
Progress dialog widgets for chisurf.

This module provides enhanced progress dialog widgets that can be used
throughout the chisurf application for displaying progress information
to the user during long-running operations.
"""

import traceback
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
        self.setMinimumDuration(0)
        self.setAutoClose(False)
        self.setAutoReset(False)
        # Internal state to support deferred finalization
        self._pending_auto_close = True
        self._auto_timer = None
        
    def update_text(self, text):
        """Update the label text without closing the dialog"""
        self.setLabelText(text)
        QtWidgets.QApplication.processEvents()
        
    def update_progress(self, value, text=None):
        """Update both progress value and optionally the text"""
        if text is not None:
            self.update_text(text)
        self.setValue(value)
        QtWidgets.QApplication.processEvents()
        
    def finish(self, final_text=None, auto_close=True, wait_for_user=False, close_delay_ms=1500):
        """
        Finish the progress operation with optional final text.
        By default, automatically closes or hides the dialog after a short delay.
        
        Parameters:
        -----------
        final_text : str, optional
            Final text to display before closing
        auto_close : bool, optional
            Whether to close (True) or hide (False) the dialog when finalized
        wait_for_user : bool, optional
            If True, do not auto-finalize. Caller must invoke finalize() later (e.g., on Save).
        close_delay_ms : int, optional
            Delay in milliseconds before auto-finalizing. Ignored when wait_for_user is True.
        """
        if final_text is not None:
            self.update_text(final_text)
        
        # Set to maximum value to indicate completion
        self.setValue(self.maximum())
        QtWidgets.QApplication.processEvents()
        
        # Always stop any previous timer
        try:
            if hasattr(self, "_auto_timer") and self._auto_timer is not None:
                self._auto_timer.stop()
                self._auto_timer.deleteLater()
        except Exception:
            pass
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setSingleShot(True)
        
        # Remember desired final action for potential manual finalize
        self._pending_auto_close = bool(auto_close)

        def _finalize():
            try:
                if self._pending_auto_close:
                    self.close()
                else:
                    self.hide()
            except RuntimeError:
                # The dialog may already be deleted; ignore safely
                pass

        # Ensure the timer does not outlive the dialog
        self.destroyed.connect(lambda *_: self._auto_timer.stop())
        self._auto_timer.timeout.connect(_finalize)
        
        if wait_for_user or (isinstance(close_delay_ms, int) and close_delay_ms < 0):
            # Defer finalization until caller explicitly requests it
            return
        
        # Start auto-finalization timer
        self._auto_timer.start(int(close_delay_ms) if close_delay_ms is not None else 1500)

    def finalize(self, force_auto_close=None):
        """Finalize immediately by closing or hiding the dialog.
        Parameters
        ----------
        force_auto_close : Optional[bool]
            If provided, overrides stored behavior. True=close, False=hide.
        """
        # Stop any pending timer
        try:
            if hasattr(self, "_auto_timer") and self._auto_timer is not None:
                self._auto_timer.stop()
                self._auto_timer.deleteLater()
        except Exception:
            pass
        # Decide action
        auto = self._pending_auto_close if force_auto_close is None else bool(force_auto_close)
        try:
            if auto:
                self.close()
            else:
                self.hide()
        except RuntimeError:
            pass


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