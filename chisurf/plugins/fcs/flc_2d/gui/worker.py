"""
Worker thread for 2D-FCS calculations.
"""

from qtpy.QtCore import QThread, Signal
from chisurf import logging

try:
    from chisurf.plugins.fcs.fcs_2d.core import TwoDFDCreator
    from chisurf.plugins.fcs.fcs_2d.fit import TwoDMEMFitter
except ImportError:
    # Fallback for direct testing
    try:
        from ..core import TwoDFDCreator
        from ..fit import TwoDMEMFitter
    except ImportError:
        TwoDFDCreator = None
        TwoDMEMFitter = None

class TwoDFCSWorker(QThread):
    """Worker thread for 2D-FCS calculations to prevent UI freezing."""
    
    progress_updated = Signal(float)
    fdc_created = Signal(dict)
    fitting_complete = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        if TwoDFDCreator is None or TwoDMEMFitter is None:
            raise ImportError("Required 2D-FCS modules not available")
        
        self.fdc_creator = TwoDFDCreator()
        self.mem_fitter = TwoDMEMFitter()
        self.task_data = {}
        self.logger = logging.getLogger(__name__)
    
    def set_create_fdc_task(self, **kwargs):
        """Set task for 2D-FDC creation."""
        self.task_data = {'task': 'create_fdc', **kwargs}
    
    def set_fit_mem_task(self, **kwargs):
        """Set task for 2D-MEM fitting."""
        self.task_data = {'task': 'fit_mem', **kwargs}
    
    def run(self):
        """Execute the assigned task."""
        try:
            if self.task_data['task'] == 'create_fdc':
                self._run_create_fdc()
            elif self.task_data['task'] == 'fit_mem':
                self._run_fit_mem()
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _run_create_fdc(self):
        """Run 2D-FDC creation."""
        def progress_callback(progress):
            self.progress_updated.emit(progress)
        
        mat_lin, mat_lin_t, mat_log, mat_log_t = self.fdc_creator.create_2d_fdc(
            progress_callback=progress_callback,
            **{k: v for k, v in self.task_data.items() if k != 'task'}
        )
        
        self.fdc_created.emit({
            'mat_lin': mat_lin,
            'mat_lin_t': mat_lin_t,
            'mat_log': mat_log,
            'mat_log_t': mat_log_t
        })
    
    def _run_fit_mem(self):
        """Run 2D-MEM fitting."""
        def progress_callback(value):
            self.progress_updated.emit(value)
        
        result = self.mem_fitter.fit_2d_mem_wrapper(
            progress_callback=progress_callback,
            **{k: v for k, v in self.task_data.items() if k != 'task' and k != 'progress_callback'}
        )
        self.fitting_complete.emit(result)
