"""
Helper functions for 2D-FLCS wizard.
"""

import numpy as np
from pathlib import Path
from qtpy.QtCore import QRectF
import pyqtgraph as pg

def get_tttrlib_container_type(detector_filetype: str) -> str:
    """Convert detector wizard filetype to proper tttrlib container type."""
    filetype_mapping = {
        'PTU': 'PTU',
        'PT3': 'PT3', 
        'HT3': 'HT3',
        'HDF5': 'HDF5',
        'SPC': 'HDF5',
        'SDT': 'SDT',
        'BIN': 'BIN'
    }
    return filetype_mapping.get(detector_filetype, detector_filetype)

def get_filetype_from_path(file_path: str) -> str:
    """Get filetype from file path using standard extension mapping."""
    suffix = Path(file_path).suffix.lower()
    filetype_map = {
        '.ptu': 'PTU',
        '.pt3': 'PT3', 
        '.ht3': 'HT3',
        '.hdf5': 'HDF5',
        '.h5': 'HDF5',
        '.spc': 'SPC',
        '.sdt': 'SDT',
        '.bin': 'BIN'
    }
    return filetype_map.get(suffix, 'PTU')

def set_plot_image(image_item: pg.ImageItem, matrix, axis_values, log_scale: bool = False, micro_binning_factor: int = 1):
    """Set data on an ImageItem with optional log scaling and axis mapping."""
    if image_item is None:
        return
    
    if matrix is None or getattr(matrix, 'size', 0) == 0:
        image_item.clear()
        return
    
    data = np.array(matrix, copy=False)
    if log_scale:
        data = np.log10(data + 1.0)
    
    image_item.setImage(data, autoLevels=True)
    
    # Set axis scaling if axis values are provided
    if axis_values is not None and len(axis_values) > 1:
        axis = np.array(axis_values)
        step = float(axis[1] - axis[0]) * micro_binning_factor if len(axis) > 1 else micro_binning_factor
        start = float(axis[0]) * micro_binning_factor
    else:
        step = 1.0 * micro_binning_factor
        start = 0.0
    
    width = step * data.shape[1]
    height = step * data.shape[0]
    image_item.setRect(QRectF(start, start, width, height))

def microseconds_to_ticks(value, resolution, allow_zero=False):
    """Convert microseconds to ticks based on instrument resolution."""
    if value == 0 and not allow_zero:
        return 1
    return int(round(value * 1e-6 / resolution))

def nanoseconds_to_ticks(value_ns, resolution):
    """Convert nanoseconds to ticks based on instrument resolution."""
    return int(round(value_ns * 1e-9 / resolution))
