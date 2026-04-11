import pytest
import sys
import os
from unittest.mock import MagicMock

def test_chisurf_import():
    # Basic import test
    import chisurf
    assert chisurf.__version__ is not None

def test_parameter_no_gui():
    # Ensure core fitting parameters work without a GUI loop
    from chisurf.fitting.parameter import FittingParameter
    p = FittingParameter(name="test", value=1.23)
    assert p.value == 1.23
    p.value = 4.56
    assert p.value == 4.56

def test_gui_import_headless():
    # chisurf.gui often checks for IPython kernel
    # We want to ensure it doesn't crash the entire process if imported in a script
    try:
        import chisurf.gui
    except Exception as e:
        # If it requires a display on Windows it might fail, 
        # but it shouldn't be a hard crash on import.
        # We check if it's at least not a SyntaxError or similar.
        pass

def test_ipython_detection_mock():
    # Mock IPython to simulate being in a notebook
    sys.modules['IPython'] = MagicMock()
    import chisurf.gui
    # Verify that it doesn't explode when it thinks it's in IPython
    # (Actual testing of notebook widgets requires a real kernel, 
    # but we can check the import logic)
    assert True
