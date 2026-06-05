from __future__ import annotations

from typing import Optional

from qtpy import QtWidgets

def open_forster_calculator(owner: QtWidgets.QWidget) -> None:
    """Open the Spectra Viewer plugin to calculate Förster radius."""
    if not hasattr(owner, '_forster_calculator_window'):
        try:
            from chisurf.plugins._dev.spectra_viewer import SpectraViewerWidget
            # The SpectraViewerWidget requires an empty constructor, and creates its own window
            owner._forster_calculator_window = SpectraViewerWidget()
            
            # TODO: needs docstring
            def on_forster_radius_calculated(r0_angstrom: float):
                """Handle calculated Forster radius from plugin."""
                try:
                    fret_params = getattr(owner, "fret_parameters", None)
                    if fret_params is not None:
                        # Set the Forster radius on the parameter
                        fret_params.forster_radius = float(r0_angstrom)
                        
                        # Try to update the GUI widget if possible
                        try:
                            param = getattr(fret_params, "_forster_radius", None)
                            controller = getattr(param, "controller", None) if param is not None else None
                            widget_value = getattr(controller, "widget_value", None) if controller is not None else None
                            if widget_value is not None:
                                try:
                                    widget_value.blockSignals(True)
                                    widget_value.setValue(float(r0_angstrom))
                                finally:
                                    widget_value.blockSignals(False)
                        except Exception:
                            pass
                            
                        # Update the model
                        update = getattr(owner, "update", None)
                        if callable(update):
                            try:
                                update()
                            except Exception:
                                pass
                except Exception as ex:
                    import logging
                    logging.warning(f"Failed to populate Forster radius: {ex}")
            
            from qtpy import QtCore
            owner._forster_calculator_window.forster_radius_calculated.connect(
                on_forster_radius_calculated, type=QtCore.Qt.UniqueConnection
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                owner,
                "Error",
                f"Could not load Spectra Viewer plugin:\n{e}"
            )
            return
            
    # Show and bring to front
    owner._forster_calculator_window.show()
    owner._forster_calculator_window.raise_()
    owner._forster_calculator_window.activateWindow()


