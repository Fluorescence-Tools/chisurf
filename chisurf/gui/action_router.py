from __future__ import annotations
import typing
import chisurf.logging

if typing.TYPE_CHECKING:
    from qtpy import QtWidgets, QtCore

class ActionRouter:
    """
    A toolkit-agnostic router that connects UI elements to the Action Core.
    
    This class defines the interface for mapping UI events (clicks, triggers, 
    value changes) to chisurf actions. By using this router instead of 
    direct signal connections, we decouple the application logic from the 
    specific UI toolkit (e.g. PyQt).
    """
    
    def __init__(self):
        self._connections: typing.List[typing.Tuple[typing.Any, str, typing.Dict]] = []

    def connect(
        self, 
        ui_element: typing.Any, 
        action_name: str, 
        payload: typing.Optional[typing.Union[typing.Dict, typing.Callable]] = None,
        **kwargs
    ):
        """
        Connect a UI element to an action.
        
        :param ui_element: The toolkit-specific UI element (e.g. QAction, QPushButton).
        :param action_name: The name of the action in the Action Registry.
        :param payload: A dictionary of static parameters or a callable that 
                        returns a dictionary of dynamic parameters for the action.
        """
        raise NotImplementedError("Subclasses must implement connect()")

    def dispatch(self, action_name: str, payload: typing.Optional[typing.Dict] = None):
        """Utility to dispatch an action via the central Action Core."""
        import chisurf.core.actions
        try:
            chisurf.core.actions.dispatch(name=action_name, payload=payload or {})
        except Exception as e:
            chisurf.logging.error(f"Router failed to dispatch '{action_name}': {e}")


class QtActionRouter(ActionRouter):
    """
    Implementation of ActionRouter for the Qt toolkit (PyQt/PySide).
    """
    
    def connect(
        self, 
        ui_element: typing.Any, 
        action_name: str, 
        payload: typing.Optional[typing.Union[typing.Dict, typing.Callable]] = None,
        **kwargs
    ):
        from qtpy import QtWidgets, QtCore
        
        # Determine the appropriate signal based on the widget type
        if isinstance(ui_element, QtWidgets.QAction):
            signal = ui_element.triggered
        elif isinstance(ui_element, (QtWidgets.QPushButton, QtWidgets.QToolButton)):
            signal = ui_element.clicked
        elif isinstance(ui_element, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
            signal = ui_element.toggled
        elif hasattr(ui_element, 'editingFinished'):
            signal = ui_element.editingFinished
        else:
            chisurf.logging.warning(f"QtActionRouter: Unknown UI element type {type(ui_element)}. Using default 'triggered' if available.")
            signal = getattr(ui_element, 'triggered', None)
            
        if signal is None:
            chisurf.logging.error(f"QtActionRouter: Could not find a default signal for {ui_element}")
            return

        # Create a wrapper slot that resolves the payload and dispatches the action
        def slot(*args):
            resolved_payload = {}
            if payload is not None:
                if callable(payload):
                    try:
                        resolved_payload = payload(*args)
                    except Exception as e:
                        chisurf.logging.error(f"Error resolving dynamic payload for '{action_name}': {e}")
                        return
                else:
                    resolved_payload = payload
            
            self.dispatch(action_name, resolved_payload)

        signal.connect(slot)
        self._connections.append((ui_element, action_name, kwargs))
