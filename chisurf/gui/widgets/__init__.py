from __future__ import annotations
import importlib
import typing

# Avoid monolithic category imports at top-level to support headless operation.
# Widgets should be imported from their specific categories when needed.
# e.g. from chisurf.gui.widgets.experiments.widgets import ...

# List of modules that provide top-level widget attributes.
# We try these in order when an attribute is requested.
_DATA_PROVIDERS = [".general", ".mdi_custom_titlebar"]

# List of known sub-packages in chisurf.gui.widgets
_SUB_PACKAGES = {
    "experiments", "fio", "fitting", "fluorescence", "fortune",
    "node_editor", "parameter_editor", "pdb", "ribbon", "structure", "wizard"
}

def __getattr__(name: str) -> typing.Any:
    """Implement lazy loading for modules and attributes in chisurf.gui.widgets.
    
    This allows accessing common utility functions (like hide_items_in_layout)
    and sub-packages (like experiments) directly from the chisurf.gui.widgets 
    namespace without triggering heavy imports unless they are actually used.
    """
    # 1. Check if it's a known sub-package
    if name in _SUB_PACKAGES:
        return importlib.import_module(f".{name}", __package__)

    # 2. Try to find the attribute in our data providers (general, mdi_custom_titlebar)
    for module_name in _DATA_PROVIDERS:
        try:
            mod = importlib.import_module(module_name, __package__)
            if hasattr(mod, name):
                return getattr(mod, name)
        except (ImportError, AttributeError):
            continue

    # 3. Last resort: try to import it as a submodule directly if it wasn't in _SUB_PACKAGES
    # but might exist (e.g. newly added modules)
    try:
        return importlib.import_module(f".{name}", __package__)
    except ImportError:
        pass

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> typing.List[str]:
    """Provide a complete list of attributes for autocompletion and inspection."""
    attrs = set(globals().keys())
    attrs.update(_SUB_PACKAGES)
    
    # Add exports from data providers
    for module_name in _DATA_PROVIDERS:
        try:
            mod = importlib.import_module(module_name, __package__)
            attrs.update(dir(mod))
        except ImportError:
            continue
            
    return sorted(list(attrs))

# Legacy helper for explicit access if needed
def get_mdi_components():
    from .mdi_custom_titlebar import CustomTitleBar, CustomMdiSubWindow
    return CustomTitleBar, CustomMdiSubWindow

__all__ = [
    "CustomTitleBar",
    "CustomMdiSubWindow",
]
