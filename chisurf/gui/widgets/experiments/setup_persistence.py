from __future__ import annotations
import chisurf as cs

import json
import pathlib
from typing import Any, Dict, Optional, List

import numpy as np

from chisurf.core.settings.path_utils import get_path


SCHEMA_VERSION = 1
SETUP_DEFAULTS_FILENAME = "setup_defaults.json"


def _get_setup_defaults_path() -> pathlib.Path:
    """Return the path to the setup defaults file."""
    settings_dir = get_path('settings')
    return settings_dir / SETUP_DEFAULTS_FILENAME


def _is_basic(v: Any) -> bool:
    """Check if a value is a basic Python type."""
    return isinstance(v, (str, int, float, bool, type(None)))


def _to_basic(v: Any) -> Any:
    """Convert a value to a basic JSON-serializable type."""
    if _is_basic(v):
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return [v.item()]
        return v.tolist()
    if isinstance(v, (list, tuple)):
        out = []
        for item in v:
            if _is_basic(item) or isinstance(item, (np.integer, np.floating)):
                out.append(_to_basic(item))
            elif isinstance(item, np.ndarray):
                out.append(item.tolist())
            else:
                continue
        return out
    return None


def serialize_reader_state(reader: Any) -> Optional[Dict[str, Any]]:
    """Serialize an ExperimentReader's state to a JSON-friendly dict.
    
    This captures only the basic attributes needed for restoring defaults.
    Skip: experiment refs, controllers, Qt widgets, caches, and internal
    state that should not be persisted as defaults.
    
    Args:
        reader: An ExperimentReader instance
        
    Returns:
        Dict with module, class, and state keys, or None if not a valid reader
    """
    from chisurf.core.experiments.core.reader import ExperimentReader
    
    if not isinstance(reader, ExperimentReader):
        return None

    state: Dict[str, Any] = {}
    # Keys that should never be persisted as defaults (they're runtime/internal)
    banned_keys = {
        "experiment", "_experiment", "controller", "_readers", "setup",
        "_cache", "_cached", "_cache_filename", "_cache_ics_stack",
        "_micro_time_histogram", "_micro_time_range_histogram",
        "_photon_indices", "_routes", "_tti_record",
    }
    # Keys that start with underscore except these specific ones
    allowed_underscore = {"_irf"}
    
    for k, v in getattr(reader, "__dict__", {}).items():
        if k in banned_keys:
            continue
        if k.startswith("_") and k not in allowed_underscore:
            continue
        basic = _to_basic(v)
        if basic is not None:
            state[k] = basic

    return {
        "module": type(reader).__module__,
        "class": type(reader).__name__,
        "state": state,
    }


def deserialize_reader_state(
    reader_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract just the state dict from serialized reader info.
    
    This is a simpler counterpart to serialize_reader_state - it returns
    the state dict that can be applied to a reader instance.
    
    Args:
        reader_info: Dict with module, class, and state keys
        
    Returns:
        The state dict, or empty dict if invalid
    """
    if not isinstance(reader_info, dict):
        return {}
    return reader_info.get("state", {})


def apply_reader_state(reader: Any, state: Dict[str, Any]) -> None:
    """Apply a state dict to a reader instance.
    
    This sets attributes directly on the reader object. It does NOT
    trigger UI updates - caller is responsible for that.
    
    Args:
        reader: An ExperimentReader instance
        state: Dict of attribute names to values
    """
    from chisurf.core.experiments.core.reader import ExperimentReader
    
    if not isinstance(reader, ExperimentReader):
        return
    
    for k, v in state.items():
        try:
            setattr(reader, k, v)
        except Exception:
            pass


def load_setup_defaults() -> Dict[str, Any]:
    """Load setup defaults from the user settings file.
    
    Returns:
        Dict with schema version, last selection, and per-experiment defaults
    """
    path = _get_setup_defaults_path()
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "experiments": {}}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate schema version
        schema = data.get("schema_version", 0)
        if schema != SCHEMA_VERSION:
            # Could add migration logic here if needed
            return {"schema_version": SCHEMA_VERSION, "experiments": {}}
        return data
    except Exception:
        return {"schema_version": SCHEMA_VERSION, "experiments": {}}


def save_setup_defaults(data: Dict[str, Any]) -> bool:
    """Save setup defaults to the user settings file.
    
    This performs an atomic write using a temp file + rename.
    
    Args:
        data: Dict with schema version and experiment defaults
        
    Returns:
        True if successful, False otherwise
    """
    path = _get_setup_defaults_path()
    data["schema_version"] = SCHEMA_VERSION
    
    # Ensure parent dir exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Atomic write: write to temp, then rename
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        tmp_path.replace(path)
        return True
    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return False


def get_current_experiment_index(gui: Any) -> int:
    """Get the current experiment index from the main window."""
    try:
        return int(getattr(gui, "current_experiment_idx", 0))
    except Exception:
        return 0


def get_current_setup_index(gui: Any) -> int:
    """Get the current setup index from the main window."""
    try:
        return int(getattr(gui, "current_setup_idx", 0))
    except Exception:
        return 0


def collect_setup_defaults(gui: Any) -> Dict[str, Any]:
    """Collect current setup defaults from all readers.
    
    This serializes the state of all readers in all experiments and
    also captures the current experiment/setup selection indices.
    
    Args:
        gui: The main window's gui object
        
    Returns:
        Dict with experiments dict containing per-experiment defaults
    """
    result = {
        "schema_version": SCHEMA_VERSION,
        "last_selection": {
            "experiment_index": get_current_experiment_index(gui),
            "setup_index": get_current_setup_index(gui),
        },
        "experiments": {},
    }
    
    try:
        experiments = cs.experiment
        if not experiments:
            return result
        
        for exp_name, exp in experiments.items():
            readers = getattr(exp, "readers", []) or []
            if not readers:
                continue
            
            exp_defaults = {}
            for idx, reader in enumerate(readers):
                if reader is None:
                    continue
                reader_state = serialize_reader_state(reader)
                if reader_state:
                    # Use reader name as key, fall back to index
                    key = getattr(reader, "name", None) or str(idx)
                    exp_defaults[key] = reader_state
            
            if exp_defaults:
                result["experiments"][exp_name] = exp_defaults
                
    except Exception:
        pass
    
    return result


def apply_setup_defaults(gui: Any, defaults: Dict[str, Any]) -> None:
    """Apply saved setup defaults to all readers.
    
    This applies the serialized state to each matching reader and then
    triggers a UI sync. Callers should call this after readers are created
    but before showing the window.
    
    Args:
        gui: The main window's gui object
        defaults: Dict with experiments and last_selection
    """
    experiments = defaults.get("experiments", {})
    if not experiments:
        return
    
    try:
        exp_dict = cs.experiment
    except Exception:
        exp_dict = {}
    
    for exp_name, exp_defaults in experiments.items():
        exp = exp_dict.get(exp_name)
        if exp is None:
            continue
        
        readers = getattr(exp, "readers", []) or []
        for key, reader_info in exp_defaults.items():
            # Find matching reader by name or index
            reader = None
            # First try by name
            for r in readers:
                if r is not None and getattr(r, "name", None) == key:
                    reader = r
                    break
            # Then try by index
            if reader is None:
                try:
                    idx = int(key)
                    if 0 <= idx < len(readers):
                        reader = readers[idx]
                except (ValueError, TypeError):
                    pass
            
            if reader is None:
                continue
            
            state = deserialize_reader_state(reader_info)
            if state:
                apply_reader_state(reader, state)
    
    # Restore last selection if present
    last_selection = defaults.get("last_selection", {})
    exp_idx = last_selection.get("experiment_index", 0)
    setup_idx = last_selection.get("setup_index", 0)
    
    try:
        total_exp = gui.comboBox_experimentSelect.count()
        if 0 <= exp_idx < total_exp:
            gui.set_current_experiment_idx(exp_idx)
    except Exception:
        pass
    
    try:
        total_setup = gui.comboBox_setupSelect.count()
        if 0 <= setup_idx < total_setup:
            gui.set_current_setup_idx(setup_idx)
    except Exception:
        pass
    
    # Sync UI from the reader state
    try:
        controller = gui.current_setup
        if controller is not None:
            # If controller has updateUI, call it with signals blocked
            if hasattr(controller, "updateUI"):
                # Import and use signal blocker
                from chisurf.gui.widgets.experiments.ui_sync import block_signals
                with block_signals(controller):
                    controller.updateUI()
    except Exception:
        pass


__all__ = [
    "SCHEMA_VERSION",
    "SETUP_DEFAULTS_FILENAME",
    "serialize_reader_state",
    "deserialize_reader_state",
    "apply_reader_state",
    "load_setup_defaults",
    "save_setup_defaults",
    "collect_setup_defaults",
    "apply_setup_defaults",
]
