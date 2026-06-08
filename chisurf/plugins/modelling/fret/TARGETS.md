# FRET Modeling Plugin Targets

This document outlines the design and architectural goals of the FRET modeling plugin in ChiSurf.

## Architectural Goals

1. **Clear Separation of Concerns**:
   - **Core Logic**: Reusable, testable, and GUI-independent classes and functions in pure Python (e.g., `av.py`, `docking.py`, `screening.py`, `evaluate.py`, `pair_selection.py`).
   - **Command Line Interface (CLI)**: Implemented using `click` in `cli/main.py` and delegated via `__main__.py`.
   - **Web Interface Compatibility (API)**: Implemented using `fastapi` and `pydantic` in `api/router.py`.
   - **Graphical User Interface (GUI)**: Implemented in `gui/wizard.py` embedding `LabelStructure` and `MolView`/`AVViewer3D`.

2. **Full Compatibility**:
   - Strictly compatible with the legacy C# FPS parameter/label format and OLGA trajectory parameters.
   - Built-in conversion from tab-separated C# parameter text files to standard `fps.json` schema.

3. **High-Performance Pruning**:
   - Fast clash-detection using Numba-based AABB filtering for rigid bodies.

4. **Integrated Example Workflows**:
   - **FPS (4wj docking)**: Multi-body docking of DNA and protein components.
   - **Olga (md traj screening)**: Screening and evaluation of molecular dynamics trajectories.
