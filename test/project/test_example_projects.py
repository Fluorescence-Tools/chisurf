"""Tests that the example projects under examples/projects can be loaded.

These tests ensure that the project JSON format used by the example
projects is compatible with the loader and that the project state
(datasets and, where applicable, fits) is restored.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# Ensure chisurf package is on the path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import chisurf as cs  # noqa: E402
import chisurf.gui  # noqa: E402,F401  # ensure cs.gui is a module
import chisurf.gui.widgets  # noqa: E402,F401  # ensure cs.gui.widgets resolves
from chisurf.macros.core_fit import load_project_data  # noqa: E402


EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "examples", "projects")
)


def _reset_state():
    """Reset module-level state to a clean headless baseline."""
    cs.fits.clear()
    cs.imported_datasets.clear()
    try:
        cs.cs = None
    except Exception:
        pass


def test_cs_gui_is_module():
    """`chisurf.gui` must remain a module — code paths like
    `cs.gui.widgets.hide_items_in_layout(...)` rely on it. A
    regression where something assigns a Main instance to
    ``cs.gui`` would break every GUI call site.
    """
    # Sanity: it's a module and exposes the standard submodules
    assert isinstance(cs.gui.__name__, str) and cs.gui.__name__ == "chisurf.gui"
    assert cs.gui.widgets.__name__ == "chisurf.gui.widgets"


def _is_csp_project(path):
    """Check if path is a .csp archive or a directory containing project.csp."""
    if os.path.isfile(path) and path.endswith(".csp"):
        return True
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "project.csp")):
        return True
    return False


def test_t4l_chimol_project_loads():
    """The t4l_chimol example project should load without errors."""
    project_path = os.path.join(EXAMPLES_DIR, "t4l_chimol")
    csp_path = project_path + ".csp"
    if _is_csp_project(csp_path):
        load_path = csp_path
    elif _is_csp_project(project_path):
        load_path = project_path
    elif os.path.isdir(project_path):
        pytest.skip(f"Example project {project_path} has not been converted to .csp format yet")
    else:
        pytest.skip(f"Example project not found: {project_path}")

    _reset_state()
    fit_uids = load_project_data(load_path)
    assert isinstance(fit_uids, list)
    # 0 fits by design
    assert len(cs.fits) == 0
    # Loading must not have corrupted `cs.gui`.
    assert cs.gui.__name__ == "chisurf.gui"


def test_t4l_proteinmc_project_loads():
    """The t4l_proteinmc example project should load and restore its
    ProteinMC fit in headless mode up to the model-resolution step.

    In a headless environment (no QApplication) the ProteinMC fit
    itself is skipped because it requires Qt. The test verifies that
    the surrounding infrastructure (datasets, experiment context,
    history) is restored without errors.
    """
    project_path = os.path.join(EXAMPLES_DIR, "t4l_proteinmc")
    csp_path = project_path + ".csp"
    if _is_csp_project(csp_path):
        load_path = csp_path
    elif _is_csp_project(project_path):
        load_path = project_path
    elif os.path.isdir(project_path):
        pytest.skip(f"Example project {project_path} has not been converted to .csp format yet")
    else:
        pytest.skip(f"Example project not found: {project_path}")

    _reset_state()
    fit_uids = load_project_data(load_path)
    assert isinstance(fit_uids, list)
    # In headless mode the ProteinMC model is not created, so fits is 0.
    # In GUI mode the fit would be added. We only assert no crash here.
    # The dataset list contains the proteinmc input + the global-fit slot.
    assert len(cs.imported_datasets) >= 1
    # Loading must not have corrupted `cs.gui`.
    assert cs.gui.__name__ == "chisurf.gui"


def test_load_project_payload_reinitializes_gui_without_messages():
    """Project loading should not show confirmation or completion dialogs."""
    from chisurf.core.project import Project
    from chisurf.macros.core_fit import load_project_payload

    calls = []

    class FakeGui:
        def reinitialize(self, show_confirmation=True, show_success=True):
            calls.append((show_confirmation, show_success))

        def update(self):
            pass

        dataset_selector = SimpleNamespace(update=lambda: None)
        fit_selector = SimpleNamespace(update=lambda: None)

    saved_cs_cs = getattr(cs, "cs", None)
    try:
        cs.cs = FakeGui()
        load_project_payload(Project(name="test"), project_path=None)
    finally:
        cs.cs = saved_cs_cs

    assert calls == [(False, False)]


def test_reinitialize_application_does_not_overwrite_cs_gui():
    """`reinitialize_application` previously did
    ``setattr(cs, 'gui', main_window)`` which clobbered the
    ``chisurf.gui`` module reference with a Main instance. Every
    later call into ``cs.gui.widgets.*`` then raised
    ``AttributeError: 'Main' object has no attribute 'widgets'``.
    """
    # Pre-import to ensure the package attribute is set
    from chisurf.macros.core_data import reinitialize_application

    class _MockMain:
        def onCloseAllFits(self):
            pass

    # Save and restore cs.cs to avoid polluting subsequent tests
    saved_cs_cs = getattr(cs, "cs", None)
    try:
        # Run the full reinitialize sequence. Internal steps that need
        # a real Main window log a warning instead of crashing.
        reinitialize_application(main_window=_MockMain())

        # The critical assertion: cs.gui is still the module.
        assert cs.gui.__name__ == "chisurf.gui"
        assert cs.gui.widgets.__name__ == "chisurf.gui.widgets"
    finally:
        cs.cs = saved_cs_cs

