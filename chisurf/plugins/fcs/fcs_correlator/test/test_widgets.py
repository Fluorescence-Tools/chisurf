import pathlib

import pytest
from qtpy import QtWidgets

_SPC = (
    pathlib.Path(__file__).resolve().parents[5]
    / "test" / "data" / "tttr" / "BH" / "132" / "BH_SPC132.spc"
)


def test_fcs_correlator_tool(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_correlator.tool import FcsCorrelatorTool
    from chisurf.gui.widgets.navigation import NavigationPanelTool
    widget = FcsCorrelatorTool()
    qtbot.addWidget(widget)
    assert isinstance(widget, NavigationPanelTool)


@pytest.mark.skipif(not _SPC.exists(), reason="SPC test data not available")
def test_files_carry_over_to_correlator(qapp, qtbot):
    """Selecting files in step 2 must load photon data into the correlator (step 4).

    Regression: the correlator read the container type from a non-existent
    ``filetype`` key (it lives under ``tttr_reading.file_type``) and
    ``tttrlib.TTTR(path, "")`` returns an empty object without raising, so the
    correlator silently received zero photons.
    """
    from chisurf.plugins.fcs.fcs_correlator.tool import FcsCorrelatorTool

    tool = FcsCorrelatorTool()
    qtbot.addWidget(tool)

    tool.nav_list.setCurrentRow(1)  # Files & Steps
    tool._workflow_panels["files"].file_list.add_files([str(_SPC)])

    tool.nav_list.setCurrentRow(3)  # Correlator
    model = tool._correlator_model
    assert model._tttr is not None
    assert len(model._tttr) > 0
