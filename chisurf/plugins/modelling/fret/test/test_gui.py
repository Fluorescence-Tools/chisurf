"""GUI tests for the FRET Docking & Screening plugin and fps.json editor.
"""

from __future__ import annotations

from chisurf.plugins.modelling.fret.gui import FretDockingTool
from chisurf.plugins.modelling.fps_json_editor.gui.editor import FpsJsonEditor


def test_docking_tool_opens(qtbot):
    """The AutoForm docking tool opens and renders its form."""
    window = FretDockingTool()
    qtbot.addWidget(window)
    assert "Docking" in window.windowTitle()
    # AutoForm rendered the view.json into field widgets
    from chisurf.gui.autoform.sections.builtin import ValueWidget
    assert window._form.findChildren(ValueWidget)
    assert window._model.operation == "dock"


def test_fps_json_editor_panels(qtbot):
    """Verify that FpsJsonEditor initializes sub-panels correctly."""
    editor = FpsJsonEditor()
    qtbot.addWidget(editor)

    assert editor.position_panel is not None
    assert editor.distance_panel is not None
    assert editor.flexfit_panel is not None
    assert editor.json_editor is not None

    # Check that positions list is empty initially
    assert editor.position_panel.positions_list.count() == 0
    assert len(editor.positions) == 0


def test_fps_json_editor_payload_roundtrip(qtbot):
    """Test loading and modifying configurations in FpsJsonEditor."""
    editor = FpsJsonEditor()
    qtbot.addWidget(editor)

    payload = {
        "Positions": {
            "D1": {
                "atom_name": "CA",
                "chain_identifier": "A",
                "residue_seq_number": 10,
                "residue_name": "ALA",
                "linker_length": 20.0,
                "linker_width": 1.0,
                "radius1": 3.5,
                "radius2": 0.0,
                "radius3": 0.0,
                "simulation_grid_resolution": 1.5,
                "body_id": 0
            }
        },
        "Distances": {
            "D1_D2": {
                "Forster_radius": 52.0,
                "distance_type": "RDAMean",
                "position1_name": "D1",
                "position2_name": "D2",
                "distance": 45.0,
                "error_neg": 5.0,
                "error_pos": 5.0
            }
        },
        "χ²": {
            "Group1": {
                "distances": ["D1_D2"]
            }
        }
    }

    editor.fps_json_payload = payload

    # Check model contents
    assert "D1" in editor.positions
    assert "D1_D2" in editor.distances
    assert "Group1" in editor.score_sets

    # Check position panel
    assert editor.position_panel.positions_list.count() == 1
    assert editor.position_panel.positions_list.item(0).text() == "D1"

    # Check distance table
    assert editor.distance_panel.distances_table.rowCount() == 1
    assert editor.distance_panel.distances_table.item(0, 0).text() == "D1_D2"
