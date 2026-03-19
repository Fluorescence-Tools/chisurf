import logging

logging.basicConfig(level=logging.DEBUG)

import logging

# Avoid top-level GUI imports here to support headless operation of submodules (model, graph, registry).
# Components should be imported from their respective submodules:
# from chisurf.gui.widgets.node_editor.ui import apply_node_ui_theme, ...
# from chisurf.gui.widgets.node_editor.node_editor import NodeEditorWidget

__all__ = [
    "apply_node_ui_theme",
    "StyledComboBox",
    "InlineLabeledSlider",
    "Vector1DWidget",
    "NodeEditorWidget",
]
