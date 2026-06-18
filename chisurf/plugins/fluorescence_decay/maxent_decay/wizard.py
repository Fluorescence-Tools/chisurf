"""MaxEnt MEM Wizard."""

from .gui.gui import MaxentDecayWidget

if __name__ == "plugin":
    widget = MaxentDecayWidget()
    widget.show()
