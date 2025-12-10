"""MaxEnt TCSPC lifetime MEM (development plugin)."""

from __future__ import annotations

name = "Fluorescence decay: MaxEnt lifetime MEM"
# Expose the Click CLI via chisurf.cli under the "maxent-decay" subcommand,
# implemented in the fmem namespace.
cli_entrypoint = "maxent-decay=chisurf.plugins.fluorescence_decay.maxent_decay.fmem.cli:cli"


def load():
    """Return the plugin's main widget instance."""
    from .gui import MaxentDecayWidget

    return MaxentDecayWidget()


def _bootstrap_plugin() -> None:
    from qtpy import QtWidgets, QtCore  # type: ignore

    try:
        import chisurf  # type: ignore
        parent = getattr(chisurf, "cs", None)
    except Exception:
        parent = None

    widget = load()
    if parent is not None and hasattr(parent, "addDockWidget"):
        try:
            dock = QtWidgets.QDockWidget("MaxEnt lifetime MEM", parent)
            dock.setWidget(widget)
            # Give the dock a generous initial width so that the plots
            # are clearly visible; users can still resize it afterwards.
            try:
                dock.resize(1100, 700)
            except Exception:
                pass
            parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            dock.show()
            return
        except Exception:
            pass

    widget.show()


if __name__ == "plugin":  # pragma: no cover - GUI bootstrap
    _bootstrap_plugin()


__all__ = ["name", "cli_entrypoint", "load"]
