"""Live-wiring seam between a fit's model and its on-screen editor/plots.

This is the compatibility bridge that lets the new *pure model + AutoModelWidget*
path coexist with the legacy *model-is-a-widget* classes during migration:

* :func:`build_model_editor` returns the editor widget to place in the model
  panel. For a legacy model that is itself a ``QWidget`` it returns the model
  unchanged (identical behaviour); for a pure (Qt-free) model it builds an
  :class:`~chisurf.gui.widgets.models.auto_model_widget.AutoModelWidget`.
* :func:`model_plot_specs` returns the ``(plot_class, options)`` list for the
  fit subwindow, preferring the model's data-driven ``view_spec().plots`` and
  falling back to the legacy ``plot_classes`` attribute.

Because every model registered today is still a widget, both functions take the
legacy branch unless a model opts in by being a pure model — so wiring these in
is behaviour-preserving.
"""
from __future__ import annotations

from qtpy import QtWidgets

from chisurf import logging

#: Attribute under which a pure model caches its on-screen editor widget. Stored
#: in the model's ``__dict__`` (skipped by ``view_spec()`` auto-derive because it
#: is underscore-prefixed, and not pickled because ``Base.__getstate__`` only
#: keeps metadata + name).
_EDITOR_ATTR = "_chisurf_model_editor"


def _is_alive(widget) -> bool:
    """Return True if ``widget`` is a live Qt object (not a deleted C++ shell).

    Clearing the model layout can destroy a cached editor; accessing such a
    dangling wrapper raises ``RuntimeError``. A cheap attribute touch detects it
    across PyQt/PySide without importing binding-specific helpers.
    """
    if widget is None:
        return False
    try:
        widget.objectName()
        return True
    except RuntimeError:
        return False


def build_model_editor(model) -> QtWidgets.QWidget:
    """Return the editor widget for ``model`` (legacy widget or auto-built).

    For a legacy model that is itself a ``QWidget`` the model is returned
    unchanged. For a pure model an :class:`AutoModelWidget` is built once and
    cached on the model (:data:`_EDITOR_ATTR`) so later show/hide/lookup operate
    on the same widget.

    Parameters
    ----------
    model : chisurf.core.models.model.Model
        The fit's model. May be a legacy model-widget or a pure model.

    Returns
    -------
    QtWidgets.QWidget
        ``model`` itself when it is already a widget, otherwise the (cached)
        :class:`AutoModelWidget` bound to it.
    """
    if isinstance(model, QtWidgets.QWidget):
        return model
    existing = getattr(model, _EDITOR_ATTR, None)
    if _is_alive(existing):
        return existing
    from chisurf.gui.widgets.models.auto_model_widget import AutoModelWidget
    widget = AutoModelWidget(model)
    try:
        setattr(model, _EDITOR_ATTR, widget)
    except Exception:  # pragma: no cover - defensive
        pass
    return widget


def model_editor_widget(model):
    """Return the on-screen editor widget for ``model``, or ``None``.

    Legacy widget models *are* their own editor; pure models return the cached
    :class:`AutoModelWidget` if one has been built (via
    :func:`build_model_editor`). Use this where code historically manipulated the
    model widget directly (show/hide/screenshot).
    """
    if isinstance(model, QtWidgets.QWidget):
        return model
    widget = getattr(model, _EDITOR_ATTR, None)
    return widget if _is_alive(widget) else None


def show_model_editor(model) -> None:
    """Show ``model``'s editor widget if it has one (no-op otherwise)."""
    widget = model_editor_widget(model)
    if widget is not None:
        widget.show()


def hide_model_editor(model) -> None:
    """Hide ``model``'s editor widget if it has one (no-op otherwise)."""
    widget = model_editor_widget(model)
    if widget is not None:
        widget.hide()


def model_plot_specs(model):
    """Return ``[(plot_class, options), ...]`` for the fit subwindow.

    Legacy model-widgets keep their own ``plot_classes`` (a hand-written widget
    must not inherit, say, the lifetime view-spec's plots just because it
    multiply-inherits ``LifetimeModel``). Only pure models — the ones rendered by
    :class:`AutoModelWidget` — drive their plots from ``view_spec().plots``.
    """
    if isinstance(model, QtWidgets.QWidget):
        return list(getattr(model, "plot_classes", []))

    try:
        view = model.view_spec()
    except Exception as exc:  # pragma: no cover - defensive
        logging.debug(f"model_plot_specs: view_spec() failed: {exc}")
        view = None

    if view is not None:
        try:
            from chisurf.gui.widgets.models.sections.registry import resolve_plot_specs
            from chisurf.gui.widgets.models.sections.builtin import (
                resolve_distribution_options,
            )
            specs = resolve_plot_specs(view)
            if specs:
                # resolve string accessors (e.g. distribution plots) to callables
                return [
                    (
                        cls,
                        resolve_distribution_options(opts)
                        if "distribution_options" in opts
                        else opts,
                    )
                    for cls, opts in specs
                ]
        except Exception as exc:  # pragma: no cover - defensive
            logging.debug(f"model_plot_specs: resolve failed, using legacy: {exc}")

    return list(getattr(model, "plot_classes", []))
