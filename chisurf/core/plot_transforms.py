from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chisurf import typing


@dataclass(frozen=True)
class PlotReferenceParameter:
    """Editable plot-only parameter for a reference transform.

    Parameters
    ----------
    key : str
        Stable parameter identifier.
    label : str
        Text shown in the plot controller.
    kind : str
        Widget type: ``"float"``, ``"int"``, ``"bool"``, or ``"choice"``.
    default : object
        Initial value used when the mode is selected or reset.
    minimum : float, optional
        Lower bound for numeric parameters.
    maximum : float, optional
        Upper bound for numeric parameters.
    step : float, optional
        Step size for numeric parameters.
    choices : tuple, optional
        Choices for ``"choice"`` parameters. Entries may be plain values or
        ``(value, label)`` pairs.
    """

    key: str
    label: str
    kind: str
    default: typing.Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple = ()


@dataclass(frozen=True)
class PlotReferenceContext:
    """Runtime context passed to a plot reference transform callback.

    Parameters
    ----------
    fit : object
        Fit that owns the currently plotted curve.
    model : object
        Model attached to ``fit``.
    curve_key : str
        Name of the plotted curve.
    x : numpy.ndarray
        Current curve x-values.
    y : numpy.ndarray
        Current curve y-values.
    curves : dict
        All curves returned by the fit for this update.
    group_fits : tuple
        Grouped local fits, if any.
    group_index : int, optional
        Index of ``fit`` inside ``group_fits``.
    selected_group_index : int, optional
        Active local-fit index in grouped display mode.
    parameters : dict
        Plot-controller parameter values for the selected mode.
    """

    fit: typing.Any
    model: typing.Any
    curve_key: str
    x: np.ndarray
    y: np.ndarray
    curves: typing.Mapping[str, typing.Any]
    group_fits: tuple = ()
    group_index: int | None = None
    selected_group_index: int | None = None
    parameters: typing.Mapping[str, typing.Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlotReferenceResult:
    """Output produced by a plot reference transform.

    Parameters
    ----------
    x : numpy.ndarray
        Transformed x-values.
    y : numpy.ndarray
        Transformed y-values.
    visible : bool
        If false, the curve is hidden.
    y_label : str, optional
        Optional y-axis label override.
    """

    x: np.ndarray
    y: np.ndarray
    visible: bool = True
    y_label: str | None = None


@dataclass(frozen=True)
class PlotReferenceMode:
    """Named plot-time transform exposed in the line-plot controller.

    Parameters
    ----------
    key : str
        Stable mode identifier.
    label : str
        Human-readable label.
    callback : callable
        Function called with :class:`PlotReferenceContext`.
    parameters : tuple
        Plot-only parameter declarations.
    applies_to : object, optional
        Curve filter. May be a callable receiving the context, or an iterable
        of exact curve keys.
    y_label : str, optional
        Default y-axis label while this mode is active.
    y_range : tuple, optional
        (ymin, ymax) axis range applied when this mode is active.
    y_padding : float, optional
        Fractional padding added above/below y_range.
        E.g. 0.05 adds 5% of the range span on each side.
    x_range : tuple, optional
        (xmin, xmax) axis range applied when this mode is active.
    x_padding : float, optional
        Fractional padding for x_range.
    """

    key: str
    label: str
    callback: typing.Callable[[PlotReferenceContext], typing.Any]
    parameters: tuple[PlotReferenceParameter, ...] = ()
    applies_to: typing.Any = None
    y_label: str | None = None
    y_range: tuple[float, float] | None = None
    y_padding: float | None = None
    x_range: tuple[float, float] | None = None
    x_padding: float | None = None

    def applies(self, context: PlotReferenceContext) -> bool:
        """Return whether this mode should be applied to ``context``.

        Parameters
        ----------
        context : PlotReferenceContext
            Current transform context.

        Returns
        -------
        bool
            True when the callback should run.
        """
        if self.applies_to is None:
            return True
        if callable(self.applies_to):
            return bool(self.applies_to(context))
        try:
            return context.curve_key in set(self.applies_to)
        except Exception:
            return False

