"""GUI-side registry that resolves view-spec keys to concrete widgets/plots.

A model describes its editor with string keys (see
:mod:`chisurf.core.models.view_spec`) and never references a Qt class. This
module is the GUI-side counterpart: it maps those keys to real plot classes and
custom-section factories. Keeping the mapping here — and only here — is what
lets the model layer stay GUI-free while still driving a rich UI.
"""
from __future__ import annotations

from chisurf import typing

#: key -> plot class (resolved lazily so importing this module is cheap).
_PLOT_REGISTRY: typing.Dict[str, typing.Callable[[], type]] = {}
#: key -> custom-section factory ``(model, target, **options) -> QWidget``.
_SECTION_REGISTRY: typing.Dict[str, typing.Callable[..., typing.Any]] = {}


def register_plot(key: str, factory: typing.Callable[[], type]) -> None:
    """Register a plot class factory under ``key``.

    Parameters
    ----------
    key : str
        The key emitted by a model's :class:`~chisurf.core.models.view_spec.PlotSpec`.
    factory : callable
        Zero-argument callable returning the plot *class*. A factory (rather
        than the class directly) avoids importing the GUI plots eagerly.
    """
    _PLOT_REGISTRY[key] = factory


def get_plot_class(key: str) -> typing.Optional[type]:
    """Return the plot class registered under ``key`` (or ``None``)."""
    factory = _PLOT_REGISTRY.get(key)
    return factory() if factory is not None else None


def register_section(key: str):
    """Decorator registering a custom-section widget factory under ``key``.

    The decorated callable is invoked as ``factory(model=..., target=..., **options)``
    and must return a ``QWidget``.
    """

    def _decorator(factory: typing.Callable[..., typing.Any]):
        _SECTION_REGISTRY[key] = factory
        return factory

    return _decorator


def get_section_factory(key: str) -> typing.Optional[typing.Callable[..., typing.Any]]:
    """Return the custom-section factory registered under ``key`` (or ``None``)."""
    return _SECTION_REGISTRY.get(key)


def resolve_plot_specs(view) -> typing.List[typing.Tuple[type, dict]]:
    """Translate a :class:`ModelView` into legacy ``plot_classes`` tuples.

    Returns a list of ``(plot_class, options)`` pairs compatible with the
    existing fit-subwindow plot loader, dropping any keys that are not
    registered. This is the compatibility bridge that lets view-spec plots flow
    through the current plotting code unchanged.
    """
    specs: typing.List[typing.Tuple[type, dict]] = []
    for plot in getattr(view, "plots", ()):  # PlotSpec items
        plot_class = get_plot_class(plot.key)
        if plot_class is not None:
            specs.append((plot_class, dict(plot.options)))
    return specs


@register_section("fitting_parameter")
def _fitting_parameter_section_factory(model, target: str, **opts):
    """Render a single FittingParameter as a FittingParameterWidget."""
    from chisurf.gui.widgets.fitting.parameter_widgets import FittingParameterWidget
    fp = getattr(model, target)
    return FittingParameterWidget(fitting_parameter=fp, **opts)
