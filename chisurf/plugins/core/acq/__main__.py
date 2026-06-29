"""Module entry point for the SM Acquisition plugin.

This allows running the plugin as a module, e.g.::

    python -m chisurf.plugins.core.acq
    python -m chisurf.plugins.core.acq --cli config ...

Dispatch is handled by ``standalone.main()``, which supports both GUI and
CLI modes.
"""

from .standalone import main


if __name__ == "__main__":  # pragma: no cover - executed via -m
    import sys
    sys.exit(main())
