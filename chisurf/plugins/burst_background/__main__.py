"""Entry point for running the Burst Background Estimation plugin.

This allows the plugin to be launched with::

    python -m chisurf.plugins.burst_background
"""

import sys

from PyQt5.QtWidgets import QApplication

from . import BurstBackgroundEstimator


def main() -> None:
    app = QApplication(sys.argv)
    window = BurstBackgroundEstimator()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover - manual GUI entry
    main()
