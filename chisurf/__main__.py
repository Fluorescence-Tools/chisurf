from __future__ import annotations

import sys

from chisurf.gui import qt_app


def main():
     app = qt_app()
     sys.exit(app.exec_())


if __name__ == "__main__":
    main()

