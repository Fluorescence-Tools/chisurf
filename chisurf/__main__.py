from __future__ import annotations

import sys
from chisurf.gui import get_app


def main():
    app = get_app()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
