from __future__ import annotations

import sys

if __name__ == "__main__":
    from chisurf.gui import get_app
    app = get_app()
    sys.exit(app.exec_())
