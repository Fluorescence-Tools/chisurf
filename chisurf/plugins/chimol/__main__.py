"""ChiMol plugin entry point.

    python -m chisurf.plugins.chimol          # Qt GUI (default)
    python -m chisurf.plugins.chimol cli ...  # headless ptpython REPL
"""

from __future__ import annotations
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].lower() == "cli":
        sys.argv = [sys.argv[0] + " cli"] + args[1:]
        from .chimol.app.cli import main as cli_main
        cli_main()
    else:
        from . import main
        main()
