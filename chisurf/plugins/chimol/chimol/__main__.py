"""ChiMol package entry point.

Supports two run modes:

    python -m chisurf.plugins.chimol          # Qt GUI (default)
    python -m chisurf.plugins.chimol cli ...  # headless ptpython REPL
"""

from __future__ import annotations

import sys


def _dispatch() -> None:
    args = sys.argv[1:]

    if args and args[0].lower() == "cli":
        # Strip the "cli" subcommand so argparse in cli.main() sees clean args
        sys.argv = [sys.argv[0] + " cli"] + args[1:]
        from .app.cli import main as cli_main
        cli_main()
    else:
        from .. import main as gui_main
        gui_main()


if __name__ == "__main__":
    _dispatch()
