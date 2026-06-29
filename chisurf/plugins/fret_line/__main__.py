"""CLI entry point for the FRET Line Generator.

List available component models::

    python -m chisurf.plugins.fret_line --list-models

Compute from a JSON spec file (compute_fret_line() kwargs)::

    python -m chisurf.plugins.fret_line --params spec.json

    # spec.json:
    # {
    #   "components": [
    #     {"model_name": "FRET: FD (Gaussian)", "params": {"R(G,1)": 35, "s(G,1)": 4}},
    #     {"model_name": "FRET: FD (Worm-like chain)", "params": {"l": 80, "lp": 60}}
    #   ],
    #   "sweep": {"kind": "fraction", "component": 1},
    #   "param_min": 0.0, "param_max": 1.0, "n_points": 100,
    #   "fractions": [1.0, 0.0]
    # }

Batch (JSON array of such specs)::

    python -m chisurf.plugins.fret_line --batch batch.json --output results/

Interactive GUI::

    python -m chisurf.plugins.fret_line
"""

from __future__ import annotations

import json
import pathlib
import sys

from .core.algorithms import compute_fret_line, list_models


def main() -> None:
    """CLI entry point — dispatch on argv flags or launch the GUI."""
    _run(sys.argv[1:])


def _run(argv: list[str]) -> None:
    if "--list-models" in argv:
        for m in list_models():
            print(m)
        return

    if "--params" in argv:
        idx = argv.index("--params")
        with open(argv[idx + 1]) as f:
            spec = json.load(f)
        print(json.dumps(compute_fret_line(**spec), indent=2))
        return

    if "--batch" in argv:
        idx = argv.index("--batch")
        with open(argv[idx + 1]) as f:
            batch = json.load(f)
        out_dir = None
        if "--output" in argv:
            out_dir = pathlib.Path(argv[argv.index("--output") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
        for i, spec in enumerate(batch):
            result = compute_fret_line(**spec)
            if out_dir is not None:
                with open(out_dir / f"result_{i:04d}.json", "w") as f:
                    json.dump(result, f, indent=2)
            else:
                print(json.dumps(result, indent=2))
        return

    # No flags → launch the GUI.
    from qtpy import QtWidgets

    from .gui.tool import FRETLineTool

    app = QtWidgets.QApplication(sys.argv)
    win = FRETLineTool()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
