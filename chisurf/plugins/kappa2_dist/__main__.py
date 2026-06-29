"""
Entry points for the Kappa2 Distribution tool.

CLI usage
---------

Single computation from JSON parameter file::

    python -m chisurf.plugins.kappa2_dist --params params.json

Batch computation from a JSON array of parameter sets::

    python -m chisurf.plugins.kappa2_dist --batch batch.json --output results/

Interactive GUI::

    python -m chisurf.plugins.kappa2_dist
"""

from __future__ import annotations

import json
import pathlib
import sys

from .core.algorithms import compute_kappa2_dist


def main() -> None:
    _run(sys.argv[1:])


def _run(argv: list[str]) -> None:
    if "--params" in argv:
        idx = argv.index("--params")
        param_path = pathlib.Path(argv[idx + 1])
        with open(param_path) as f:
            params = json.load(f)
        result = compute_kappa2_dist(**params)
        print(json.dumps(result, indent=2))
        return

    if "--batch" in argv:
        idx = argv.index("--batch")
        batch_path = pathlib.Path(argv[idx + 1])
        with open(batch_path) as f:
            batch = json.load(f)
        out_dir = None
        if "--output" in argv:
            out_idx = argv.index("--output")
            out_dir = pathlib.Path(argv[out_idx + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
        for i, params in enumerate(batch):
            result = compute_kappa2_dist(**params)
            if out_dir is not None:
                out_path = out_dir / f"result_{i:04d}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
            else:
                print(json.dumps(result, indent=2))
        return

    # No CLI flags → launch the GUI.
    from qtpy import QtWidgets
    from .k2dgui import Kappa2Dist

    app = QtWidgets.QApplication(sys.argv)
    win = Kappa2Dist()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
