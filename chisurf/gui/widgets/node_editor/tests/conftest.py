import sys
import pathlib

_modules_dir = (
    pathlib.Path(__file__).resolve().parents[4] / "modules"
)
for module in ("chinet",):
    p = str(_modules_dir / module)
    if p not in sys.path:
        sys.path.insert(0, p)
