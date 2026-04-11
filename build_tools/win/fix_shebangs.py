import inspect
import json
import os
import os.path
import sys
import time
from pathlib import Path

from pip._vendor.distlib.scripts import ScriptMaker

if os.name != 'nt':
    raise OSError('Launcher regeneration is only supported on Windows')

CURRENT_DIR = Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
SCRIPTS_DIR = CURRENT_DIR / 'Scripts'
ENTRY_POINTS_FILE = CURRENT_DIR / 'entry_points.json'
PYTHON_EXE = CURRENT_DIR / 'python.exe'

sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))


def _open_log():
    try:
        return open(CURRENT_DIR / 'install.log', 'w', encoding='utf-8')
    except PermissionError:
        class Log:
            write = staticmethod(print)

        return Log()


def _load_entry_points():
    with open(ENTRY_POINTS_FILE, 'r', encoding='utf-8') as fp:
        payload = json.load(fp)
    return payload.get('entry_points', [])


def _remove_existing_wrappers(name: str, log) -> None:
    for suffix in ('.exe', '-script.py', '-script.pyw', '.py', '.pyw'):
        path = SCRIPTS_DIR / f'{name}{suffix}'
        if path.exists():
            try:
                path.unlink()
                log.write(f'Removed stale launcher {path.name}\n')
            except OSError as exc:
                log.write(f'WARNING: Could not remove {path.name}: {exc}\n')


def _regenerate_launchers(log) -> None:
    entry_points = _load_entry_points()
    maker = ScriptMaker(None, str(SCRIPTS_DIR))
    maker.executable = str(PYTHON_EXE)
    maker.variants = {''}
    maker.clobber = True
    maker.set_mode = False

    for spec in entry_points:
        name = spec.split('=')[0].strip()
        _remove_existing_wrappers(name, log)
        # Always target python.exe for launcher compatibility across install scopes.
        options = {'gui': False}
        maker.executable = str(PYTHON_EXE)
        created = maker.make(spec, options)
        log.write(f'Regenerated {name}: {created}\n')


log = _open_log()
try:
    if not ENTRY_POINTS_FILE.exists():
        raise FileNotFoundError(f'Missing entry_points.json at {ENTRY_POINTS_FILE}')
    _regenerate_launchers(log)
finally:
    if getattr(log, 'write', None) is print:
        print("Couldn't create install.log, so waiting...")
        time.sleep(10.0)
    else:
        log.close()
