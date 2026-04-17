import json
import os
import sys
from pathlib import Path

# Use the bundled distlib within pip
try:
    from pip._vendor.distlib.scripts import ScriptMaker
except ImportError:
    print("ERROR: Could not find distlib. Ensure 'pip' is installed in the build environment.")
    sys.exit(1)

def fix_launchers(target_prefix=None):
    # Use provided prefix, or fallback to PREFIX env var (standard in conda builds)
    prefix_str = target_prefix or os.environ.get('PREFIX')
    if not prefix_str:
        print("ERROR: No prefix provided and PREFIX environment variable not set.")
        sys.exit(1)
    
    prefix = Path(prefix_str)
    scripts_dir = prefix / 'Scripts'
    python_exe = prefix / 'python.exe'
    
    # The entry_points.json is in the same directory as this script (rattler-recipe/)
    script_dir = Path(__file__).parent
    entry_points_file = script_dir / 'entry_points.json'

    if not entry_points_file.exists():
        print(f"ERROR: {entry_points_file} not found.")
        sys.exit(1)

    with open(entry_points_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Fixing launchers in {scripts_dir} using executable {python_exe}")
    
    maker = ScriptMaker(None, str(scripts_dir))
    maker.executable = str(python_exe)
    maker.variants = {''}
    maker.clobber = True
    maker.set_mode = False

    # Fix console scripts
    console_scripts = data.get('console_scripts', {})
    for name, func in console_scripts.items():
        spec = f"{name} = {func}"
        created = maker.make(spec, {'gui': False})
        print(f"  [Console] {name} -> {created}")

    # Fix GUI scripts
    gui_scripts = data.get('gui_scripts', {})
    for name, func in gui_scripts.items():
        spec = f"{name} = {func}"
        created = maker.make(spec, {'gui': True})
        print(f"  [GUI] {name} -> {created}")

if __name__ == "__main__":
    # If a path is passed as an argument, use it as the prefix
    target = sys.argv[1] if len(sys.argv) > 1 else None
    fix_launchers(target)

