import os
import sys
import json
import pathlib

# Add project root to sys.path to import chisurf.info
root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

try:
    from chisurf.info import __version__
    version = __version__
except ImportError:
    # Fallback if chisurf is not importable
    import datetime
    import subprocess
    
    today = datetime.datetime.now()
    year = today.strftime("%y")
    try:
        count = subprocess.check_output(['git', 'rev-list', '--count', 'HEAD'], text=True).strip()
    except:
        count = "0"
    version = f"{year}.dev{count}"

version_file = pathlib.Path(__file__).resolve().parent / "version.json"
if "--print" in sys.argv:
    print(version)
else:
    with open(version_file, "w") as f:
        json.dump({"version": version}, f, indent=2)
    print(f"Generated {version_file} with version: {version}")
