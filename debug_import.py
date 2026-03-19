import sys
import os
# Add the plugins directory to sys.path
plugins_dir = r"E:\dev\chisurf\chisurf\plugins"
if plugins_dir not in sys.path:
    sys.path.insert(0, plugins_dir)

try:
    from chimol.chimol.io import load_rmf_full
    print("SUCCESS: imported load_rmf_full")
except Exception as e:
    import traceback
    traceback.print_exc()
