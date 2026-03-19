import sys
import os

# Root of the repo
repo_root = r"E:\dev\chisurf"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("PYTHONPATH:", sys.path[:3])

try:
    print("Testing import of rmf submodule...")
    from chisurf.plugins.chimol.chimol.io import rmf
    print("SUCCESS: imported rmf module")
    print("Names in rmf module:", [n for n in dir(rmf) if not n.startswith('__')])
    
    print("\nTesting import of load_rmf_full from io...")
    from chisurf.plugins.chimol.chimol.io import load_rmf_full
    print("SUCCESS: imported load_rmf_full")
except Exception as e:
    import traceback
    traceback.print_exc()
