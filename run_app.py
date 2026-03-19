import sys
import os
import traceback

# Force local path
local_path = r'e:\dev\chisurf'
if local_path not in sys.path:
    sys.path.insert(0, local_path)

print(f"Starting ChiSurf from {local_path}...")

try:
    import chisurf.__main__
    chisurf.__main__.main()
except SystemExit as e:
    print(f"SystemExit: {e.code}")
    if e.code != 0:
        traceback.print_exc()
except Exception as e:
    print("CRASH DETECTED")
    traceback.print_exc()
    sys.exit(1)
print("Application exited normally.")
