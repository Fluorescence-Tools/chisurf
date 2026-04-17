#!/usr/bin/env python
"""
Build setup.exe installer wrapper.
"""
import subprocess
import os
import sys
from pathlib import Path

# Get the repo root
script_dir = Path(__file__).parent  # build_tools/win
repo_root = script_dir.parent.parent  # chisurf
os.chdir(repo_root)

print(f"Working directory: {os.getcwd()}")

# Run the batch script
result = subprocess.run(
    [str(script_dir / "build-setup.bat")] + sys.argv[1:],
    cwd=str(script_dir),
    shell=True
)
sys.exit(result.returncode)
