#!/usr/bin/env python
"""
Build Inno Setup installer directly without full environment creation.
Uses the already-built package.
"""
import subprocess
import os
import sys
from pathlib import Path

# Get paths
script_dir = Path(__file__).parent  # build_tools/win
repo_root = script_dir.parent.parent  # chisurf
os.chdir(script_dir)

print(f"Building installer from {repo_root}")
print(f"Script directory: {script_dir}")

# Get version
result = subprocess.run(
    [sys.executable, str(repo_root / "rattler-recipe" / "generate_version.py"), "--print"],
    capture_output=True,
    text=True,
    cwd=repo_root
)
version = result.stdout.strip()
if not version:
    print("ERROR: Could not get version")
    sys.exit(1)

print(f"Version: {version}")

# Generate the Inno Setup script
print("\nGenerating Inno Setup script...")
result = subprocess.run(
    [sys.executable, "create_installer_script.py"],
    cwd=script_dir
)
if result.returncode != 0:
    print("ERROR: Failed to generate Inno Setup script")
    sys.exit(1)

# Run Inno Setup
inno_path = Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe")
if not inno_path.exists():
    print(f"ERROR: Inno Setup 6 not found at {inno_path}")
    sys.exit(1)

print("Building setup.exe with Inno Setup...")
result = subprocess.run(
    [str(inno_path), "installer_config.iss"],
    cwd=script_dir
)
if result.returncode != 0:
    print("ERROR: Inno Setup failed")
    sys.exit(1)

print(f"\n✓ Setup complete: setup.exe built for version {version}")
sys.exit(0)
