#!/usr/bin/env python
"""
Build script wrapper that sets CHISURF_VERSION before building.
"""
import subprocess
import os
import sys
import re
from pathlib import Path

# Get the repo root - the script is in rattler-recipe/ so go up 2 levels
script_dir = Path(__file__).parent  # rattler-recipe
repo_root = script_dir.parent       # chisurf
os.chdir(repo_root)

print(f"Working directory: {os.getcwd()}")

# Generate version
result = subprocess.run(
    [sys.executable, str(script_dir / "generate_version.py"), "--print"],
    capture_output=True,
    text=True,
    cwd=repo_root
)
version = result.stdout.strip()
if result.returncode != 0:
    print(f"ERROR: Failed to generate version. stderr: {result.stderr}")
    sys.exit(1)
if not version:
    print("ERROR: Version is empty")
    sys.exit(1)

print(f"Building with CHISURF_VERSION={version}")

# Read the recipe.yaml file
recipe_file = script_dir / "recipe.yaml"
with open(recipe_file, 'r') as f:
    recipe_content = f.read()

# Store the original content
original_content = recipe_content

# Replace the version line in the recipe context
# Find the line: version: ${{ env.get("CHISURF_VERSION") if "CHISURF_VERSION" in env else "26.dev0" }}
# and replace with: version: "..." (hardcoded version)
recipe_content = re.sub(
    r'  version: \$\{\{ env\.get\("CHISURF_VERSION"\).*?\}\}',
    f'  version: "{version}"',
    recipe_content
)

try:
    # Write the modified recipe
    with open(recipe_file, 'w') as f:
        f.write(recipe_content)
    
    # Run rattler-build
    result = subprocess.run(
        ["rattler-build", "build", 
         "--recipe", "rattler-recipe/recipe.yaml", 
         "--output-dir", "conda-bld", 
         "--no-test"],
        cwd=repo_root
    )
finally:
    # Restore the original recipe
    with open(recipe_file, 'w') as f:
        f.write(original_content)

sys.exit(result.returncode)
