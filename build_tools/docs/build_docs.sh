#!/usr/bin/env bash
# Build documentation using pixi

# Get repo root
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
cd "$REPO_ROOT"

echo "Building ChiSurf documentation via pixi..."
pixi run -e docs docs-clean
pixi run -e docs docs-html

echo "Done."
