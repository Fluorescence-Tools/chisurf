#!/usr/bin/env bash
# Build documentation using pixi.
#
# Usage:
#   ./build_tools/docs/build_docs.sh          # build HTML docs
#   ./build_tools/docs/build_docs.sh --help    # this message
#
# Requires pixi to be installed and on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "${1:-}" == "--help" ]]; then
  sed -n '3,8p' "$0"
  exit 0
fi

if ! command -v pixi &>/dev/null; then
  echo "Error: pixi is required but not found on PATH."
  echo "Install it from https://pixi.sh/latest/"
  exit 1
fi

cd "$REPO_ROOT"

echo "==> Installing docs dependencies (pixi environment 'docs')..."
pixi install -e docs

echo "==> Cleaning previous build..."
pixi run -e docs docs-clean

echo "==> Building HTML documentation..."
pixi run -e docs docs-html

echo ""
echo "Done. Open file://$REPO_ROOT/docs/_build/index.html in your browser."
