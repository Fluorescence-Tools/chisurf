#!/usr/bin/env bash
# Download/update mmCIF dictionary files from wwPDB.
# Run from any directory — files are placed next to this script.
#
# Usage:
#   bash chisurf/core/mfdb/data/update_dictionaries.sh
#   # or from this directory:
#   ./update_dictionaries.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_URL="https://mmcif.wwpdb.org/dictionaries/ascii"

DICTS=(
    # Core PDBx/mmCIF
    "mmcif_pdbx_v50.dic"          # PDBx/mmCIF v5 (current stable)
    "mmcif_pdbx_v5_next.dic"      # PDBx/mmCIF v5-next (development)
    "mmcif_std.dic"               # Original mmCIF standard
    "mmcif_ddl.dic"               # DDL v2 (dictionary definition language)

    # Extensions
    "mmcif_ihm_ext.dic"           # IHM (integrative/hybrid modeling)
    "mmcif_ihm_flr_ext.dic"       # flrCIF (fluorescence/FRET)
    "mmcif_ma.dic"                # ModelCIF (computed structure models)
)

echo "Downloading mmCIF dictionaries to: $SCRIPT_DIR"
echo "Source: $BASE_URL"
echo ""

for dic in "${DICTS[@]}"; do
    printf "  %-35s ... " "$dic"
    if curl -sf -o "$dic" "$BASE_URL/$dic"; then
        size=$(wc -c < "$dic" | tr -d ' ')
        printf "OK (%s bytes)\n" "$size"
    else
        printf "FAILED\n"
    fi
done

echo ""
echo "Done. $(ls -1 *.dic 2>/dev/null | wc -l | tr -d ' ') dictionary files present."
