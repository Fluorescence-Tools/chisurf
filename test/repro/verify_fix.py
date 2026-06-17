import sys
from pathlib import Path
import sqlite3

# Add project root
sys.path.append(r"e:\dev\chisurf")

from chisurf.core.mfdb.repository import MFDatabase

db_path = r"e:\dev\chisurf\chisurf\plugins\_dev\fluorophore_db\spectra.db"

print(f"Initializing MFDatabase with {db_path}...")
db = MFDatabase(db_path)

print(f"New Database Version: {db._get_schema_version()}")

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

print("--- PROBE_TYPES SCHEMA AFTER MIGRATION ---")
cols = [r[1] for r in conn.execute("PRAGMA table_info(probe_types)").fetchall()]
print(f"Columns: {cols}")

# Test the failing query
try:
    types = db.get_probe_types_dict()
    print(f"PASSED: get_probe_types_dict returned {len(types)} items.")
except Exception as e:
    print(f"FAILED: get_probe_types_dict raised: {e}")

db.close()
conn.close()
