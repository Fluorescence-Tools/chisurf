import sqlite3
import os

db_path = r"e:\dev\chisurf\chisurf\plugins\_dev\fluorophore_db\spectra.db"

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

print("--- PROBE_TYPES SCHEMA ---")
cols = [dict(r) for r in conn.execute("PRAGMA table_info(probe_types)").fetchall()]
for c in cols:
    print(c)

print("--- PROBES SCHEMA ---")
cols = [dict(r) for r in conn.execute("PRAGMA table_info(probes)").fetchall()]
for c in cols:
    print(c)

conn.close()
