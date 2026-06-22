#!/usr/bin/env python3
"""Regenerate the shipped curated MFDB on the current schema.

The curated source database (``chisurf/core/fio/mmCIF/db/sample_management.db``)
is copied to each user's settings dir on first run. After PRD-19 removed the
versioned migration waterfall (option B), a pre-PRD-19 curated DB is no longer
fully migrated forward: missing *columns* are added by ``_ensure_canonical_columns``
but stale *table structure* (e.g. a missing ``flr_sample_users.user_uuid`` UNIQUE
that an FK needs) cannot be fixed by ``ALTER ADD COLUMN``, so first-run object
registration fails with a foreign-key mismatch.

This rebuilds the curated DB on a **fresh current schema** and copies the
reference/demo data (probes, spectra, optical properties, entities, samples,
experiments, …) into the correctly-structured tables, fixing the mismatch.

Usage
-----
    python build_tools/regenerate_curated_db.py            # write to a temp file + verify
    python build_tools/regenerate_curated_db.py --replace  # back up + overwrite the shipped DB
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHIPPED_DB = REPO / "chisurf" / "core" / "fio" / "mmcif" / "db" / "sample_management.db"


def _common_columns(old: sqlite3.Connection, new: sqlite3.Connection, table: str) -> list[str]:
    old_cols = [r[1] for r in old.execute(f"PRAGMA table_info({table})")]
    new_cols = {r[1] for r in new.execute(f"PRAGMA table_info({table})")}
    return [c for c in old_cols if c in new_cols]


def regenerate(source_db: Path, out_db: Path) -> dict[str, int]:
    """Build a fresh current-schema DB at *out_db* and copy *source_db*'s data."""
    if out_db.exists():
        out_db.unlink()
    out_db.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fresh current-schema database (triggers migrate_schema + bootstrap).
    from chisurf.core.mfdb.repository import MFDatabase

    MFDatabase(str(out_db)).close()

    old = sqlite3.connect(source_db)
    new = sqlite3.connect(out_db)
    copied: dict[str, int] = {}
    try:
        old_tables = {
            r[0] for r in old.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        new_tables = {
            r[0] for r in new.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        # Copy with FK enforcement off; the fresh tables already have the correct
        # structure, so copying valid data into them is safe. INSERT OR IGNORE
        # skips rows that violate a (now-correct) UNIQUE/PK constraint.
        new.execute("PRAGMA foreign_keys=OFF")
        for table in sorted(old_tables & new_tables):
            if table.startswith("sqlite_") or table == "_schema_version":
                continue
            cols = _common_columns(old, new, table)
            if not cols:
                continue
            rows = old.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
            if not rows:
                continue
            placeholders = ", ".join("?" for _ in cols)
            new.executemany(
                f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
                rows,
            )
            copied[table] = new.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        new.commit()
    finally:
        old.close()
        new.close()
    return copied


def verify(db_path: Path) -> None:
    """Open via the normal path and exercise object registration (FK-sensitive)."""
    fk_violations = []
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        fk_violations = con.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        con.close()
    if fk_violations:
        raise SystemExit(f"FK violations after regen: {fk_violations[:10]}")

    # Real round trip: register a raw measurement against a COPY (so we don't
    # mutate the regenerated artifact), which exercises put_object's FK path.
    from chisurf.core.mfdb.repository import MFDatabase
    from chisurf.core.mfdb.result_registry import register_raw_measurement, set_global_db

    tmp = Path(tempfile.mkdtemp()) / "verify.db"
    shutil.copy(db_path, tmp)
    f = Path(tempfile.mkdtemp()) / "m.ptu"
    f.write_bytes(b"\x00\x01\x02")
    db = MFDatabase(str(tmp))
    try:
        artifact_id = register_raw_measurement(str(f), db=db)
        if not artifact_id:
            raise SystemExit("Verification FAILED: register_raw_measurement returned ''")
    finally:
        set_global_db(None)
        db.close()
    print(f"  verify: register_raw_measurement OK (artifact {artifact_id}); 0 FK violations")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replace", action="store_true", help="overwrite the shipped DB (with backup)")
    ap.add_argument("--source", type=Path, default=SHIPPED_DB)
    args = ap.parse_args()

    out = Path(tempfile.mkdtemp()) / "sample_management.db"
    print(f"Regenerating from {args.source}")
    copied = regenerate(args.source, out)
    print(f"  copied {sum(copied.values())} rows across {len(copied)} tables: {copied}")
    verify(out)

    if args.replace:
        backup = args.source.with_suffix(".db.pre-prd19.bak")
        shutil.copy(args.source, backup)
        shutil.copy(out, args.source)
        print(f"  backed up old -> {backup}")
        print(f"  replaced shipped DB: {args.source}")
    else:
        print(f"  regenerated DB at {out} (use --replace to install)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
