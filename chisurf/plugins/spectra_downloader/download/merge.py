#!/usr/bin/env python3
"""Merge per-source staging databases into one canonical ``spectra.db``.

Scrapers run **in parallel**, each into its own ``spectra_<source>.db`` (no
SQLite write contention). This module is the final merge step: it re-ingests
every probe from a per-source DB into the target through
``FluorophoreDatabase.register_component``, so the merged database stays
canonical (consistent category / provenance / ``component_kind`` / spectra).
Run ``consolidate_probes`` afterwards to de-duplicate across sources.
"""

import argparse
import sqlite3

import numpy as np

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import (
    DEFAULT_DATABASE_PATH,
    FluorophoreDatabase,
)


def _key(conn: sqlite3.Connection, table: str) -> str:
    """Return whether ``table`` references probes via ``probe_id`` or ``item_id``."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return "probe_id" if "probe_id" in cols else "item_id"


def merge_database(
    db: FluorophoreDatabase,
    source_path: str,
    probe_ids: "set[int] | list[int] | None" = None,
) -> int:
    """Re-ingest probes from ``source_path`` into ``db``.

    The component kind is read back from each probe's ``component_kind`` optical
    property (set by ``register_component`` on the original scrape), so the
    merged record keeps its canonical category and spectrum grouping.

    Parameters
    ----------
    probe_ids : optional
        Restrict the merge to these source probe ids (e.g. a selection in the
        browser). ``None`` merges every probe.

    Returns the number of probes merged.
    """
    src = sqlite3.connect(source_path)
    src.row_factory = sqlite3.Row

    probe_cols = [r[1] for r in src.execute("PRAGMA table_info(probes)").fetchall()]
    name_col = "chromophore_name" if "chromophore_name" in probe_cols else "name"
    has_source = "source" in probe_cols
    has_source_ref = "source_ref" in probe_cols
    has_desc = "description" in probe_cols
    spec_key = _key(src, "spectra")
    opt_key = _key(src, "optical_properties")

    if probe_ids is not None:
        ids = list(probe_ids)
        if not ids:
            src.close()
            return 0
        placeholders = ",".join("?" * len(ids))
        rows = src.execute(
            f"SELECT * FROM probes WHERE deleted_at IS NULL AND probe_id IN ({placeholders})",
            ids,
        ).fetchall()
    else:
        rows = src.execute("SELECT * FROM probes WHERE deleted_at IS NULL").fetchall()

    merged = 0
    with db:
        for p in rows:
            name = str(p[name_col] or "").strip()
            if not name:
                continue

            properties = {
                str(r["property_name"]): r["property_value"]
                for r in src.execute(
                    f"SELECT property_name, property_value FROM optical_properties "
                    f"WHERE {opt_key} = ?",
                    (p["probe_id"],),
                )
            }
            kind = str(properties.get("component_kind") or "other")

            spectra: dict[str, tuple] = {}
            for s in src.execute(
                f"SELECT spectrum_type, wavelengths, intensity_values FROM spectra "
                f"WHERE {spec_key} = ?",
                (p["probe_id"],),
            ):
                wl = np.frombuffer(s["wavelengths"], dtype=np.float64)
                iv = np.frombuffer(s["intensity_values"], dtype=np.float64)
                if wl.size and wl.size == iv.size:
                    spectra[s["spectrum_type"]] = (wl, iv)

            source = (p["source"] if has_source else None) or "unknown"
            source_ref = p["source_ref"] if has_source_ref else None
            description = str(p["description"]) if has_desc and p["description"] else ""

            db.register_component(
                name=name,
                source=source,
                kind=kind,
                source_ref=source_ref,
                description=description,
                properties=properties,
                spectra=spectra,
            )
            merged += 1
        db.conn.commit()

    src.close()
    return merged


def merge_all(target_path: str, source_paths: list[str], consolidate: bool = True) -> dict:
    """Merge several per-source DBs into ``target_path`` and (optionally) dedup.

    Returns a summary dict ``{path: merged_count, ..., "consolidated": {...}}``.
    """
    summary: dict = {}
    db = FluorophoreDatabase(target_path)
    with db:
        for path in source_paths:
            n = merge_database(db, path)
            summary[path] = n
            print(f"  merged {n:5d} probes from {path}")
        if consolidate:
            summary["consolidated"] = db.consolidate_probes()
            print(f"  consolidated: {summary['consolidated']}")
    return summary


def push_staging_to_mfdb(
    staging_path: str,
    probe_ids: "set[int] | list[int] | None" = None,
    mfdb_path: str | None = None,
    consolidate: bool = True,
) -> dict:
    """Push staging probes (a selection, or all) into the connected MFDB.

    Re-ingests via ``register_component`` so the MFDB records stay canonical,
    then de-duplicates. Used by the browser's "Push selected / Push all" actions.
    Returns ``{"merged": n, "consolidated": {...}}``.
    """
    from chisurf.core.mfdb.store.database_resolver import resolve_database_path

    target = mfdb_path or str(resolve_database_path())
    # Open as FluorophoreDatabase so register_component is available (it is an
    # MFDatabase subclass, so it works on the live MFDB too).
    db = FluorophoreDatabase(target)
    summary: dict = {}
    with db:
        summary["merged"] = merge_database(db, staging_path, probe_ids=probe_ids)
        if consolidate:
            summary["consolidated"] = db.consolidate_probes()
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Merge per-source spectra DBs into one canonical spectra.db",
    )
    parser.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="Target spectra.db")
    parser.add_argument("sources", nargs="+", help="Per-source spectra DB paths to merge in")
    parser.add_argument("--no-consolidate", action="store_true", help="Skip de-duplication")
    args = parser.parse_args()

    print(f"Merging {len(args.sources)} source DB(s) into {args.db} …")
    merge_all(args.db, args.sources, consolidate=not args.no_consolidate)
    print("Merge complete.")


if __name__ == "__main__":
    main()
