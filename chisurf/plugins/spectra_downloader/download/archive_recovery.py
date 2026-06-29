#!/usr/bin/env python3
"""Re-ingest probes from an archived ``spectra.db`` through the canonical contract.

Some upstream sources retire their public catalogue (e.g. ATTO-TEC's per-dye
spectra pages disappeared after the Leica acquisition). The last good scrape
still lives in git history, so the archive becomes the data source. This module
reads probes from an archived ``spectra.db`` and re-registers them via
``FluorophoreDatabase.register_component`` so the recovered records get the same
canonical category / provenance / ``component_kind`` / spectra as a live scrape.

Recover the archive from git, then run::

    git cat-file -p <old_rev>:chisurf/plugins/_dev/fluorophore_db/spectra.db > /tmp/archive.db
    python -m chisurf.plugins.spectra_downloader.download.archive_recovery \\
        --archive /tmp/archive.db --db <target spectra.db> --name-like 'ATTO%' \\
        --source atto --kind organic_dye
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


def import_from_archive(
    db: FluorophoreDatabase,
    archive_path: str,
    name_like: str,
    source: str,
    kind: str,
    source_ref: str | None = None,
) -> int:
    """Recover probes whose name matches ``name_like`` from an archived DB.

    Parameters
    ----------
    db : FluorophoreDatabase
        Open target database (the staging ``spectra.db``).
    archive_path : str
        Path to the archived ``spectra.db`` extracted from git history.
    name_like : str
        SQL ``LIKE`` pattern matched case-insensitively against the probe name
        (e.g. ``"ATTO%"``).
    source : str
        Canonical provenance source slug to stamp on the recovered probes.
    kind : str
        Canonical component kind (key of ``COMPONENT_KINDS``).
    source_ref : str, optional
        Per-record reference; defaults to ``"<source> (archived)"``.

    Returns
    -------
    int
        Number of probes recovered.
    """
    source_ref = source_ref or f"{source} (archived)"
    arc = sqlite3.connect(archive_path)
    arc.row_factory = sqlite3.Row

    probe_cols = [r[1] for r in arc.execute("PRAGMA table_info(probes)").fetchall()]
    name_col = "chromophore_name" if "chromophore_name" in probe_cols else "name"
    spec_key = _key(arc, "spectra")
    opt_key = _key(arc, "optical_properties")

    rows = arc.execute(
        f"SELECT * FROM probes WHERE upper({name_col}) LIKE upper(?)",
        (name_like,),
    ).fetchall()

    recovered = 0
    with db:
        for p in rows:
            name = str(p[name_col] or "").strip()
            if not name:
                continue

            properties = {
                str(r["property_name"]): r["property_value"]
                for r in arc.execute(
                    f"SELECT property_name, property_value FROM optical_properties "
                    f"WHERE {opt_key} = ?",
                    (p["probe_id"],),
                )
            }

            spectra: dict[str, tuple] = {}
            for s in arc.execute(
                f"SELECT spectrum_type, wavelengths, intensity_values FROM spectra "
                f"WHERE {spec_key} = ?",
                (p["probe_id"],),
            ):
                wl = np.frombuffer(s["wavelengths"], dtype=np.float64)
                iv = np.frombuffer(s["intensity_values"], dtype=np.float64)
                if wl.size and wl.size == iv.size:
                    spectra[s["spectrum_type"]] = (wl, iv)

            description = ""
            if "description" in probe_cols:
                description = str(p["description"] or "")

            db.register_component(
                name=name,
                source=source,
                kind=kind,
                source_ref=source_ref,
                description=description,
                properties=properties,
                spectra=spectra,
            )
            recovered += 1
        db.conn.commit()

    arc.close()
    print(f"Recovered {recovered} '{name_like}' probes from {archive_path} "
          f"(source={source}, kind={kind}).")
    return recovered


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Recover probes from an archived spectra.db via register_component",
    )
    parser.add_argument("--archive", required=True, help="Archived spectra.db path")
    parser.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="Target spectra.db")
    parser.add_argument("--name-like", required=True, help="LIKE pattern, e.g. 'ATTO%%'")
    parser.add_argument("--source", required=True, help="Provenance source slug, e.g. atto")
    parser.add_argument("--kind", required=True, help="Component kind, e.g. organic_dye")
    parser.add_argument("--source-ref", default=None, help="Optional source reference")
    args = parser.parse_args()

    db = FluorophoreDatabase(args.db)
    with db:
        import_from_archive(
            db, args.archive, args.name_like, args.source, args.kind, args.source_ref,
        )


if __name__ == "__main__":
    main()
