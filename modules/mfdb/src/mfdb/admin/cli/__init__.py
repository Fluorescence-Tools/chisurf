"""Fluorophore curation CLI (``csc fluorophore``).

Migrated from the standalone fluorophore_db plugin into mfdb-admin (PRD-06). It
operates directly on the live MFDB via the core repository — no legacy adapter —
so the headless curation path is preserved::

    csc fluorophore import-reference-set [--mark-verified]
    csc fluorophore list [--status STATUS] [--category CATEGORY] [--json]
    csc fluorophore approve <name> [--by WHO]
    csc fluorophore reject <name> [--by WHO]
    csc fluorophore set-quality <name> <grade>
    csc fluorophore review-queue [--json]
    csc fluorophore ai-triage [--status STATUS] [--provider KEY] [--json]
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Iterator

import click

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _open_db() -> Iterator["object"]:
    """Open the configured MFDB for the duration of a command."""
    from mfdb.repository import MFDatabase
    from mfdb.database_resolver import resolve_database_path

    db = MFDatabase(resolve_database_path())
    db.connect()
    try:
        yield db
    finally:
        db.close()


def _resolve_probe_id(db, name: str) -> int | None:
    row = db.conn.execute(
        "SELECT probe_id, chromophore_name FROM probes "
        "WHERE chromophore_name = ? AND deleted_at IS NULL",
        (name,),
    ).fetchone()
    if not row:
        row = db.conn.execute(
            "SELECT probe_id, chromophore_name FROM probes "
            "WHERE chromophore_name LIKE ? AND deleted_at IS NULL",
            (f"%{name}%",),
        ).fetchone()
        if row:
            click.echo(f"  (matched by partial name: '{row['chromophore_name']}')")
    if row:
        return int(row["probe_id"])
    click.echo(f"Error: no probe found matching '{name}'")
    return None


@click.group()
def cli() -> None:
    """Manage and curate fluorophores in the MFDB."""


@cli.command("import-reference-set")
@click.option("--source", default=None, help="Path to the reference spectra.db.")
@click.option("--mark-verified", is_flag=True, default=False,
              help="Stamp imported probes as approved (default: unverified).")
@click.option("--replace", is_flag=True, default=False,
              help="Purge existing reference probes first, then import cleanly.")
def import_reference_set(source: str | None, mark_verified: bool, replace: bool) -> None:
    """Import the scraped reference dataset into the MFDB."""
    click.echo("Importing fluorophore reference set..."
               + (" (replacing existing)" if replace else ""))
    with _open_db() as db:
        counts = db.import_reference_set(
            source_path=source, mark_verified=mark_verified, replace=replace,
        )
    if counts.get("purged"):
        click.echo(f"  Purged existing: {counts['purged']}")
    click.echo(
        f"  Probes imported: {counts['probes']}\n"
        f"  Spectra imported: {counts['spectra']}\n"
        f"  Optical properties: {counts['optical_properties']}\n"
        f"  Skipped: {counts['skipped']}\n"
        f"  Consolidated: {counts.get('consolidated')}"
    )


@cli.command("list")
@click.option("--status", default=None, help="Filter by verification_status.")
@click.option("--category", default=None, help="Filter by category.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def list_probes(status: str | None, category: str | None, as_json: bool) -> None:
    """List probes in the database."""
    query = ("SELECT probe_id, chromophore_name, category, verification_status, "
             "quality, is_curated FROM probes WHERE deleted_at IS NULL")
    params: list = []
    if status:
        query += " AND verification_status = ?"
        params.append(status)
    if category:
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY chromophore_name"
    with _open_db() as db:
        rows = db.conn.execute(query, params).fetchall()
    if as_json:
        click.echo(json.dumps([dict(r) for r in rows], indent=2))
        return
    if not rows:
        click.echo("No probes found.")
        return
    click.echo(f"{'ID':>4}  {'Name':<30} {'Category':<15} {'Status':<15} {'Quality':<8} {'Curated':>3}")
    click.echo("-" * 80)
    for r in rows:
        click.echo(
            f"{r['probe_id']:>4}  {r['chromophore_name']:<30} "
            f"{str(r['category'] or ''):<15} {str(r['verification_status'] or ''):<15} "
            f"{str(r['quality'] or ''):<8} {r['is_curated']:>3}"
        )


@cli.command()
@click.argument("name")
@click.option("--by", "verified_by", default="cli", help="Who verified this probe.")
def approve(name: str, verified_by: str) -> None:
    """Approve a probe."""
    with _open_db() as db:
        probe_id = _resolve_probe_id(db, name)
        if probe_id is None:
            raise click.Abort()
        db.approve_probe(probe_id, verified_by=verified_by)
    click.echo(f"Approved probe '{name}' (id={probe_id})")


@cli.command()
@click.argument("name")
@click.option("--by", "verified_by", default="cli", help="Who rejected this probe.")
def reject(name: str, verified_by: str) -> None:
    """Reject a probe."""
    with _open_db() as db:
        probe_id = _resolve_probe_id(db, name)
        if probe_id is None:
            raise click.Abort()
        db.reject_probe(probe_id, verified_by=verified_by)
    click.echo(f"Rejected probe '{name}' (id={probe_id})")


@cli.command("set-quality")
@click.argument("name")
@click.argument("grade", type=click.Choice(["unknown", "low", "medium", "high"]))
def set_quality(name: str, grade: str) -> None:
    """Set the quality grade for a probe."""
    with _open_db() as db:
        probe_id = _resolve_probe_id(db, name)
        if probe_id is None:
            raise click.Abort()
        db.set_probe_quality(probe_id, grade)
    click.echo(f"Set quality of '{name}' to {grade}")


@cli.command("review-queue")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def review_queue(as_json: bool) -> None:
    """Show unverified probes that need review."""
    with _open_db() as db:
        rows = db.conn.execute(
            "SELECT probe_id, chromophore_name, category, quality, source, created_at "
            "FROM probes WHERE verification_status = 'unverified' AND deleted_at IS NULL "
            "ORDER BY created_at"
        ).fetchall()
    if as_json:
        click.echo(json.dumps([dict(r) for r in rows], indent=2))
        return
    if not rows:
        click.echo("No unverified probes — review queue is empty.")
        return
    click.echo(f"{'ID':>4}  {'Name':<30} {'Category':<15} {'Quality':<8} {'Source':<12} {'Created'}")
    click.echo("-" * 85)
    for r in rows:
        click.echo(
            f"{r['probe_id']:>4}  {r['chromophore_name']:<30} "
            f"{str(r['category'] or ''):<15} {str(r['quality'] or ''):<8} "
            f"{str(r['source'] or ''):<12} {str(r['created_at'] or '')[:19]}"
        )


@cli.command("ai-triage")
@click.option("--status", default="unverified", help="Filter by verification status.")
@click.option("--provider", default=None, help="LLM provider key override.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def ai_triage_cli(status: str, provider: str | None, as_json: bool) -> None:
    """Run AI-assisted triage on probes (deterministic checks + optional LLM)."""
    from chisurf.core.fluorescence.curation.ai_triage import run_triage

    if not as_json:
        click.echo(f"Running AI triage on probes with status='{status}'...")
    with _open_db() as db:
        results = run_triage(db, status=status, provider=provider)
    if as_json:
        click.echo(json.dumps(results, indent=2, default=str))
        return
    click.echo(f"\nProcessed {len(results)} probes:\n")
    for r in results:
        flags = "⚠️" if r["issues"] else "✓"
        click.echo(
            f"  {r['probe_id']:>4} {r['name']:<30} "
            f"quality={r['proposed_quality']:<6} rec={r['recommendation']:<12} {flags}"
        )
        if r["rationale"]:
            click.echo(f"       rationale: {r['rationale'][:120]}")
    click.echo(
        f"\nSummary: {sum(1 for r in results if not r['issues'])} clean, "
        f"{sum(1 for r in results if r['issues'])} with issues"
    )


if __name__ == "__main__":
    cli()
