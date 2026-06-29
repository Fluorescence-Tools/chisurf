import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click

# Network scrapers that can run in parallel, each into its own per-source DB.
# Values are extra CLI args appended after ``--db <tmp>``.
PARALLEL_SOURCES: dict[str, list[str]] = {
    "fpbase": [],
    "chroma": [],
    "thorlabs": [],
    "threed_optix": ["--max-pages", "0"],
    "atto": [],
    "omega_optical": [],
}

def get_available_sources():
    download_dir = Path(__file__).parent / "download"
    sources = {}
    if download_dir.exists():
        for script_file in download_dir.glob("*.py"):
            if script_file.name == "__init__.py" or script_file.name.startswith("import_"):
                continue
            if script_file.name.startswith("probe_"):
                continue
            sources[script_file.stem] = script_file
    return sources

@click.group()
def cli():
    """Spectra Downloader CLI tool."""
    pass

@click.command("list-sources")
def list_sources():
    """List all available downloader sources."""
    sources = get_available_sources()
    click.echo("Available downloader sources:")
    for src in sorted(sources.keys()):
        click.echo(f"  - {src}")

@cli.command("run")
@click.argument("source")
@click.option("--db", default=None, help="Path to the SQLite database.")
@click.option("--extra", "extra_args", default=None, help="Extra arguments to pass to the script (space separated).")
def run_source(source, db, extra_args):
    """Run a specific downloader source."""
    sources = get_available_sources()
    if source not in sources:
        click.echo(f"Error: Unknown source '{source}'. Available: {', '.join(sorted(sources.keys()))}")
        sys.exit(1)
        
    script_path = sources[source]
    args = [sys.executable, str(script_path)]
    if db:
        args += ["--db", db]
        
    if extra_args:
        args += extra_args.split()
        
    click.echo(f"Running {source} downloader ({script_path.name}) ...")
    res = subprocess.run(args)
    sys.exit(res.returncode)

@cli.command("run-all")
@click.option("--db", default=None, help="Target spectra.db to merge everything into.")
@click.option("--only", default=None, help="Comma-separated subset of sources (default: all).")
@click.option("--keep-temp", is_flag=True, help="Keep the per-source temp DBs/logs.")
@click.option("--no-consolidate", is_flag=True, help="Skip de-duplication after merge.")
def run_all(db, only, keep_temp, no_consolidate):
    """Run scrapers in parallel, each into its own DB, then merge + consolidate.

    Stage 1+2 (download & save) run concurrently per source with no SQLite
    contention; the final stage merges every per-source DB into one canonical
    ``spectra.db`` and de-duplicates across sources.
    """
    from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import DEFAULT_DATABASE_PATH
    from chisurf.plugins.spectra_downloader.download.merge import merge_all

    target = db or str(DEFAULT_DATABASE_PATH)
    if only:
        names = [s.strip() for s in only.split(",") if s.strip()]
        unknown = [n for n in names if n not in PARALLEL_SOURCES]
        if unknown:
            click.echo(f"Unknown source(s): {', '.join(unknown)}. "
                       f"Available: {', '.join(PARALLEL_SOURCES)}")
            sys.exit(1)
    else:
        names = list(PARALLEL_SOURCES)

    tmpdir = tempfile.mkdtemp(prefix="spectra_parallel_")
    click.echo(f"Launching {len(names)} scraper(s) in parallel (temp: {tmpdir}) …")

    procs = {}
    for name in names:
        tmp = os.path.join(tmpdir, f"{name}.db")
        args = [sys.executable, "-m",
                f"chisurf.plugins.spectra_downloader.download.{name}", "--db", tmp]
        args += PARALLEL_SOURCES[name]
        logf = open(os.path.join(tmpdir, f"{name}.log"), "w")
        procs[name] = (subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT), tmp, logf)
        click.echo(f"  ▶ {name}")

    paths = []
    for name, (proc, tmp, logf) in procs.items():
        rc = proc.wait()
        logf.close()
        ok = os.path.exists(tmp)
        if ok:
            paths.append(tmp)
        click.echo(f"  {'✓' if rc == 0 and ok else '✗'} {name} (rc={rc})")

    if not paths:
        click.echo("No per-source databases were produced; nothing to merge.")
        sys.exit(1)

    click.echo(f"Merging {len(paths)} source DB(s) into {target} …")
    merge_all(target, paths, consolidate=not no_consolidate)

    if not keep_temp:
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        click.echo(f"Per-source DBs/logs kept in {tmpdir}")
    click.echo("run-all complete.")


@cli.command("merge")
@click.option("--db", default=None, help="Target spectra.db.")
@click.argument("sources", nargs=-1, required=True)
@click.option("--no-consolidate", is_flag=True, help="Skip de-duplication after merge.")
def merge_cmd(db, sources, no_consolidate):
    """Merge already-scraped per-source DBs into the target spectra.db."""
    from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import DEFAULT_DATABASE_PATH
    from chisurf.plugins.spectra_downloader.download.merge import merge_all

    target = db or str(DEFAULT_DATABASE_PATH)
    click.echo(f"Merging {len(sources)} source DB(s) into {target} …")
    merge_all(target, list(sources), consolidate=not no_consolidate)
    click.echo("Merge complete.")


@cli.command("consolidate")
@click.option("--db", default=None, help="Path to the SQLite database.")
def consolidate(db):
    """Consolidate duplicate probes and merge their spectra/properties."""
    from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import DEFAULT_DATABASE_PATH, FluorophoreDatabase
    db_path = db or str(DEFAULT_DATABASE_PATH)
    click.echo(f"Consolidating database at {db_path}...")
    with FluorophoreDatabase(db_path) as fdb:
        res = fdb.consolidate_probes()
    click.echo(f"Consolidation complete: merged {res['merged_groups']} groups, deleted {res['deleted_probes']} duplicate probes.")

if __name__ == "__main__":
    cli()
