#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChiSurf plugin CLI wrapper for ndXplorer filter and image subcommands (PRD-31)."""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
import click

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def _get_ndxplorer_env():
    """Ensure modules/ndxplorer is on PYTHONPATH."""
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[4]
    ndx_path = root / "modules" / "ndxplorer"
    python_path = env.get("PYTHONPATH", "")
    if str(ndx_path) not in python_path:
        env["PYTHONPATH"] = os.path.pathsep.join(filter(None, [str(ndx_path), python_path]))
    return env


@click.group()
def cli():
    """ndXplorer Headless CLI with MFDB integration."""
    pass


@cli.command("filter")
@click.option('--from-mfdb', required=True, type=str, help="Source MFDB burst selection artifact ID.")
@click.option('--select', '-s', multiple=True, type=str, help="Selection format: param:min-max")
@click.option('--query', '-q', type=str, help="Pandas eval query string.")
@click.option('--out', '-o', type=click.Path(), help="Output folder. If --to-mfdb is set, defaults to a sibling directory.")
@click.option('--to-mfdb/--no-to-mfdb', default=False, show_default=True, help="Register the resulting folder in MFDB.")
@click.option('--sample-id', type=str, help="MFDB sample ID to link the filtered selection to.")
@click.option('--db', 'db_path', type=click.Path(dir_okay=False), help="SQLite database path. Defaults to configured DB.")
@click.option('--skip-nth-row', type=int, default=1, show_default=True, help="Skip every Nth row (1 to load all).")
def filter_cmd(from_mfdb, select, query, out, to_mfdb, sample_id, db_path, skip_nth_row):
    """Run parameter-based burst filtering on a burst selection from MFDB."""
    from chisurf.core.mfdb.database_resolver import resolve_database_path
    from chisurf.core.mfdb.repository import MFDatabase
    from chisurf.core.mfdb.result_registry import register_result
    
    resolved_db_path = db_path or resolve_database_path()
    logging.info(f"Opening database: {resolved_db_path}")
    
    with MFDatabase(resolved_db_path) as db:
        local_path = db.open_dataset(from_mfdb)
        if not local_path:
            click.echo(json.dumps({"error": f"Failed to resolve artifact: {from_mfdb}"}), err=True)
            sys.exit(1)
            
    logging.info(f"Resolved source burst selection to: {local_path}")
    
    # Resolve output directory
    if not out:
        # Default output folder to a sibling of the original with a suffix
        src_path = Path(local_path)
        out = str(src_path.parent / f"{src_path.name}_filtered")
        
    # Execute the pure CLI
    env = _get_ndxplorer_env()
    args = [sys.executable, "-m", "ndxplorer", "filter", "--folder", local_path, "--out", out, "--skip-nth-row", str(skip_nth_row)]
    for s in select:
        args += ["--select", s]
    if query:
        args += ["--query", query]
        
    logging.info(f"Running ndxplorer filter command: {' '.join(args)}")
    res = subprocess.run(args, capture_output=True, text=True, env=env)
    
    if res.returncode != 0:
        click.echo(json.dumps({"error": f"Filter subcommand failed: {res.stderr}"}), err=True)
        sys.exit(1)
        
    stdout_clean = res.stdout.strip()
    try:
        result_json = json.loads(stdout_clean)
    except Exception:
        start_idx = stdout_clean.find('{')
        end_idx = stdout_clean.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                result_json = json.loads(stdout_clean[start_idx:end_idx+1])
            except Exception as e:
                click.echo(json.dumps({"error": f"Failed to parse filter output JSON block: {stdout_clean}, error: {e}"}), err=True)
                sys.exit(1)
        else:
            click.echo(json.dumps({"error": f"Failed to parse filter output: {res.stdout}"}), err=True)
            sys.exit(1)
        
    new_artifact_id = None
    if to_mfdb:
        # Register in MFDB
        with MFDatabase(resolved_db_path) as db:
            try:
                db.validate_extensible_vocab("operation_type", "burst_filter")
            except ValueError:
                db.register_vocabulary_value(
                    "operation_type",
                    "burst_filter",
                    display_name="Burst Filter",
                    description="Filter burst selection by parameters",
                    is_active=True,
                )
            new_artifact_id = register_result(
                kind="external_reference",
                data=None,
                sample_id=sample_id or "",
                parent_artifact_id=from_mfdb,
                operation_type="burst_filter",
                metadata={
                    "path": str(Path(out).resolve()),
                    "folder_path": str(Path(out).resolve()),
                    "n_in": result_json["n_in"],
                    "n_out": result_json["n_out"],
                    "select": list(select),
                    "query": query,
                },
                data_format="directory",
                db=db,
            )
            logging.info(f"Registered filtered burst selection as artifact: {new_artifact_id}")
            
    final_result = {
        "ok": True,
        "artifact_id": new_artifact_id,
        "n_in": result_json["n_in"],
        "n_out": result_json["n_out"],
        "out": result_json["out"],
    }
    click.echo(json.dumps(final_result, indent=2))


@cli.command("image")
@click.option('--from-mfdb', required=True, type=str, help="Source MFDB image/TTTR artifact ID.")
@click.option('--map', 'map_param', required=True, type=str, help="Parameter map to render (intensity, lifetime, etc.)")
@click.option('--select', '-s', multiple=True, type=str, help="Selection format: param:min-max")
@click.option('--query', '-q', type=str, help="Pandas eval query string.")
@click.option('--roi', type=click.Path(exists=True), help="TIFF file class mask ROI.")
@click.option('--out', '-o', required=True, type=click.Path(), help="Rendered map output image path (.png, .tiff).")
@click.option('--out-selection', type=click.Path(), help="Folder to write filtered burst sub-selection from ROI/gate.")
@click.option('--to-mfdb/--no-to-mfdb', default=False, show_default=True, help="Register the resulting image in MFDB.")
@click.option('--sample-id', type=str, help="MFDB sample ID to link the filtered selection to.")
@click.option('--db', 'db_path', type=click.Path(dir_okay=False), help="SQLite database path. Defaults to configured DB.")
@click.option('--skip-nth-row', type=int, default=1, show_default=True, help="Skip every Nth row (1 to load all).")
def image_cmd(from_mfdb, map_param, select, query, roi, out, out_selection, to_mfdb, sample_id, db_path, skip_nth_row):
    """Run parameter map rendering and ROI selection from MFDB image/TTTR data."""
    from chisurf.core.mfdb.database_resolver import resolve_database_path
    from chisurf.core.mfdb.repository import MFDatabase
    from chisurf.core.mfdb.result_registry import register_result
    
    resolved_db_path = db_path or resolve_database_path()
    logging.info(f"Opening database: {resolved_db_path}")
    
    with MFDatabase(resolved_db_path) as db:
        local_path = db.open_dataset(from_mfdb)
        if not local_path:
            click.echo(json.dumps({"error": f"Failed to resolve artifact: {from_mfdb}"}), err=True)
            sys.exit(1)
            
    logging.info(f"Resolved source artifact to: {local_path}")
    
    # Execute the pure CLI
    env = _get_ndxplorer_env()
    args = [sys.executable, "-m", "ndxplorer", "image", "--file", local_path, "--map", map_param, "--out", out, "--skip-nth-row", str(skip_nth_row)]
    for s in select:
        args += ["--select", s]
    if query:
        args += ["--query", query]
    if roi:
        args += ["--roi", roi]
    if out_selection:
        args += ["--out-selection", out_selection]
        
    logging.info(f"Running ndxplorer image command: {' '.join(args)}")
    res = subprocess.run(args, capture_output=True, text=True, env=env)
    
    if res.returncode != 0:
        click.echo(json.dumps({"error": f"Image subcommand failed: {res.stderr}"}), err=True)
        sys.exit(1)
        
    stdout_clean = res.stdout.strip()
    try:
        result_json = json.loads(stdout_clean)
    except Exception:
        start_idx = stdout_clean.find('{')
        end_idx = stdout_clean.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                result_json = json.loads(stdout_clean[start_idx:end_idx+1])
            except Exception as e:
                click.echo(json.dumps({"error": f"Failed to parse image output JSON block: {stdout_clean}, error: {e}"}), err=True)
                sys.exit(1)
        else:
            click.echo(json.dumps({"error": f"Failed to parse image output: {res.stdout}"}), err=True)
            sys.exit(1)
        
    new_artifact_id = None
    new_selection_artifact_id = None
    if to_mfdb:
        # Register in MFDB
        with MFDatabase(resolved_db_path) as db:
            try:
                db.validate_extensible_vocab("operation_type", "image_generation")
            except ValueError:
                db.register_vocabulary_value(
                    "operation_type",
                    "image_generation",
                    display_name="Image Generation",
                    description="Generate parameter map image from TTTR data",
                    is_active=True,
                )
            if out_selection:
                try:
                    db.validate_extensible_vocab("operation_type", "roi_selection")
                except ValueError:
                    db.register_vocabulary_value(
                        "operation_type",
                        "roi_selection",
                        display_name="ROI Selection",
                        description="Select bursts within a pixel region of interest (ROI)",
                        is_active=True,
                    )
            new_artifact_id = register_result(
                kind="processed_data",
                data=out,
                sample_id=sample_id or "",
                parent_artifact_id=from_mfdb,
                operation_type="image_generation",
                metadata={
                    "map": map_param,
                    "shape": result_json["shape"],
                    "select": list(select),
                    "query": query,
                    "n_selected_px": result_json["n_selected_px"],
                },
                db=db,
            )
            logging.info(f"Registered generated map image as artifact: {new_artifact_id}")
            
            if out_selection:
                new_selection_artifact_id = register_result(
                    kind="external_reference",
                    data=None,
                    sample_id=sample_id or "",
                    parent_artifact_id=from_mfdb,
                    operation_type="roi_selection",
                    metadata={
                        "path": str(Path(out_selection).resolve()),
                        "folder_path": str(Path(out_selection).resolve()),
                        "n_selected_px": result_json["n_selected_px"],
                    },
                    data_format="directory",
                    db=db,
                )
                logging.info(f"Registered ROI sub-selection as artifact: {new_selection_artifact_id}")
                
    final_result = {
        "ok": True,
        "artifact_id": new_artifact_id,
        "selection_artifact_id": new_selection_artifact_id,
        "map": map_param,
        "shape": result_json["shape"],
        "n_selected_px": result_json["n_selected_px"],
        "out": result_json["out"],
    }
    click.echo(json.dumps(final_result, indent=2))
