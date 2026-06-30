"""Click CLI for FRET modeling.
"""

from __future__ import annotations

import os
import sys
import json
import click
import numpy as np

from ..core import av, engine, io, results, screening, evaluate, pair_selection


def _parse_pdb_paths(pdb_arg: str) -> list[str]:
    if not pdb_arg:
        return []
    if isinstance(pdb_arg, str):
        if "," in pdb_arg:
            return [p.strip() for p in pdb_arg.split(",")]
        return [pdb_arg]
    return list(pdb_arg)


@click.group()
def main():
    """FRET Modeling CLI: Docking, Screening, and Error Analysis."""
    pass


@main.command("info-backends")
def info_backends():
    """Print information about available and active AV backends."""
    ll_avail = "AVAILABLE" if av._HAS_LABELLIB else "not available"
    if av._HAS_LABELLIB:
        ll_ver = "unknown"
        if hasattr(av._ll, "__version__"):
            ll_ver = av._ll.__version__
        elif hasattr(av._ll, "version"):
            ll_ver = av._ll.version
        ll_avail += f"  (version: {ll_ver})"
    imp_avail = "AVAILABLE" if av._HAS_IMP_BFF else "not available"
    
    click.echo("AV backends:")
    click.echo(f"  LabelLib : {ll_avail}")
    click.echo(f"  IMP.bff  : {imp_avail}")
    active = "labellib" if av._LABELLIB_BACKEND else "imp-bff"
    click.echo(f"\nActive backend: {active}")


@main.command("info")
@click.option("--fps", required=True, help="Path to labeling.fps.json file.")
def info(fps: str):
    """Print a summary of positions and distances in an fps.json."""
    if not os.path.exists(fps):
        click.echo(f"Error: file not found: {fps}", err=True)
        sys.exit(1)
    
    positions, distances, score_sets, extra = io.read_fps_json(fps)
    click.echo(f"Positions: {len(positions)}")
    for pname, pdef in sorted(positions.items()):
        click.echo(
            f"  {pname}: chain={pdef.get('chain_identifier','')} "
            f"res={pdef.get('residue_seq_number',0)} "
            f"atom={pdef.get('atom_name','CA')} "
            f"L={pdef.get('linker_length',20)} "
            f"R={pdef.get('radius1',3.5)}"
        )
    click.echo(f"\nDistances: {len(distances)}")
    for dname, ddef in sorted(distances.items()):
        click.echo(
            f"  {dname}: {ddef.get('position1_name')} - {ddef.get('position2_name')}  "
            f"d={ddef.get('distance',0):.1f} "
            f"err=[{ddef.get('error_neg',5):.1f},{ddef.get('error_pos',5):.1f}] "
            f"type={ddef.get('distance_type','RDAMean')}"
        )
    if score_sets:
        click.echo(f"\nScore sets: {list(score_sets.keys())}")


@main.command("screen")
@click.option("--fps", required=True, help="Path to labeling.fps.json.")
@click.option("--pdb-dir", required=True, help="Directory containing structures to screen.")
@click.option("--output", required=True, help="Output CSV file path.")
@click.option("--n-threads", default=4, show_default=True, type=int, help="Number of worker threads.")
@click.option("--av-backend", default="auto", type=click.Choice(["auto", "labellib", "imp-bff"]), help="Accessible Volume backend.")
def screen_cmd(fps: str, pdb_dir: str, output: str, n_threads: int, av_backend: str):
    """Screen a structure library against FRET restraints."""
    av.select_backend(av_backend)
    positions, distances, _score_sets, _extra = io.read_fps_json(fps)
    scr_results = screening.screen_structure_library(
        pdb_dir,
        positions,
        distances,
        n_threads=n_threads,
    )
    results.write_screening_results_csv(scr_results, output)
    click.echo(f"Screening complete. Saved results to {output}")


@main.command("evaluate")
@click.option("--fps", required=True, help="Path to labeling.fps.json.")
@click.option("--pdb", required=True, help="PDB file path or directory.")
@click.option("--traj", default=None, help="Trajectory file path (DCD/XTC) for MDTraj evaluation.")
@click.option("--output", required=True, help="Output CSV file path.")
@click.option("--input-type", default="Single PDB File", type=click.Choice(["Single PDB File", "PDB Directory", "MDTraj Trajectory"]), help="Type of input.")
@click.option("--av-backend", default="auto", type=click.Choice(["auto", "labellib", "imp-bff"]), help="Accessible Volume backend.")
def evaluate_cmd(fps: str, pdb: str, traj: str | None, output: str, input_type: str, av_backend: str):
    """Run OLGA-style structure evaluations."""
    av.select_backend(av_backend)
    positions, distances, _, _ = io.read_fps_json(fps)
    evaluators = io.read_evaluators_json(fps)
    if not evaluators:
        # Construct DistanceEvaluators from distances
        from ..evaluators import DistanceEvaluator
        evaluators = [
            DistanceEvaluator(name, d["position1_name"], d["position2_name"], distance_type=d.get("distance_type", "RDAMean"))
            for name, d in distances.items()
        ]

    if input_type == "Single PDB File":
        res = evaluate.evaluate_structure(pdb, positions, evaluators)
        storage = evaluate.EvaluationStorage()
        storage.add_frame(os.path.basename(pdb), res)
        storage.to_csv(output)
    elif input_type == "PDB Directory":
        storage = evaluate.evaluate_directory(pdb, positions, evaluators)
        storage.to_csv(output)
    elif input_type == "MDTraj Trajectory":
        if not traj:
            click.echo("Error: Trajectory file (--traj) is required for MDTraj Trajectory evaluation.", err=True)
            sys.exit(1)
        storage = evaluate.evaluate_trajectory(pdb, traj, positions, evaluators)
        storage.to_csv(output)
    
    click.echo(f"Evaluation complete. Results written to {output}")


@main.command("select-pairs")
@click.option("--fps", required=True, help="Path to labeling.fps.json.")
@click.option("--pdb-dir", required=True, help="Directory of PDB ensemble structures.")
@click.option("--output", required=True, help="Output file to save decay report.")
@click.option("--max-pairs", default=3, show_default=True, type=int, help="Maximum number of pairs to select.")
@click.option("--err", default=5.0, type=float, help="FRET distance measurement error.")
@click.option("--av-backend", default="auto", type=click.Choice(["auto", "labellib", "imp-bff"]), help="Accessible Volume backend.")
def select_pairs_cmd(fps: str, pdb_dir: str, output: str, max_pairs: int, err: float, av_backend: str):
    """Run informative pair selection on ensemble."""
    av.select_backend(av_backend)
    positions, distances, _, _ = io.read_fps_json(fps)
    
    rmsds, filenames = pair_selection.compute_rmsd_matrix_from_pdb_dir(pdb_dir)
    effs, pair_names = pair_selection.compute_efficiency_matrix_from_evaluators(
        pdb_dir, positions, distances
    )
    effs_clean, rmsds_clean, valid_indices = pair_selection.preprocess_efficiency_matrix(
        effs, rmsds
    )
    clean_pair_names = [pair_names[i] for i in valid_indices]

    selected_indices, decay = pair_selection.select_informative_pairs(
        effs_clean, rmsds_clean, err=err, max_pairs=max_pairs
    )
    selected_pair_names = [clean_pair_names[i] for i in selected_indices]
    pair_selection.write_pair_selection_report(
        selected_pair_names, decay, output, rmsds.mean()
    )
    click.echo(f"Pair selection complete. Saved decay report to {output}")


# ---------------------------------------------------------------------------
# IMP + IMP.bff engine (the maintained backend; thin shim around IMP.pmi)
# ---------------------------------------------------------------------------


@main.group("imp")
def imp_group():
    """FRET docking via the IMP + IMP.bff engine (PMI-based)."""
    pass


def _split_pdbs(pdb):
    return [p.strip() for p in pdb.split(",")] if "," in pdb else [pdb]


@imp_group.command("info")
def imp_info():
    """Report IMP/IMP.bff backend availability."""
    from ..api import operations as ops
    click.echo(json.dumps(ops.backend_info(), indent=2))


@imp_group.command("score")
@click.option("--pdb", required=True, help="PDB file(s), comma-separated for multi-body.")
@click.option("--fps", "fps_json", required=True, help="fps.json labelling/distance file.")
@click.option("--score-set", default="", help="Named score set (default: all).")
@click.option("--out", "output_csv", default=None, help="Write per-pair distances CSV.")
@click.option("--mean-position/--full-av", default=False, help="Fast mean-AV vs full AV recompute.")
def imp_score(pdb, fps_json, score_set, output_csv, mean_position):
    """Score a structure against FRET restraints."""
    from ..api import operations as ops
    res = ops.score({"pdb_paths": _split_pdbs(pdb), "fps_json": fps_json,
                     "score_set": score_set, "output_csv": output_csv,
                     "mean_position_restraint": mean_position})
    click.echo(json.dumps(res, indent=2))


@imp_group.command("dock")
@click.option("--pdb", required=True, help="One PDB per rigid body, comma-separated.")
@click.option("--fps", "fps_json", required=True)
@click.option("--out", "output_dir", required=True, help="Output directory (RMF/PDB/CSV).")
@click.option("--frames", "n_frames", default=500, type=int)
@click.option("--mc-steps", default=10, type=int)
@click.option("--score-set", default="")
@click.option("--n-best", default=20, type=int)
@click.option("--anneal/--no-anneal", "simulated_annealing", default=False)
@click.option("--fixed-body", default=0, type=int)
@click.option("--sigma-da", "sigma_da", default=6.0, type=float,
              help="Mean-position transfer-function width (Angstrom).")
@click.option("--method", default="minimize", type=click.Choice(["minimize", "mc"]),
              help="minimize = fast IMP conjugate-gradient docking (default); mc = replica-exchange MC.")
def imp_dock(pdb, fps_json, output_dir, n_frames, mc_steps, score_set,
             n_best, simulated_annealing, fixed_body, sigma_da, method):
    """Run FRET-restrained rigid-body docking (minimisation or Monte-Carlo)."""
    from ..api import operations as ops
    res = ops.dock({"pdb_paths": _split_pdbs(pdb), "fps_json": fps_json,
                    "output_dir": output_dir, "n_frames": n_frames,
                    "mc_steps": mc_steps, "score_set": score_set, "n_best": n_best,
                    "simulated_annealing": simulated_annealing, "fixed_body": fixed_body,
                    "sigma_da": sigma_da, "method": method})
    click.echo(json.dumps(res, indent=2))


@imp_group.command("dock-project")
@click.option("--project", "project_path", required=True,
              help="Docking project .json (bundles PDBs, fps.json and parameters).")
@click.option("--out", "output_dir", default=None,
              help="Override the project's output directory.")
@click.option("--frames", "n_frames", default=None, type=int,
              help="Override the number of MC frames (e.g. a quick smoke run).")
@click.option("--mc-steps", default=None, type=int, help="Override MC steps per frame.")
@click.option("--n-best", default=None, type=int, help="Override number of best models kept.")
@click.option("--method", default=None, type=click.Choice(["minimize", "mc"]),
              help="Override the docking method (minimize / mc).")
def imp_dock_project(project_path, output_dir, n_frames, mc_steps, n_best, method):
    """Run FRET docking from a saved project file."""
    from ..api import operations as ops
    overrides = {"output_dir": output_dir, "n_frames": n_frames,
                 "mc_steps": mc_steps, "n_best": n_best, "method": method}
    res = ops.dock_project(project_path, overrides)
    click.echo(json.dumps(res, indent=2))


@imp_group.command("refine")
@click.option("--pdb", required=True)
@click.option("--fps", "fps_json", required=True)
@click.option("--out", "output_dir", required=True)
@click.option("--steps", default=500, type=int)
@click.option("--score-set", default="")
def imp_refine(pdb, fps_json, output_dir, steps, score_set):
    """Conjugate-gradient local refinement of a pose."""
    from ..api import operations as ops
    res = ops.refine({"pdb_paths": _split_pdbs(pdb), "fps_json": fps_json,
                      "output_dir": output_dir, "steps": steps, "score_set": score_set})
    click.echo(json.dumps(res, indent=2))


@imp_group.command("screen")
@click.option("--pdb", required=True, help="PDB files/dirs, comma-separated.")
@click.option("--fps", "fps_json", required=True)
@click.option("--score-set", default="")
@click.option("--out", "output_csv", default=None)
def imp_screen(pdb, fps_json, score_set, output_csv):
    """Score and rank a structure library."""
    from ..api import operations as ops
    res = ops.screen({"pdb_inputs": _split_pdbs(pdb), "fps_json": fps_json,
                      "score_set": score_set, "output_csv": output_csv})
    click.echo(json.dumps(res, indent=2))


@imp_group.command("errors")
@click.option("--pdb", required=True)
@click.option("--fps", "fps_json", required=True)
@click.option("--out", "output_dir", required=True)
@click.option("--trials", "n_trials", default=10, type=int)
@click.option("--frames", "n_frames", default=200, type=int)
@click.option("--score-set", default="")
@click.option("--method", default="minimize", type=click.Choice(["minimize", "mc"]))
@click.option("--workers", "n_workers", default=None, type=int,
              help="Parallel worker processes (default: CPU count; 1 = serial).")
def imp_errors(pdb, fps_json, output_dir, n_trials, n_frames, score_set, method, n_workers):
    """Repeated-trial docking error estimation (trials run in parallel)."""
    from ..api import operations as ops
    res = ops.estimate_errors({"pdb_paths": _split_pdbs(pdb), "fps_json": fps_json,
                               "output_dir": output_dir, "n_trials": n_trials,
                               "n_frames": n_frames, "score_set": score_set,
                               "method": method, "n_workers": n_workers})
    click.echo(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
