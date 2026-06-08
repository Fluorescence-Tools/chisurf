"""Click CLI for FRET modeling.
"""

from __future__ import annotations

import os
import sys
import json
import click
import numpy as np

from ..core import av, docking, engine, io, results, sampling, screening, refine, bootstrap, evaluate, pair_selection


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


@main.command("dock")
@click.option("--fps", required=True, help="Path to labeling.fps.json.")
@click.option("--pdb", required=True, help="Path to input PDB (or comma-separated paths for multi-body).")
@click.option("--output", required=True, help="Output directory to save docking results.")
@click.option("--n-trials", default=3, show_default=True, type=int, help="Number of independent docking trials.")
@click.option("--max-iterations", default=50000, show_default=True, type=int, help="Maximum number of simulation iterations.")
@click.option("--max-force", default=100.0, show_default=True, type=float, help="Maximum simulation force threshold.")
@click.option("--k-clash", default=10.0, show_default=True, type=float, help="Spring constant for inter-body clash repulsion.")
@click.option("--f-tol", default=0.1, type=float, help="Force tolerance for convergence.")
@click.option("--t-tol", default=0.01, type=float, help="Torque tolerance for convergence.")
@click.option("--av-backend", default="auto", type=click.Choice(["auto", "labellib", "imp-bff"]), help="Accessible Volume backend.")
def dock(fps: str, pdb: str, output: str, n_trials: int, max_iterations: int, max_force: float, k_clash: float, f_tol: float, t_tol: float, av_backend: str):
    """Run FRET-restrained rigid-body docking."""
    av.select_backend(av_backend)
    pdb_paths = _parse_pdb_paths(pdb)
    positions, distances, _score_sets, _extra = io.read_fps_json(fps, pdb_paths=pdb_paths)
    
    params = engine.SpringParameters(
        max_iterations=max_iterations,
        max_force=max_force,
        k_clash=k_clash,
        F_tolerance=f_tol,
        T_tolerance=t_tol,
    )
    res_list, avs, bodies = docking.run_docking(
        pdb_paths,
        positions,
        distances,
        params=params,
        n_trials=n_trials,
    )
    
    os.makedirs(output, exist_ok=True)
    atoms_per_body = [b.atoms_local for b in bodies]
    results.write_docking_results_pdb(res_list, atoms_per_body, output)
    
    summary_path = os.path.join(output, "summary.json")
    summary = [
        {
            "trial": i,
            "converged": sr.converged,
            "iterations": sr.iterations,
            "energy": sr.energy,
            "clash_energy": sr.clash_energy,
            "restraint_energy": sr.restraint_energy,
        }
        for i, sr in enumerate(res_list)
    ]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    click.echo(f"Wrote {len(res_list)} docking results to {output}/")
    click.echo(f"Converged: {sum(1 for sr in res_list if sr.converged)} / {len(res_list)}")


@main.command("refine")
@click.option("--fps", required=True, help="Path to labeling.fps.json.")
@click.option("--pdb", required=True, help="Path to input PDB.")
@click.option("--output", required=True, help="Output directory to save refined structure.")
@click.option("--n-cycles", default=3, show_default=True, type=int, help="Number of refinement cycles.")
@click.option("--max-iterations", default=10000, type=int, help="Max iterations per spring cycle.")
@click.option("--k-clash", default=10.0, type=float, help="Clash spring constant.")
@click.option("--av-backend", default="auto", type=click.Choice(["auto", "labellib", "imp-bff"]), help="Accessible Volume backend.")
def refine_cmd(fps: str, pdb: str, output: str, n_cycles: int, max_iterations: int, k_clash: float, av_backend: str):
    """Run iterative refinement (docking + AV recalculation)."""
    av.select_backend(av_backend)
    pdb_paths = _parse_pdb_paths(pdb)
    positions, distances, _score_sets, _extra = io.read_fps_json(fps, pdb_paths=pdb_paths)

    atoms_xyzr_list = [av.load_structure_with_vdw(p) for p in pdb_paths]
    atoms_xyzr = atoms_xyzr_list[0] if atoms_xyzr_list else np.zeros((0, 4))
    avs = av.compute_avs_for_structure(atoms_xyzr, positions, pdb_path=pdb_paths)

    body_map = {pname: int(pdef.get("body_id", 0)) for pname, pdef in positions.items()}
    n_bodies = max(max(body_map.values()) + 1 if body_map else 1, len(pdb_paths))

    bodies = []
    for bi in range(n_bodies):
        b_atoms_xyzr = atoms_xyzr_list[bi] if bi < len(atoms_xyzr_list) else atoms_xyzr
        b_xyz = b_atoms_xyzr[:, :3]
        com = b_xyz.mean(axis=0) if b_xyz.shape[0] > 0 else np.zeros(3)
        local_coords = b_xyz - com
        local_xyzr = np.column_stack([local_coords, b_atoms_xyzr[:, 3]])
        rb = engine.RigidBody(
            name=f"body_{bi}",
            atoms_local=local_xyzr,
            com=com.copy(),
            rotation=np.eye(3),
            translation=com.copy(),
            mass=float(b_xyz.shape[0]) if b_xyz.shape[0] > 0 else 1.0,
            inertia=np.eye(3) * 1000.0,
        )
        bodies.append(rb)

    restraints = []
    for dname, ddef in distances.items():
        p1 = ddef["position1_name"]
        p2 = ddef["position2_name"]
        if p1 not in avs or p2 not in avs:
            continue
        if not avs[p1].has_volume or not avs[p2].has_volume:
            continue
        b1 = body_map.get(p1, 0)
        b2 = body_map.get(p2, 0)
        offset_a = avs[p1].mean_position - bodies[b1].com
        offset_b = avs[p2].mean_position - bodies[b2].com
        rst = engine.DistanceRestraint(
            name=dname,
            body_a=b1,
            offset_a=offset_a,
            body_b=b2,
            offset_b=offset_b,
            distance_exp=float(ddef.get("distance", 0.0)),
            error_neg=float(ddef.get("error_neg", 5.0)),
            error_pos=float(ddef.get("error_pos", 5.0)),
            distance_type=str(ddef.get("distance_type", "RDAMean")),
            forster_radius=float(ddef.get("Forster_radius", 52.0)),
            active=True,
        )
        restraints.append(rst)

    params = engine.SpringParameters(
        max_iterations=max_iterations,
        k_clash=k_clash,
    )
    final_bodies = refine.run_refinement(
        bodies,
        restraints,
        positions,
        atoms_xyzr,
        params=params,
        n_cycles=n_cycles,
    )

    os.makedirs(output, exist_ok=True)
    for bi, body in enumerate(final_bodies):
        out_path = os.path.join(output, f"refined_body_{bi}.pdb")
        io.write_pdb(body.global_coords(), out_path)
    click.echo(f"Refinement complete. Saved {len(final_bodies)} body structures to {output}/")


@main.command("bootstrap")
@click.option("--fps", required=True, help="Path to labeling.fps.json.")
@click.option("--pdb", required=True, help="Path to input PDB.")
@click.option("--output", required=True, help="Output directory to save error distribution.")
@click.option("--n-bootstrap", default=100, show_default=True, type=int, help="Number of bootstrap iterations.")
@click.option("--max-iterations", default=10000, type=int, help="Max iterations per fit.")
@click.option("--k-clash", default=10.0, type=float, help="Clash spring constant.")
@click.option("--av-backend", default="auto", type=click.Choice(["auto", "labellib", "imp-bff"]), help="Accessible Volume backend.")
def bootstrap_cmd(fps: str, pdb: str, output: str, n_bootstrap: int, max_iterations: int, k_clash: float, av_backend: str):
    """Run parametric bootstrap error estimation."""
    av.select_backend(av_backend)
    positions, distances, _score_sets, _extra = io.read_fps_json(fps)
    params = engine.SpringParameters(
        max_iterations=max_iterations,
        k_clash=k_clash,
    )
    boot_res = bootstrap.run_bootstrap(
        pdb,
        positions,
        distances,
        params=params,
        n_bootstrap=n_bootstrap,
    )
    os.makedirs(output, exist_ok=True)
    results.write_bootstrap_results(boot_res, output)
    click.echo(f"Bootstrap complete. Results written to {output}/")


@main.command("sample")
@click.option("--fps", required=True, help="Path to labeling.fps.json.")
@click.option("--pdb", required=True, help="Path to input PDB.")
@click.option("--output", required=True, help="Output directory to save sample structures.")
@click.option("--n-samples", default=1000, show_default=True, type=int, help="Number of Monte Carlo samples.")
@click.option("--step-size", default=0.5, type=float, help="Metropolis step size in angstroms/radians.")
@click.option("--av-backend", default="auto", type=click.Choice(["auto", "labellib", "imp-bff"]), help="Accessible Volume backend.")
def sample_cmd(fps: str, pdb: str, output: str, n_samples: int, step_size: float, av_backend: str):
    """Run Metropolis Monte Carlo sampling."""
    av.select_backend(av_backend)
    positions, distances, _score_sets, _extra = io.read_fps_json(fps)
    
    atoms_xyzr = av.load_structure_with_vdw(pdb)
    com = atoms_xyzr[:, :3].mean(axis=0)
    local_xyzr = atoms_xyzr.copy()
    local_xyzr[:, :3] -= com
    body = engine.RigidBody(
        name="body_0",
        atoms_local=local_xyzr,
        com=com.copy(),
        rotation=np.eye(3),
        translation=com.copy(),
        mass=float(atoms_xyzr.shape[0]) if atoms_xyzr.shape[0] > 0 else 1.0,
        inertia=np.eye(3) * 1000.0,
    )
    
    avs = av.compute_avs_for_structure(atoms_xyzr, positions, pdb_path=pdb)
    restraints = []
    for dname, ddef in distances.items():
        p1 = ddef["position1_name"]
        p2 = ddef["position2_name"]
        if p1 not in avs or p2 not in avs:
            continue
        offset_a = avs[p1].mean_position - body.com
        offset_b = avs[p2].mean_position - body.com
        rst = engine.DistanceRestraint(
            name=dname,
            body_a=0,
            offset_a=offset_a,
            body_b=0,
            offset_b=offset_b,
            distance_exp=float(ddef.get("distance", 0.0)),
            error_neg=float(ddef.get("error_neg", 5.0)),
            error_pos=float(ddef.get("error_pos", 5.0)),
            distance_type=str(ddef.get("distance_type", "RDAMean")),
            forster_radius=float(ddef.get("Forster_radius", 52.0)),
            active=True,
        )
        restraints.append(rst)

    samples = sampling.run_metropolis_sampling(
        [body],
        restraints,
        n_samples=n_samples,
        step_size=step_size,
    )
    os.makedirs(output, exist_ok=True)
    out_pdb = os.path.join(output, "sampled_trajectory.pdb")
    
    with open(out_pdb, "w") as f:
        for idx, s in enumerate(samples):
            f.write(f"MODEL     {idx + 1:4d}\n")
            b_global = s.rotations[0] @ body.atoms_local[:, :3].T + s.translations[0][:, np.newaxis]
            for atom_idx, xyz in enumerate(b_global.T):
                f.write(f"ATOM  {atom_idx+1:5d}  CA  ALA A   1    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00\n")
            f.write("ENDMDL\n")
            
    click.echo(f"Generated {len(samples)} samples. Saved trajectory to {out_pdb}")


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


if __name__ == "__main__":
    main()
