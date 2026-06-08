"""Example script demonstrating programmatic usage of the FRET Superset plugin.

Covers:
1. Rigid-body docking using SpringEngine
2. Refinement cycles (Docking + AV recomputation)
3. Parametric bootstrap for error estimation
4. OLGA-style trajectory/structure evaluation
5. OLGA-style informative pair selection
"""

import os
import numpy as np
from chisurf.plugins.modelling.fret import (
    av,
    docking,
    engine,
    io,
    refine,
    bootstrap,
    evaluate,
    pair_selection,
)

# Paths to example files in the repository
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "projects", "t4l_chimol", "data")
)
PDB_PATH = os.path.join(DATA_DIR, "3GUN.pdb")
FPS_PATH = os.path.join(DATA_DIR, "fret.fps.json")


def run_docking_example():
    print("--- 1. Running FRET Docking Example ---")
    positions, dists, score_sets, extra = io.read_fps_json(FPS_PATH)

    params = engine.SpringParameters(
        max_iterations=1000,
        max_force=50.0,
        k_clash=10.0,
    )

    results, av_dict, bodies = docking.run_docking(
        PDB_PATH, positions, dists, params=params, n_trials=2
    )

    print(f"Completed {len(results)} docking trials.")
    for idx, r in enumerate(results):
        print(f"  Trial {idx}: converged={r.converged}, energy={r.energy:.2f}")


def run_refinement_example():
    print("\n--- 2. Running Refinement Example ---")
    positions, dists, score_sets, extra = io.read_fps_json(FPS_PATH)

    atoms_xyzr = av.load_structure_with_vdw(PDB_PATH)
    bodies = evaluate.make_bodies_for_structure(atoms_xyzr, positions)
    avs = av.compute_avs_for_structure(atoms_xyzr, positions, pdb_path=PDB_PATH)

    body_map = {pname: int(pdef.get("body_id", 0)) for pname, pdef in positions.items()}
    restraints = []
    for dname, ddef in dists.items():
        p1 = ddef["position1_name"]
        p2 = ddef["position2_name"]
        if p1 not in avs or p2 not in avs:
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
        )
        restraints.append(rst)

    params = engine.SpringParameters(
        max_iterations=1000,
        max_force=50.0,
        k_clash=10.0,
    )

    # Run 2 refinement cycles
    refined_bodies = refine.run_refinement(
        bodies=bodies,
        restraints=restraints,
        positions=positions,
        atoms_xyzr=atoms_xyzr,
        params=params,
        n_cycles=2,
    )
    print("Refinement cycle complete.")


def run_bootstrap_example():
    print("\n--- 3. Running Parametric Bootstrap Example ---")
    positions, dists, score_sets, extra = io.read_fps_json(FPS_PATH)

    params = engine.SpringParameters(
        max_iterations=500,
        max_force=50.0,
        k_clash=10.0,
    )

    print("Running parametric bootstrap (2 iterations)...")
    res = bootstrap.run_bootstrap(
        PDB_PATH,
        positions,
        dists,
        params=params,
        n_bootstrap=2,
    )

    print(f"Bootstrap complete. Model distance statistics:")
    for key in res.model_distances_mean:
        mean_d = res.model_distances_mean[key]
        std_d = res.model_distances_std[key]
        print(f"  {key}: {mean_d:.2f} +/- {std_d:.2f} Å")


def run_evaluation_example():
    print("\n--- 4. Running OLGA Evaluation Example ---")
    positions, dists, score_sets, extra = io.read_fps_json(FPS_PATH)
    evaluators = io.read_evaluators_json(FPS_PATH)

    if not evaluators:
        print("No evaluators defined in project.json, using standard distance evaluators.")
        # Create standard evaluators programmatically
        from chisurf.plugins.modelling.fret.evaluators.distance import DistanceEvaluator
        evaluators = [
            DistanceEvaluator(
                name="dist_D1_D2",
                position1="D1",
                position2="D2",
                distance_type="RDAMean",
            )
        ]

    # Evaluate the single PDB file
    results = evaluate.evaluate_structure(PDB_PATH, positions, evaluators)
    for name, r in results.items():
        print(f"  Evaluated '{name}': {r.value:.4f} {r.unit}")


def run_pair_selection_example():
    print("\n--- 5. Running Informative Pair Selection Example ---")
    positions, dists, score_sets, extra = io.read_fps_json(FPS_PATH)

    # We mock a small trajectory of 2 frames by using PDB_PATH as a dummy PDB directory
    pdb_dir = os.path.dirname(PDB_PATH)

    # 1. Compute RMSD matrix
    rmsds, filenames = pair_selection.compute_rmsd_matrix_from_pdb_dir(
        pdb_dir, pattern="*3GUN.pdb"
    )

    # 2. Compute FRET efficiencies
    effs, pair_names = pair_selection.compute_efficiency_matrix_from_evaluators(
        pdb_dir, positions, dists, pattern="*3GUN.pdb"
    )

    # 3. Preprocess matrices
    effs_clean, rmsds_clean, valid_indices = pair_selection.preprocess_efficiency_matrix(
        effs, rmsds
    )

    clean_pair_names = [pair_names[i] for i in valid_indices]

    # 4. Run pair selection
    selected_indices, decay = pair_selection.select_informative_pairs(
        effs_clean, rmsds_clean, err=5.0, max_pairs=3
    )

    selected_pair_names = [clean_pair_names[i] for i in selected_indices]
    print("Informative pairs selected in order:")
    for idx, name in enumerate(selected_pair_names):
        print(f"  {idx + 1}: {name}")


if __name__ == "__main__":
    run_docking_example()
    run_refinement_example()
    run_bootstrap_example()
    run_evaluation_example()
    run_pair_selection_example()
