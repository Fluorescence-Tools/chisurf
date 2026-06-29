"""Pure computation layer for molecule-wise MLE analysis.

Runs the sm_image_mle CLI in a subprocess for each PTU file and collects results.
No Qt imports; safe to call from CLI or headless tests.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import MoleculeMleRequest, MoleculeMleResult


def _cli_script_path() -> Path:
    """Return path to the sm_image_mle CLI script."""
    plugin_dir = Path(__file__).parent.parent
    for candidate in ("sm_image_mle.py", "ptu_processor.py"):
        p = plugin_dir / candidate
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find CLI script (sm_image_mle.py or ptu_processor.py) "
        f"in {plugin_dir}"
    )


def _build_args(
    ptu_path: Path,
    irf_file: str,
    output_dir: Path,
    settings: "MoleculeMleSettings",  # type: ignore[name-defined]
    cli_script: Path,
) -> list[str]:
    """Build the subprocess argument list for one PTU file."""
    from .models import MoleculeMleSettings  # local import keeps module importable without deps

    s = settings
    parent_dir = ptu_path.parent.resolve()
    pattern = ptu_path.name
    output_file = parent_dir / f"{ptu_path.stem}_analysis" / "molecule_data.tsv"

    args: list[str] = [
        sys.executable,
        str(cli_script),
        "run",
        "--ptu-pattern", pattern,
        "--irf-file", str(irf_file),
    ]

    for ch in s.detector_chs:
        args.extend(["--detector-chs", str(ch)])

    args.extend([
        "--micro-time-range",
        str(s.micro_time_range[0]),
        str(s.micro_time_range[1]),
    ])
    args.extend(["--micro-time-binning", str(s.micro_time_binning)])
    args.extend(["--normalize-counts", str(s.normalize_counts)])
    args.extend(["--threshold", str(s.threshold)])
    args.extend(["--minlength", str(s.minlength)])
    args.extend(["--shift-sp", str(s.shift_sp)])
    args.extend(["--shift-ss", str(s.shift_ss)])
    args.extend(["--irf-threshold-fraction", str(s.irf_threshold_fraction)])

    args.append("--fit-initial-values")
    for val in s.fit_initial_values:
        args.append(str(val))

    args.append("--fit-fixed-flags")
    for flag in s.fit_fixed_flags:
        args.append(str(flag))

    args.extend(["--l1", str(s.l1)])
    args.extend(["--l2", str(s.l2)])
    args.append("--twoi-star" if s.twoi_star else "--no-twoi-star")
    args.append("--bifl-scatter" if s.bifl_scatter else "--no-bifl-scatter")
    args.extend(["--seg-sigma", str(s.seg_sigma)])
    args.extend(["--seg-threshold", str(s.seg_threshold)])
    args.extend(["--peak-footprint-size", str(s.peak_footprint_size)])
    args.extend(["--output-file", str(output_file)])
    args.extend(["--output-dir", str(parent_dir)])

    return args


def analyze_request(request: "MoleculeMleRequest") -> "MoleculeMleResult":  # type: ignore[name-defined]
    """Run molecule-wise MLE analysis for all files in *request*.

    Parameters
    ----------
    request:
        A :class:`~api.models.MoleculeMleRequest` describing which files to
        process, the IRF, and all analysis settings.

    Returns
    -------
    MoleculeMleResult
        Paths to individual TSV outputs and the merged joint TSV.
    """
    from .models import MoleculeMleRequest, MoleculeMleResult  # local import

    cli_script = _cli_script_path()
    output_paths: list[str] = []
    warnings: list[str] = []
    processed: list[str] = []

    for file_str in request.files:
        ptu_path = Path(file_str)
        parent_dir = ptu_path.parent.resolve()
        args = _build_args(ptu_path, request.irf_file, parent_dir, request.settings, cli_script)

        try:
            result = subprocess.run(
                args,
                cwd=str(parent_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                warnings.append(f"{ptu_path.name}: process exited {result.returncode}: {result.stderr[:200]}")
            else:
                processed.append(file_str)
                tsv = parent_dir / f"{ptu_path.stem}_analysis" / "molecule_data.tsv"
                output_paths.append(str(tsv))
        except Exception as exc:
            warnings.append(f"{ptu_path.name}: {exc}")

    # Merge individual TSVs into a joint output
    joint_tsv = ""
    if output_paths:
        try:
            import pandas as pd

            dfs = []
            for tsv_path in output_paths:
                p = Path(tsv_path)
                if p.exists():
                    dfs.append(pd.read_csv(p, sep="\t"))
                else:
                    warnings.append(f"TSV not found: {tsv_path}")
            if dfs:
                first = Path(output_paths[0]).parent.parent
                joint_path = first / "joint_output.tsv"
                pd.concat(dfs, ignore_index=True).to_csv(str(joint_path), sep="\t", index=False)
                joint_tsv = str(joint_path)
        except Exception as exc:
            warnings.append(f"Failed to merge TSVs: {exc}")

    return MoleculeMleResult(
        processed_files=processed,
        output_paths=output_paths,
        joint_tsv=joint_tsv,
        warnings=warnings,
    )
