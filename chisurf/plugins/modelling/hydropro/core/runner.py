"""Headless HYDROPRO / HYDRO++ execution and report parsing (no Qt).

This module assembles the program's main input file, runs the external
executable for each structure, and parses the resulting ``*.res`` report for the
translational diffusion coefficient. It is shared by the GUI, the CLI and the
RPC service so the same logic is exercised in every entry point.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .settings import HydroProSettings

# Optional progress/log callbacks used to drive a UI without coupling to Qt.
LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]

_DIFFUSION_RE = re.compile(
    r"(?i)diffusion\s+coefficient\s*:\s*"
    r"([-+]?\d*\.\d+E[-+]?\d+|[-+]?\d+\.\d+|[-+]?\d+)"
)


@dataclass
class HydroResult:
    """Result of a single HYDRO run."""

    struct_file: str
    diffusion_coefficient: Optional[float]


def parse_diffusion_coefficient(res_path: Path) -> Optional[float]:
    """Extract the translational diffusion coefficient (cm²/s) from a ``*.res``.

    Scans for the first ``diffusion coefficient: <value>`` line and returns the
    numeric value, or ``None`` if it cannot be found or parsed.
    """
    try:
        with Path(res_path).open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = _DIFFUSION_RE.search(line)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue
    except FileNotFoundError:
        return None
    return None


def construct_input_file(struct_files: List[Path], input_path: Path) -> List[str]:
    """Write a HYDRO++ main input file (one block per structure).

    Returns the list of output basenames (without extension) used per case.
    """
    output_basenames: List[str] = []
    lines: List[str] = []
    for idx, struct_file in enumerate(struct_files, start=1):
        title = struct_file.stem[:20].ljust(20)
        basename = f"case{idx:03d}"
        output_basenames.append(basename)
        lines.extend([
            title,
            basename.ljust(30),
            str(struct_file).ljust(30),
            "12",        # ICASE (12 = recommended option for HYDRO++)
            "20.0",      # Temperature (°C)
            "0.010",     # Solvent viscosity (poise)
            "100000.0",  # Molecular weight (Da)
            "0.74",      # Partial specific volume (cm^3/g)
            "1.0",       # Solution density (g/cm^3)
            "0",         # NQ
            "0.0",       # HMAX
            "0",         # distance-distribution intervals
            "0.0",       # RMAX
            "0",         # NTRIALS
            "1",         # IDIF
        ])
    lines.append("*")
    input_path.write_text("\n".join(lines))
    return output_basenames


def write_hydropro_input(
    struct_file: Path, job_dir: Path, settings: HydroProSettings
) -> Tuple[str, Path]:
    """Write a ``hydropro.dat`` for one structure. Returns (basename, path)."""
    generic = struct_file.stem
    title = (generic[:28] + "…") if len(generic) > 28 else generic

    lines: List[str] = [
        f"{title}                        !TITLE (CHAR*20)",
        f"{generic}                        !FILENAME (base for outputs)",
        f"{struct_file.name}        !INPUT PDB filename (relative)",
        f"{settings.indmode}               !INDMODE: 1 atomic/shell, 2 residue/shell, 4 residue/bead",
        f"{settings.aer},            !AER (Å) hydrodynamic radius of primary elements",
    ]
    if settings.indmode in (1, 2):
        if settings.nsig == -1:
            lines.append("-1,              !NSIG=-1 for automatic sigma range")
        else:
            lines.append(f"{int(settings.nsig)},              !NSIG (>=3 typical 5-8)")
            lines.append(f"{settings.sigmin},            !SIGMIN (Å) minibead radius")
            lines.append(f"{settings.sigmax},            !SIGMAX (Å) minibead radius")
    lines += [
        f"{settings.t},            !T (°C)",
        f"{settings.eta},           !ETA (poise)",
        f"{settings.rm},        !RM (Da)",
        f"{settings.vbar},           !VBAR (cm3/g)",
        f"{settings.rho},            !RHO (g/cm3)",
        f"{int(settings.nq)}              !NQ: 0 omit, -1 automatic, >0 specify QMAX",
    ]
    if settings.nq and settings.nq > 0:
        lines.append(f"{settings.qmax}              !QMAX (cm^-1)")
    lines.append(f"{int(settings.ns)}              !NS: 0 omit, -1 automatic, >0 specify RMAX")
    if settings.ns and settings.ns > 0:
        lines.append(f"{settings.rmax}              !RMAX (cm)")
    lines.append(f"{int(settings.ntrials)},              !NTRIALS for covolume (MC)")
    lines.append(f"{int(settings.idif)}               !IDIF=1 for full diffusion tensors")
    lines.append("*                                    !End of file")

    input_path = job_dir / "hydropro.dat"
    input_path.write_text("\n".join(lines))
    return generic, input_path


def _noop_log(_msg: str) -> None:
    pass


def run_hydro(
    struct_files: List[Path],
    settings: HydroProSettings,
    exe_path: Path,
    work_dir: Optional[Path] = None,
    *,
    on_log: Optional[LogCallback] = None,
    on_progress: Optional[ProgressCallback] = None,
    should_cancel: Optional[CancelCallback] = None,
    timeout: int = 300,
) -> List[HydroResult]:
    """Run HYDROPRO / HYDRO++ over ``struct_files`` and return per-file results.

    The executable flavour is chosen from its filename (``hydropro`` → HYDROPRO,
    otherwise HYDRO++). Each structure runs in its own job directory under
    ``work_dir`` (default ``~/.hydropp_gui``). ``on_log`` / ``on_progress`` /
    ``should_cancel`` let a UI observe and interrupt the run without this module
    depending on Qt.
    """
    log = on_log or _noop_log
    exe_path = Path(exe_path)
    work_dir = Path(work_dir) if work_dir is not None else (Path.home() / ".hydropp_gui")
    work_dir.mkdir(parents=True, exist_ok=True)

    total = len(struct_files)
    is_hydropro = "hydropro" in exe_path.name.lower()
    results: List[HydroResult] = []

    for idx, struct_file in enumerate(struct_files, start=1):
        if should_cancel and should_cancel():
            break
        struct_file = Path(struct_file)
        job_dir = work_dir / f"job_{idx:03d}"
        job_dir.mkdir(parents=True, exist_ok=True)

        if is_hydropro:
            job_struct = job_dir / struct_file.name
            try:
                shutil.copyfile(str(struct_file), str(job_struct))
            except OSError as ex:
                log(f"ERROR: failed to copy structure file: {ex}")
            generic, hp_input = write_hydropro_input(job_struct, job_dir, settings)
            input_to_feed = hp_input.name
            expected_res = job_dir / f"{generic}-res.txt"
        else:
            input_file = job_dir / "hydro_input.dat"
            basenames = construct_input_file([struct_file], input_file)
            basename = basenames[0] if basenames else "case001"
            input_to_feed = input_file.name
            expected_res = job_dir / f"{basename}-res.txt"

        log(f"\n=== Job {idx}/{total}: {struct_file} ===")
        log(f"Working directory: {job_dir}")
        try:
            process = subprocess.Popen(
                [str(exe_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(job_dir),
            )
            out_bytes, err_bytes = process.communicate(
                input=f"{input_to_feed}\n".encode("utf-8"), timeout=timeout
            )
            out_text = out_bytes.decode("utf-8", errors="ignore") if out_bytes else ""
            err_text = err_bytes.decode("utf-8", errors="ignore") if err_bytes else ""
            if out_text.strip():
                log(out_text.strip())
            if err_text.strip():
                log("\n[STDERR]\n" + err_text.strip())
        except FileNotFoundError:
            log("ERROR: Executable not found.")
            raise
        except subprocess.TimeoutExpired:
            process.kill()
            log(f"ERROR: Timeout running {struct_file.name}.")
            raise

        coeff = parse_diffusion_coefficient(expected_res)
        results.append(HydroResult(str(struct_file), coeff))
        if coeff is not None:
            log(f"Result: diffusion coefficient = {coeff:.3e} cm^2/s")
        else:
            log("Result: diffusion coefficient not found (see output above)")

        if on_progress:
            on_progress(idx, total)

    return results


__all__ = [
    "HydroResult",
    "parse_diffusion_coefficient",
    "construct_input_file",
    "write_hydropro_input",
    "run_hydro",
]
