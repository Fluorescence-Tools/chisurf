"""ServiceDispatcher-compatible RPC handlers for H2MM burst analysis."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from ..api.contract import (
    METHOD_COMPUTE,
    METHOD_DESCRIBE_CONTRACT,
    METHOD_PREPARE_WORKFLOW,
    contract_descriptor,
    service_success,
)
from ..api.models import H2mmResult, H2mmSettings, StateFitSummary, StreamSettings
from ..api.serialization import settings_from_dict, to_jsonable


def register_services(dispatcher: Any) -> None:
    """Register H2MM RPC handlers with a ServiceDispatcher."""
    dispatcher.register(METHOD_COMPUTE, lambda params: compute_handler(**(params or {})))
    dispatcher.register(METHOD_PREPARE_WORKFLOW, lambda params: prepare_workflow_handler(**(params or {})))
    dispatcher.register(METHOD_DESCRIBE_CONTRACT, lambda params: contract_handler(**(params or {})))


def list_methods() -> dict[str, str]:
    """Return H2MM RPC method descriptions."""
    return {
        METHOD_COMPUTE: "Fit photon-by-photon HMM (H2MM) models over burst data.",
        METHOD_PREPARE_WORKFLOW: "Resolve H2MM settings and folders from a burst workflow context.",
        METHOD_DESCRIBE_CONTRACT: "Return the H2MM workflow contract.",
    }


# ---------------------------------------------------------------------------
# Shared analysis runner (also used by the CLI)
# ---------------------------------------------------------------------------


class H2mmAnalysisBundle:
    """In-memory container returned to the GUI (not serialised over RPC)."""

    def __init__(self, analysis, data, settings):
        self.analysis = analysis
        self.data = data
        self.settings = settings


def run_analysis(
    settings: H2mmSettings,
    analysis_folder: str | pathlib.Path | None = None,
    files: list[str] | None = None,
    pattern: str = "*.bur",
) -> tuple[H2mmResult, H2mmAnalysisBundle]:
    """Load bursts, fit H2MM models, and build a serialisable result.

    Parameters
    ----------
    settings : H2mmSettings
        Analysis configuration.
    analysis_folder : str or Path, optional
        Folder to search for ``.bur`` files (``pattern``).
    files : list of str, optional
        Explicit ``.bur`` file paths (take precedence over the folder).
    pattern : str
        Glob for ``.bur`` files when ``analysis_folder`` is used.

    Returns
    -------
    result : H2mmResult
        JSON-serialisable summary.
    bundle : H2mmAnalysisBundle
        The full in-memory analysis (for GUI plotting).
    """
    from ..core import photons as photons_mod
    from ..core.analysis import analyze
    from ..core.photons import StreamDef, bursts_from_dataframe

    bur_paths = _resolve_bur_files(files, analysis_folder, pattern)
    if not bur_paths:
        raise ValueError("no .bur files found for H2MM analysis")

    df = photons_mod.load_bur_dataframe(bur_paths)
    data_dir = pathlib.Path(bur_paths[0]).parent
    tttrs = _load_tttrs(df, data_dir, settings.file_type)

    base_time_s = _macro_resolution(tttrs) * max(int(settings.time_scale), 1)

    stream_defs = [
        StreamDef(s.name, list(s.channels), [tuple(r) for r in s.micro_time_ranges])
        for s in settings.streams
    ]
    data, meta = bursts_from_dataframe(
        df, tttrs, stream_defs,
        time_scale=int(settings.time_scale),
        min_photons=int(settings.min_photons),
        return_meta=True,
    )

    acceptor = 1 if len(stream_defs) > 1 else 0
    ana = analyze(
        data,
        state_counts=settings.state_counts,
        criterion=settings.criterion,
        base_time_s=base_time_s,
        acceptor_stream=acceptor,
        donor_stream=0,
        n_restarts=int(settings.n_restarts),
        max_iter=int(settings.max_iter),
        tol=float(settings.tol),
        seed=int(settings.seed),
        engine=settings.engine,
        surrogates=_load_surrogates(settings),
        refine_iters=int(settings.refine_iters),
        patience=settings.patience,
    )

    result = _result_from_analysis(ana, settings)
    bundle = H2mmAnalysisBundle(analysis=ana, data=data, settings=settings)
    bundle.meta = meta
    return result, bundle


def _load_surrogates(settings: H2mmSettings) -> dict[int, object] | None:
    """Load a trained surrogate for the surrogate engines (keyed by its n_states)."""
    if not getattr(settings, "surrogate_path", "") or "surrogate" not in settings.engine:
        return None
    from ..core.surrogate import SurrogateModel

    sm = SurrogateModel.load(settings.surrogate_path)
    return {int(sm.n_states): sm}


def write_result_tables(
    result: H2mmResult,
    bundle: H2mmAnalysisBundle,
    out_dir: pathlib.Path,
) -> None:
    """Write the JSON summary plus the ndX-openable per-photon/per-burst tables.

    Records every written path in ``result.output_paths``.
    """
    from ..core.export import build_tables, write_csv, write_hdf5
    from ..core.h2mm import viterbi

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "h2mm_result.json"
    with open(out_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    result.output_paths["result_json"] = str(out_path)

    meta = getattr(bundle, "meta", None)
    if meta is None or not getattr(bundle.settings, "write_photons", True):
        return
    ana = bundle.analysis
    path, _ = viterbi(ana.best.model, bundle.data)
    tables = build_tables(bundle.data, meta, path, ana.fret, ana.base_time_s)
    try:
        result.output_paths["photons_hdf5"] = write_hdf5(tables.photons, out_dir / "h2mm_photons.h5")
    except Exception:  # pragma: no cover - pytables optional; CSV is the fallback
        result.output_paths["photons_csv"] = write_csv(tables.photons, out_dir / "h2mm_photons.csv")
    result.output_paths["bursts_csv"] = write_csv(tables.bursts, out_dir / "h2mm_bursts.csv")


def _result_from_analysis(ana, settings: H2mmSettings) -> H2mmResult:
    """Convert a :class:`H2mmAnalysis` into the serialisable result model."""
    import numpy as np

    best = ana.best.model
    scan = [
        StateFitSummary(
            n_states=f.n_states,
            loglik=f.loglik,
            bic=f.bic,
            icl=f.icl,
            converged=bool(f.model.converged),
            n_iter=int(f.model.n_iter),
        )
        for f in ana.scan
    ]
    dwell_mean = [
        float(np.mean(v)) * ana.base_time_s if v.size else 0.0
        for _, v in sorted(ana.dwell_times.items())
    ]
    return H2mmResult(
        n_states=ana.best.n_states,
        criterion=settings.criterion,
        scan=scan,
        prior=[float(x) for x in best.prior],
        trans=[[float(x) for x in row] for row in best.trans],
        obs=[[float(x) for x in row] for row in best.obs],
        trans_rates=[[float(x) for x in row] for row in ana.trans_rates],
        fret=[float(x) for x in ana.fret],
        populations=[float(x) for x in ana.populations],
        dwell_mean_s=dwell_mean,
        n_transitions=len(ana.transitions),
        n_bursts=ana.n_bursts,
        n_photons=ana.n_photons,
        base_time_s=ana.base_time_s,
        settings_applied=to_jsonable(settings),
    )


# ---------------------------------------------------------------------------
# RPC handlers
# ---------------------------------------------------------------------------


def compute_handler(
    files: list[str] | None = None,
    analysis_folder: str | None = None,
    pattern: str = "*.bur",
    settings: dict[str, Any] | None = None,
    workflow_context: dict[str, Any] | None = None,
    write_output: bool = True,
) -> dict[str, Any]:
    """Run H2MM analysis from explicit parameters or a workflow handoff."""
    try:
        resolved_folder = _resolve_analysis_folder(analysis_folder, files, workflow_context)
        h2mm_settings = _settings_from_workflow(settings, workflow_context)

        result, bundle = run_analysis(
            h2mm_settings,
            analysis_folder=resolved_folder,
            files=files,
            pattern=pattern,
        )

        if write_output and resolved_folder is not None:
            write_result_tables(result, bundle, pathlib.Path(resolved_folder) / "h2mm")

        payload = to_jsonable(result)
        payload["analysis_folder"] = str(resolved_folder) if resolved_folder else None
        payload["workflow_context"] = workflow_context or {}
        return service_success(payload)
    except Exception as exc:  # pragma: no cover - defensive envelope
        return _service_error(str(exc))


def prepare_workflow_handler(
    workflow_context: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    analysis_folder: str | None = None,
    files: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve H2MM inputs from explicit params plus a workflow context."""
    try:
        resolved_folder = _resolve_analysis_folder(analysis_folder, files, workflow_context)
        h2mm_settings = _settings_from_workflow(settings, workflow_context)
        return service_success(
            {
                "analysis_folder": str(resolved_folder) if resolved_folder else None,
                "settings": to_jsonable(h2mm_settings),
                "workflow_context": workflow_context or {},
            }
        )
    except Exception as exc:  # pragma: no cover
        return _service_error(str(exc))


def contract_handler() -> dict[str, Any]:
    """Return the H2MM workflow contract descriptor."""
    return service_success(contract_descriptor())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_bur_files(
    files: list[str] | None,
    analysis_folder: str | pathlib.Path | None,
    pattern: str,
) -> list[pathlib.Path]:
    """Return the list of ``.bur`` files to analyse."""
    if files:
        return [pathlib.Path(f) for f in files if str(f).endswith(".bur")]
    if analysis_folder:
        folder = pathlib.Path(analysis_folder)
        if folder.is_file() and folder.suffix == ".bur":
            return [folder]
        return sorted(folder.glob("**/" + pattern))
    return []


def _resolve_analysis_folder(
    analysis_folder: str | None,
    files: list[str] | None,
    workflow_context: dict[str, Any] | None,
) -> pathlib.Path | None:
    """Resolve the analysis folder from supported inputs."""
    candidates: list[str] = []
    if analysis_folder:
        candidates.append(str(analysis_folder))
    if workflow_context:
        folder = workflow_context.get("burst_folder") or workflow_context.get("analysis_folder")
        if isinstance(folder, str):
            candidates.append(folder)
    if files:
        candidates.append(str(pathlib.Path(files[0]).parent))
    for candidate in candidates:
        path = pathlib.Path(candidate).expanduser()
        if path.exists():
            return path
    return pathlib.Path(candidates[0]).expanduser() if candidates else None


def _load_tttrs(df, data_dir: pathlib.Path, file_type: str):
    """Load TTTR objects, trying the folder and its parent for each file."""
    import tttrlib

    tttrs: dict[str, Any] = {}
    for ff in df["First File"].unique():
        if ff in tttrs:
            continue
        cand = pathlib.Path(ff)
        for path in (cand, data_dir / ff, data_dir.parent / ff):
            if path.exists():
                ftype = file_type
                if not ftype or ftype.lower() == "auto":
                    ftype = tttrlib.inferTTTRFileType(str(path))
                tttrs[ff] = tttrlib.TTTR(str(path), ftype)
                break
    if not tttrs:
        raise FileNotFoundError("could not locate any TTTR file referenced by the .bur data")
    return tttrs


def _macro_resolution(tttrs) -> float:
    """Return the macro-time resolution (seconds) from the first TTTR header."""
    for tttr in tttrs.values():
        try:
            return float(tttr.header.tag("MeasDesc_GlobalResolution")["value"])
        except Exception:
            try:
                return float(tttr.header.macro_time_resolution)
            except Exception:
                continue
    return 1.0


def _settings_from_workflow(
    settings: dict[str, Any] | None,
    workflow_context: dict[str, Any] | None,
) -> H2mmSettings:
    """Build H2MM settings, deriving streams from a workflow context."""
    if settings:
        return settings_from_dict(H2mmSettings, settings)

    streams: list[StreamSettings] = []
    file_type = "SPC-130"
    if workflow_context:
        channel_settings = workflow_context.get("channel_settings") or {}
        detectors = channel_settings.get("detectors") or {}
        for name, det in detectors.items():
            streams.append(
                StreamSettings(
                    name=str(name),
                    channels=[int(c) for c in det.get("chs", [])],
                    micro_time_ranges=[
                        (int(a), int(b)) for a, b in det.get("micro_time_ranges", [])
                    ],
                )
            )
        tttr_reading = channel_settings.get("tttr_reading") or {}
        if tttr_reading.get("file_type"):
            file_type = str(tttr_reading["file_type"])

    kwargs: dict[str, Any] = {"file_type": file_type}
    if streams:
        kwargs["streams"] = streams
    return H2mmSettings(**kwargs)


def _service_error(message: str) -> dict[str, Any]:
    """Return a ServiceDispatcher-compatible error envelope."""
    try:
        from chisurf.server.services import OPERATION_FAILED, service_error

        return service_error(message, error_code=OPERATION_FAILED)
    except Exception:
        return {"ok": False, "error": message}
