"""Burst selection and summarization functions."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tttrlib

from chisurf.core.fio.fluorescence.burst import generate_burst_dataframe, write_mti_summary
from chisurf.core.fluorescence.burst import burst_filter, count_rate_filter, cusum_filter
import chisurf.core.fluorescence.burst.bocpd as bocpd_mod
import chisurf.core.fluorescence.burst.kalman as kalman_mod
from chisurf.core.fluorescence.burst.utils import create_array_with_ones
from chisurf.core.math.signal import fill_small_gaps_in_array
from chisurf.core.math.signal import find_bursts as signal_find_bursts

from .io import get_unique_folder_path, load_tttr, write_bur, write_hdf5, zip_output_folder
from .models import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisSettings,
    BurstDetectionSettings,
    BurstFilterMode,
    PhotonFilterSettings,
)


def _delta_macro_time_ms(tttr: tttrlib.TTTR) -> np.ndarray:
    """Return delta macro times in milliseconds."""
    macro_times = tttr.macro_times
    d_t = np.diff(macro_times, prepend=macro_times[0])
    return d_t * tttr.header.macro_time_resolution * 1000.0


def apply_photon_filters(
    tttr: tttrlib.TTTR,
    settings: PhotonFilterSettings,
    burst_detection: BurstDetectionSettings | None = None,
) -> np.ndarray:
    """Apply ChiSurf-compatible photon pre-filters to a TTTR object.

    Parameters
    ----------
    tttr : tttrlib.TTTR
        TTTR object.
    settings : PhotonFilterSettings
        Photon filter settings.
    burst_detection : BurstDetectionSettings, optional
        Burst filter settings used when ``settings.used_filter`` is ``"burst"``.

    Returns
    -------
    numpy.ndarray
        Boolean selection mask.
    """
    delta_t = _delta_macro_time_ms(tttr)
    selected = np.ones_like(delta_t, dtype=bool)

    if settings.channels:
        mask = tttrlib.TTTRMask()
        mask.select_channels(tttr, settings.channels, mask=True)
        selected = np.logical_and(selected, mask.get_mask())

    if settings.microtime_ranges:
        mask = tttrlib.TTTRMask()
        mask.select_microtime_ranges(tttr, settings.microtime_ranges)
        mask.flip()
        selected = np.logical_and(selected, mask.get_mask())

    delta_settings = settings.delta_macro_time_filter
    if delta_settings.dT_min_active:
        selected = np.logical_and(selected, delta_t >= delta_settings.dT_min)
    if delta_settings.dT_max_active:
        selected = np.logical_and(selected, delta_t <= delta_settings.dT_max)

    used_filter = BurstFilterMode(settings.used_filter)

    if settings.filter_active:
        if used_filter == BurstFilterMode.COUNT_RATE:
            count_rate_settings = settings.count_rate_filter
            selection = count_rate_filter(
                tttr=tttr,
                n_ph_max=count_rate_settings.n_ph_max,
                time_window=count_rate_settings.time_window,
                invert=settings.invert_filter or count_rate_settings.invert,
                make_mask=True,
            )
            selected = np.logical_and(selected, selection >= 0)
        elif used_filter == BurstFilterMode.BURST:
            detection = burst_detection or BurstDetectionSettings()
            detection_window = min(
                settings.delta_macro_time_filter.dT_max / 1000.0,
                detection.time_window,
            )
            selection = burst_filter(
                tttr=tttr,
                min_ph=detection.min_photons,
                ph_window=detection.photon_window,
                time_window=detection_window,
            )
            selected = np.logical_and(selected, selection)
        elif used_filter == BurstFilterMode.BOCPD:
            channel_list = settings.channels
            if len(channel_list) < 1:
                channel_list = list(tttr.get_used_routing_channels())
            macro_times = tttr.macro_times
            time_unit = tttr.header.macro_time_resolution
            timestamps = macro_times * time_unit
            channels = tttr.routing_channels
            timestamps_list = []
            for channel in channel_list:
                channel_timestamps = timestamps[channels == channel]
                timestamps_list.append(channel_timestamps)
                if len(channel_timestamps) == 0:
                    return np.zeros_like(selected, dtype=np.uint8)
            bocpd_settings = settings.bocpd_filter
            min_counts = burst_detection.min_photons if burst_detection else 60
            bursts, _, _, _, _ = bocpd_mod.bocpd_burst_detection_multi(
                timestamps_list,
                dt=bocpd_settings.dt,
                prior_count=bocpd_settings.prior_count,
                prior_duration=bocpd_settings.prior_duration,
                changepoint_prob=bocpd_settings.changepoint_prob,
                max_run=256,
                min_counts=min_counts
            )
            start_stop = bocpd_mod.convert_bursts_to_start_stop(bursts, tttr)
            if len(start_stop) > 0:
                selection = create_array_with_ones(start_stop, len(tttr))
                selected = np.logical_and(selected, selection)
            else:
                selected = np.zeros_like(selected)
        elif used_filter == BurstFilterMode.KALMAN:
            channel_list = settings.channels
            if len(channel_list) < 1:
                channel_list = list(tttr.get_used_routing_channels())
            macro_times = tttr.macro_times
            time_unit = tttr.header.macro_time_resolution
            timestamps = macro_times * time_unit
            channels = tttr.routing_channels
            timestamps_list = []
            for channel in channel_list:
                channel_timestamps = timestamps[channels == channel]
                timestamps_list.append(channel_timestamps)
                if len(channel_timestamps) == 0:
                    return np.zeros_like(selected, dtype=np.uint8)
            kalman_settings = settings.kalman_filter
            min_counts = burst_detection.min_photons if burst_detection else 60
            bursts, _, _, _, _ = kalman_mod.kalman_burst_detection_multi(
                timestamps_list,
                dt=kalman_settings.dt,
                q=kalman_settings.q,
                r_scale=kalman_settings.r_scale,
                z_thresh=kalman_settings.z_thresh,
                min_len=kalman_settings.min_len,
                merge_gap=kalman_settings.merge_gap,
                min_counts=min_counts
            )
            start_stop = kalman_mod.convert_bursts_to_start_stop(bursts, tttr)
            if len(start_stop) > 0:
                selection = create_array_with_ones(start_stop, len(tttr))
                selected = np.logical_and(selected, selection)
            else:
                selected = np.zeros_like(selected)
        elif used_filter == BurstFilterMode.CUSUM:
            cusum_settings = settings.cusum_filter
            selection = cusum_filter(
                tttr=tttr,
                min_ph=cusum_settings.min_photons,
                background_rate=cusum_settings.background_rate,
                sb_ratio=cusum_settings.sb_ratio,
                alpha=cusum_settings.alpha,
                beta=cusum_settings.beta,
            )
            selected = np.logical_and(selected, selection)
        else:
            raise ValueError(f"Unsupported filter mode: {settings.used_filter}")

    if settings.invert_filter and used_filter != BurstFilterMode.COUNT_RATE:
        selected = ~selected

    if settings.use_gap_fill and settings.max_gap > 0:
        selected = fill_small_gaps_in_array(selected, max_gap=settings.max_gap)

    return selected.astype(dtype=np.uint8)


def find_bursts(selected_mask: Iterable[int], max_gap: int = 4) -> np.ndarray:
    """Find burst start/stop pairs from a selected-photon mask.

    Parameters
    ----------
    selected_mask : iterable of int
        Boolean or integer selection mask.
    max_gap : int, default=4
        Maximum number of unselected photons to bridge inside a burst.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_bursts, 2)`` containing start/stop photon indices.
    """
    mask = np.asarray(selected_mask, dtype=np.uint8)
    if len(mask) <= max_gap:
        return np.array([], dtype=np.uint64)
    return signal_find_bursts(mask, max_gap=max_gap)


def summarize_bursts(
    start_stop: np.ndarray,
    filename: str | Path,
    tttr: tttrlib.TTTR,
    windows: dict[str, tuple[int, int]] | None = None,
    detectors: dict[str, dict[str, Any]] | None = None,
    macro_time_resolution: float | None = None,
) -> pd.DataFrame:
    """Generate a ChiSurf-compatible burst summary DataFrame.

    Parameters
    ----------
    start_stop : numpy.ndarray
        Burst start/stop indices.
    filename : str or Path
        Source TTTR filename.
    tttr : tttrlib.TTTR
        TTTR object.
    windows : dict, optional
        PIE windows.
    detectors : dict, optional
        Detector definitions.
    macro_time_resolution : float, optional
        Macro-time resolution override for output durations.

    Returns
    -------
    pandas.DataFrame
        Burst summary table.
    """
    return generate_burst_dataframe(
        start_stop=start_stop,
        filename=filename,
        tttr=tttr,
        windows=windows or {},
        detectors=detectors or {},
        include_interleaved_zeros=True,
        macro_time_resolution=macro_time_resolution,
    )


def legacy_output_folder_name(settings: AnalysisSettings) -> str:
    """Return the legacy output folder name for analysis settings."""
    channels = settings.photon_filter.channels
    channel_text = ",".join(str(channel) for channel in channels) if channels else "All"
    mode = BurstFilterMode(settings.photon_filter.used_filter)
    if mode == BurstFilterMode.COUNT_RATE:
        prefix = "countrate"
    elif mode == BurstFilterMode.BOCPD:
        prefix = "bocpd"
    elif mode == BurstFilterMode.KALMAN:
        prefix = "kalman"
    elif mode == BurstFilterMode.CUSUM:
        prefix = "cusum"
    else:
        prefix = "burstwise"
    return (
        f"{prefix}_{channel_text} "
        f"{settings.photon_filter.delta_macro_time_filter.dT_max:.4f}"
        f"#{settings.burst_detection.min_photons}"
    )


def _prepare_legacy_output_folder(request: AnalysisRequest) -> Path | None:
    """Resolve and create the legacy output folder for a request."""
    if not request.files or not request.settings.output_formats:
        return None
    first_file = Path(request.files[0]).resolve()
    folder_name = request.legacy_output_folder_name or legacy_output_folder_name(request.settings)
    output_folder = get_unique_folder_path(first_file.parent / folder_name)
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def _write_legacy_output_info(request: AnalysisRequest, output_folder: Path) -> None:
    """Write legacy ``Info`` files for a request."""
    info_dir = output_folder / "Info"
    info_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(request.settings)
    payload.update(request.legacy_parameters)
    payload.update(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "selected_setup": request.selected_setup,
            "channels": request.settings.photon_filter.channels,
            "microtime_ranges": request.settings.photon_filter.microtime_ranges,
            "files": [Path(file_name).name for file_name in request.files],
        }
    )
    with (info_dir / "photon_selection_parameters.json").open("w") as fp:
        json.dump(payload, fp, indent=4, default=str)
    with (info_dir / "datetime.txt").open("w") as fp:
        fp.write(f"Date: {time.strftime('%Y-%m-%d')}\nTime: {time.strftime('%H:%M:%S')}\n")


def analyze_file(
    path: str | Path,
    *,
    settings: AnalysisSettings | None = None,
    filetype: str | None = None,
    windows: dict[str, tuple[int, int]] | None = None,
    detectors: dict[str, dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    mti_output_dir: str | Path | None = None,
    macro_time_resolution: float | None = None,
) -> AnalysisResult:
    """Run burst selection for one TTTR file.

    Parameters
    ----------
    path : str or Path
        TTTR file path.
    settings : AnalysisSettings, optional
        Analysis settings.
    filetype : str, optional
        TTTR file type.
    windows : dict, optional
        PIE windows.
    detectors : dict, optional
        Detector definitions.
    output_dir : str or Path, optional
        Directory for ``.bur`` output.
    mti_output_dir : str or Path, optional
        Legacy analysis directory where the ``Info/*.mti`` sidecar is written.
    macro_time_resolution : float, optional
        Macro-time resolution override for output durations.

    Returns
    -------
    AnalysisResult
        Analysis result.
    """
    normalized_path = str(Path(path).resolve())
    analysis_settings = settings or AnalysisSettings()
    tttr = load_tttr(path, filetype=filetype)
    if len(tttr) == 0:
        return AnalysisResult(
            files=[str(path)],
            dataframes={str(path): []},
            metadata={"n_photons": 0, "n_selected": 0, "n_bursts": 0},
            output_paths={},
            output_paths_by_file={normalized_path: {}},
        )
    selected = apply_photon_filters(
        tttr,
        analysis_settings.photon_filter,
        burst_detection=analysis_settings.burst_detection,
    )
    start_stop = find_bursts(selected)
    output_resolution = (
        float(tttr.header.macro_time_resolution)
        if macro_time_resolution is None
        else float(macro_time_resolution)
    )
    df = summarize_bursts(
        start_stop,
        path,
        tttr,
        windows=windows or {},
        detectors=detectors or {},
        macro_time_resolution=output_resolution,
    )

    output_paths: dict[str, str] = {}
    if "bur" in analysis_settings.output_formats and output_dir is not None:
        bur_path = Path(output_dir) / f"{Path(path).stem}.bur"
        write_bur(df, bur_path)
        output_paths["bur"] = str(bur_path)
        if mti_output_dir is not None:
            max_macro_time = float(tttr.macro_times[-1]) * output_resolution
            write_mti_summary(Path(path), Path(mti_output_dir), max_macro_time, append=True)
            output_paths["mti_dir"] = str(Path(mti_output_dir) / "Info")

    return AnalysisResult(
        files=[str(path)],
        dataframes={str(path): df.to_dict(orient="records")},
        metadata={
            "n_photons": int(len(tttr)),
            "n_selected": int(np.count_nonzero(selected)),
            "n_bursts": int(len(start_stop)),
            "macro_time_resolution": output_resolution,
        },
        output_paths=output_paths,
        output_paths_by_file={normalized_path: dict(output_paths)},
    )


def analyze_request(request: AnalysisRequest) -> AnalysisResult:
    """Run burst selection for a complete analysis request.

    Parameters
    ----------
    request : AnalysisRequest
        Analysis request.

    Returns
    -------
    AnalysisResult
        Combined analysis result.
    """
    frames: dict[str, list[dict[str, Any]]] = {}
    output_paths: dict[str, str] = {}
    output_paths_by_file: dict[str, dict[str, str]] = {}
    metadata: dict[str, Any] = {
        "n_files": len(request.files),
        "n_bursts": 0,
        "n_selected": 0,
        "n_photons": 0,
    }
    legacy_output_folder = _prepare_legacy_output_folder(request) if request.legacy_output else None
    bur_output_dir = (
        legacy_output_folder / "bi4_bur"
        if legacy_output_folder is not None and "bur" in request.settings.output_formats
        else request.output_dir
    )
    hdf5_frames: list[pd.DataFrame] = []
    batch_macro_time_resolution: float | None = None

    for path in request.files:
        result = analyze_file(
            path,
            settings=request.settings,
            filetype=request.filetype,
            windows=request.windows,
            detectors=request.detectors,
            output_dir=bur_output_dir,
            mti_output_dir=legacy_output_folder if legacy_output_folder is not None else None,
            macro_time_resolution=batch_macro_time_resolution,
        )
        if batch_macro_time_resolution is None and "macro_time_resolution" in result.metadata:
            batch_macro_time_resolution = float(result.metadata["macro_time_resolution"])
        frames.update(result.dataframes)
        output_paths.update(result.output_paths)
        output_paths_by_file.update(result.output_paths_by_file)
        if legacy_output_folder is not None and "hdf5" in request.settings.output_formats:
            frame = pd.DataFrame(result.dataframes.get(str(path), []))
            frame["Source File"] = str(path)
            hdf5_frames.append(frame)
        metadata["n_bursts"] += int(result.metadata.get("n_bursts", 0))
        metadata["n_selected"] += int(result.metadata.get("n_selected", 0))
        metadata["n_photons"] += int(result.metadata.get("n_photons", 0))

    if legacy_output_folder is not None:
        _write_legacy_output_info(request, legacy_output_folder)
        output_paths["output_folder"] = str(legacy_output_folder)
        metadata["output_folder"] = str(legacy_output_folder)
        if "hdf5" in request.settings.output_formats and hdf5_frames:
            hdf5_dir = legacy_output_folder / "hdf5"
            hdf5_path = hdf5_dir / f"burst_data_{time.strftime('%Y%m%d-%H%M%S')}.h5"
            write_hdf5(hdf5_frames, hdf5_path)
            output_paths["hdf5"] = str(hdf5_path)
        if request.settings.zip_output:
            zip_path = zip_output_folder(legacy_output_folder)
            output_paths["zip"] = str(zip_path)
            if request.settings.remove_folder:
                shutil.rmtree(legacy_output_folder)
                output_paths["output_folder"] = str(zip_path)
                metadata["output_folder"] = str(zip_path)

    return AnalysisResult(
        files=request.files,
        dataframes=frames,
        output_paths=output_paths,
        output_paths_by_file=output_paths_by_file,
        metadata=metadata,
    )
