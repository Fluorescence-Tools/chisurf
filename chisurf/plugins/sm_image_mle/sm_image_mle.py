#!/usr/bin/env python3
"""
CLI Tool for Processing PTU Files and Extracting Molecule Data

This script provides a command-line interface (CLI) for:
  - Loading an Instrument Response Function (IRF) from a PTU file.
  - Segmenting image-based molecules in PTU files using watershed.
  - Computing Jordi vectors (photon arrival histograms) for each molecule.
  - Fitting each Jordi vector with a biexponential model (Fit23).
  - Saving molecule-specific outputs (indices, jordi vectors, fit plots, segmented overlays).
  - Aggregating all molecule properties into a combined TSV file (always respects --output-file).
"""

import sys
from pathlib import Path

import click
from click_didyoumean import DYMGroup

import numpy as np
import pandas as pd
import tttrlib
from skimage import filters, measure, util, io as skio
from skimage.segmentation import clear_border, watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def interpolate_shift(arr: np.ndarray, shift: float) -> np.ndarray:
    """
    Shift an array by an integer and fractional offset via interpolation.

    Parameters
    ----------
    arr : np.ndarray
        1D array of values (e.g., IRF histogram).
    shift : float
        Amount to shift: integer portion rolls the array, fractional portion
        applies linear interpolation (with zeros padding).

    Returns
    -------
    np.ndarray
        New array of the same shape, shifted by `shift`. If `shift` is zero,
        returns a copy of the original array.
    """
    result = arr.astype(np.float64).copy()
    if shift == 0:
        return result

    int_shift = int(np.trunc(shift))
    if int_shift != 0:
        result = np.roll(result, int_shift)
        if int_shift > 0:
            result[:int_shift] = 0.0
        else:
            result[int_shift:] = 0.0

    frac_shift = shift - int_shift
    if frac_shift != 0:
        x = np.arange(result.size)
        result = np.interp(x - frac_shift, x, result, left=0.0, right=0.0)

    return result


def compute_irf_component(
        tttr_obj: tttrlib.TTTR,
        micro_time_range: tuple[int, int],
        detector_chs_list: list[int],
        micro_time_binning: int,
        minlength: int = -1
) -> np.ndarray:
    """
    Compute a micro-time histogram for a subset of detector channels from a TTTR object.
    """
    start_bin, stop_bin = micro_time_range
    raw_start = start_bin * micro_time_binning
    raw_stop = stop_bin * micro_time_binning

    mt_arr = tttr_obj.micro_times
    ch_arr = tttr_obj.routing_channels

    mask = (
        (mt_arr >= raw_start) &
        (mt_arr <= raw_stop) &
        np.isin(ch_arr, detector_chs_list)
    )
    indices = np.where(mask)[0]
    sub = tttr_obj[indices]

    hist, _ = sub.get_microtime_histogram(micro_time_binning, minlength=minlength)
    return hist[start_bin:stop_bin]


def load_irf(
        irf_file: str,
        detector_chs: list[int],
        micro_time_range: tuple[int, int],
        micro_time_binning: int,
        shift_sp: float,
        shift_ss: float,
        irf_threshold_fraction: float
) -> tuple[np.ndarray, np.ndarray, tttrlib.TTTR]:
    """
    Load and normalize the Instrument Response Function (IRF) from a PTU file.
    """
    irf_tttr = tttrlib.TTTR(irf_file)

    if len(detector_chs) >= 2:
        sp_chs = detector_chs[0::2]
        ss_chs = detector_chs[1::2]
    else:
        sp_chs = ss_chs = detector_chs

    sp_irf_bins = compute_irf_component(irf_tttr, micro_time_range, sp_chs, micro_time_binning)
    ss_irf_bins = compute_irf_component(irf_tttr, micro_time_range, ss_chs, micro_time_binning)

    sp_irf_shifted = interpolate_shift(sp_irf_bins, shift_sp)
    ss_irf_shifted = interpolate_shift(ss_irf_bins, shift_ss)

    sp_irf_norm = sp_irf_shifted / sp_irf_shifted.sum() if sp_irf_shifted.sum() > 0 else sp_irf_shifted.copy()
    ss_irf_norm = ss_irf_shifted / ss_irf_shifted.sum() if ss_irf_shifted.sum() > 0 else ss_irf_shifted.copy()

    irf_stacked = np.hstack([sp_irf_norm, ss_irf_norm])
    raw_irf = irf_stacked.copy()
    irf_full = irf_stacked.copy()

    if irf_full.max() > 0:
        mask_low = irf_full < irf_threshold_fraction * irf_full.max()
        irf_full[mask_low] = 0
        if irf_full.sum() > 0:
            irf_full = irf_full / irf_full.sum()

    return irf_full, raw_irf, irf_tttr


def filter_tttr(
        tttr_obj: tttrlib.TTTR,
        micro_time_range: tuple[int, int],
        detector_chs_list: list[int],
        micro_time_binning: int
) -> tttrlib.TTTR:
    """
    Filter a TTTR object by micro-time range and routing channels.
    """
    start_bin, stop_bin = micro_time_range
    raw_start = start_bin * micro_time_binning
    raw_stop = stop_bin * micro_time_binning

    mt_arr = tttr_obj.micro_times
    ch_arr = tttr_obj.routing_channels
    mask = (
        (mt_arr >= raw_start) &
        (mt_arr <= raw_stop) &
        np.isin(ch_arr, detector_chs_list)
    )
    indices = np.where(mask)[0]
    return tttr_obj[indices]


def compute_g_factor(
        tttr_obj: tttrlib.TTTR,
        g_factor_chs: list[int],
        micro_time_range: tuple[int, int],
        micro_time_binning: int
) -> float:
    """
    Compute the G-factor (parallel/perpendicular detection efficiency) from the IRF TTTR.
    """
    start_bin, stop_bin = micro_time_range

    def get_decay(ch_list: list[int]) -> np.ndarray:
        reduced = filter_tttr(tttr_obj, micro_time_range, ch_list, micro_time_binning)
        hist = reduced.get_microtime_histogram(micro_time_binning)[0]
        return hist[start_bin:stop_bin]

    if len(g_factor_chs) >= 2:
        sp_c = [g_factor_chs[0]]
        ss_c = [g_factor_chs[1]]
    else:
        sp_c = ss_c = g_factor_chs

    p_decay = get_decay(sp_c)
    s_decay = get_decay(ss_c)
    n = len(p_decay)
    tail_start = int(np.floor(0.8 * n))

    p_tail = p_decay[tail_start:]
    s_tail = s_decay[tail_start:]
    sum_p = np.sum(p_tail)
    sum_s = np.sum(s_tail)

    return sum_p / sum_s if sum_s > 0 else 1.0


def make_jordi(
        tttr_obj: tttrlib.TTTR,
        detector_chs: list[int],
        micro_time_range: tuple[int, int],
        micro_time_binning: int,
        normalize_counts: int = 0,
        threshold: float = -1,
        minlength: int = -1,
        shift: float = 0
) -> np.ndarray:
    """
    Construct a Jordi vector (concatenated parallel & perpendicular photon histograms).
    """
    if len(detector_chs) >= 2:
        ch_group0 = detector_chs[0::2]
        ch_group1 = detector_chs[1::2]
    else:
        ch_group0 = ch_group1 = detector_chs

    tp = filter_tttr(tttr_obj, micro_time_range, ch_group0, micro_time_binning)
    ts = filter_tttr(tttr_obj, micro_time_range, ch_group1, micro_time_binning)
    cp_hist = tp.get_microtime_histogram(micro_time_binning, minlength=minlength)[0]
    cs_hist = ts.get_microtime_histogram(micro_time_binning, minlength=minlength)[0]

    start_bin, stop_bin = micro_time_range
    cp = cp_hist[start_bin:stop_bin]
    cs = cs_hist[start_bin:stop_bin]

    if threshold > 0:
        if cp.max() > 0:
            cp[cp < threshold * cp.max()] = 0
        if cs.max() > 0:
            cs[cs < threshold * cs.max()] = 0

    if normalize_counts == 1:
        ct = (cp.sum() + cs.sum()) / 2.0
        if ct > 0:
            cp = cp / ct
            cs = cs / ct
    elif normalize_counts == 2:
        if cp.sum() > 0:
            cp = cp / cp.sum()
        if cs.sum() > 0:
            cs = cs / cs.sum()
    elif normalize_counts == 3:
        acq_time = (tttr_obj.macro_times[-1] - tttr_obj.macro_times[0]) \
                   * tttr_obj.header.macro_time_resolution
        if acq_time > 0:
            cp = cp / acq_time
            cs = cs / acq_time

    if shift != 0:
        cs = np.roll(cs, int(shift))

    return np.hstack([cp, cs])


def process_ptu_file(
        ptu_path: Path,
        irf_full: np.ndarray,
        raw_irf: np.ndarray,
        irf_tttr: tttrlib.TTTR,
        detector_chs: list[int],
        micro_time_range: tuple[int, int],
        micro_time_binning: int,
        normalize_counts: int,
        threshold: float,
        minlength: int,
        l1: float,
        l2: float,
        twoi_star_flag: bool,
        bifl_scatter_flag: bool,
        fit_initial_values: np.ndarray,
        fit_fixed_flags: np.ndarray,
        seg_sigma: float,
        seg_threshold: float,
        peak_footprint_size: int
) -> pd.DataFrame:
    """
    Process a single PTU file: segment molecules, compute Jordi vectors, fit decay curves,
    and save outputs. Returns a DataFrame of molecule properties.
    """
    fn = str(ptu_path)
    base_folder = ptu_path.parent
    base_name = ptu_path.stem

    output_folder = base_folder / f"{base_name}_analysis"
    image_folder = output_folder / 'molecule_images'
    indices_folder = output_folder / 'tttr_indices'
    jordis_folder = output_folder / 'jordis'
    jordi_images_folder = output_folder / 'jordi_images'
    for folder in [image_folder, indices_folder, jordis_folder, jordi_images_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    d = tttrlib.TTTR(fn)
    import json
    hdr_json = getattr(d.header, 'json', '{}')
    hdr_dict = json.loads(hdr_json)
    has_image = False
    for t in hdr_dict.get('tags', []):
        if t.get('name') == 'ImgHdr_Dimensions' and t.get('value', 0) > 2:
            has_image = True
            break
    if not has_image:
        return pd.DataFrame()

    combined_clsm = tttrlib.CLSMImage(d)
    combined_clsm.fill(d, channels=detector_chs)
    total_intensity = combined_clsm.intensity.sum(axis=0)
    total_image_path = output_folder / f"{base_name}_total_intensity.tif"
    skio.imsave(str(total_image_path), total_intensity.astype(np.uint16), check_contrast=False)

    for ch_idx in detector_chs:
        ch_clsm = tttrlib.CLSMImage(d)
        ch_clsm.fill(d, channels=[ch_idx])
        ch_intensity = ch_clsm.intensity.sum(axis=0)
        ch_image_path = output_folder / f"{base_name}_intensity_ch{ch_idx}.tif"
        skio.imsave(str(ch_image_path), ch_intensity.astype(np.uint16), check_contrast=False)

    dt_micro = d.header.micro_time_resolution
    dt_micro_ns = dt_micro * 1e9
    DT_EFFECTIVE_ns = dt_micro_ns * micro_time_binning
    num_bins = micro_time_range[1] - micro_time_range[0]
    excitation_period = DT_EFFECTIVE_ns * num_bins

    dt_info_path = output_folder / 'microtime_resolution_ns.txt'
    with open(dt_info_path, 'w') as f:
        f.write(f"{DT_EFFECTIVE_ns:.6f}\n")

    irf_path = output_folder / 'irf_full.txt'
    np.savetxt(irf_path, irf_full, fmt='%.6f')

    g_factor = compute_g_factor(irf_tttr, detector_chs, micro_time_range, micro_time_binning)

    smoothed = filters.gaussian(total_intensity, sigma=seg_sigma)
    if seg_threshold > 0:
        thresh_val = seg_threshold
    else:
        thresh_val = filters.threshold_otsu(smoothed)
    binary = smoothed > thresh_val
    clean = clear_border(binary)
    distance = ndi.distance_transform_edt(clean)
    footprint = np.ones((peak_footprint_size, peak_footprint_size), dtype=bool)
    coords = peak_local_max(distance, footprint=footprint, labels=clean)
    mask0 = np.zeros(distance.shape, dtype=bool)
    mask0[tuple(coords.T)] = True
    markers, _ = ndi.label(mask0)
    labels = watershed(-distance, markers, mask=clean)

    smoothed_norm = smoothed / smoothed.max() if smoothed.max() > 0 else smoothed.copy()
    smoothed_uint8 = util.img_as_ubyte(smoothed_norm)

    records = []
    for prop in measure.regionprops(labels):
        lab = prop.label
        coords_arr = prop.coords
        tttr_list = []
        for (r, c_) in coords_arr:
            pix = combined_clsm[0][int(r)][int(c_)].tttr_indices
            indices = list(pix)
            if indices:
                tttr_list.extend(indices)
        tttr_array = np.array(tttr_list, dtype=np.int64)
        sub_tttr = d[tttr_array]

        indices_filename = f"indices_{lab:03d}.txt"
        indices_path = indices_folder / indices_filename
        with open(indices_path, 'w') as f:
            for idx in tttr_array:
                f.write(f"{idx}\n")

        jordi_vector = make_jordi(
            sub_tttr,
            detector_chs,
            micro_time_range,
            micro_time_binning,
            normalize_counts,
            threshold,
            minlength,
            shift=0
        )
        jordi_filename = f"molecule_{lab:03d}.jordi"
        jordi_path = jordis_folder / jordi_filename
        np.savetxt(jordi_path, jordi_vector, delimiter='\t', fmt='%.6f')

        fit = tttrlib.Fit23(
            dt=DT_EFFECTIVE_ns,
            irf=irf_full,
            background=raw_irf,
            period=excitation_period,
            g_factor=g_factor,
            l1=l1,
            l2=l2,
            p2s_twoIstar_flag=twoi_star_flag,
            soft_bifl_scatter_flag=bifl_scatter_flag
        )
        result = fit(data=jordi_vector, initial_values=fit_initial_values, fixed=fit_fixed_flags)
        tau_fit = result['x'][0]
        gamma_fit = result['x'][1]
        rho_fit = result['x'][3]
        model_curve = fit.model

        plt.figure()
        plt.semilogy(jordi_vector, label='Data')
        plt.semilogy(model_curve, label='Fit', linestyle='--')
        irf_scaled = irf_full * np.max(jordi_vector)
        plt.semilogy(irf_scaled, label='IRF', linestyle=':')
        plt.xlabel('Bin Index')
        plt.ylabel('Photon Count')
        plt.title(f"{base_name} Molecule {lab:03d}: \u03C4={tau_fit:.2f}ns, \u03B3={gamma_fit:.2f}, \u03C1={rho_fit:.2f}")
        plt.legend()
        plot_name = f"molecule_{lab:03d}_jordi.png"
        plot_path = jordi_images_folder / plot_name
        plt.savefig(plot_path)
        plt.close()

        centroid_row, centroid_col = prop.centroid
        minr, minc, maxr, maxc = prop.bbox
        area = prop.area
        perimeter = prop.perimeter
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
        eccentricity = prop.eccentricity
        solidity = prop.solidity

        H, W = total_intensity.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[..., 0] = smoothed_uint8
        rgb[..., 1] = smoothed_uint8
        rgb[..., 2] = smoothed_uint8
        mask_region = (labels == lab)
        rgb[mask_region, 0] = smoothed_uint8[mask_region]
        rgb[mask_region, 1] = 0
        rgb[mask_region, 2] = 0
        png_name = f"molecule_{lab:03d}.png"
        png_path = image_folder / png_name
        skio.imsave(str(png_path), rgb, check_contrast=False)

        # Count photons per channel group
        ch_group0 = detector_chs[0::2] if len(detector_chs) >= 2 else detector_chs
        ch_group1 = detector_chs[1::2] if len(detector_chs) >= 2 else detector_chs

        ch_arr = sub_tttr.routing_channels
        n_photons_total = ch_arr.size
        n_photons_sp = np.sum(np.isin(ch_arr, ch_group0))
        n_photons_ss = np.sum(np.isin(ch_arr, ch_group1))

        records.append({
            'source_ptu': fn,
            'label': lab,
            'centroid_row': centroid_row,
            'centroid_col': centroid_col,
            'min_row': minr,
            'min_col': minc,
            'max_row': maxr,
            'max_col': maxc,
            'pixel_count': coords_arr.shape[0],
            'area': area,
            'n_photons_total': n_photons_total,
            'n_photons_sp': n_photons_sp,
            'n_photons_ss': n_photons_ss,
            'perimeter': perimeter,
            'circularity': circularity,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'tau_fit': tau_fit,
            'gamma_fit': gamma_fit,
            'rho_fit': rho_fit,
            'tttr_indices_file': str(Path('tttr_indices') / indices_filename),
            'jordi_file': str(Path('jordis') / jordi_filename),
            'jordi_plot_file': str(Path('jordi_images') / plot_name),
            'microtime_resolution_file': str(Path('microtime_resolution_ns.txt')),
            'irf_file': str(Path('irf_full.txt')),
            'image_filename': str(Path('molecule_images') / png_name)
        })

    df = pd.DataFrame.from_records(records)
    pkl_path = output_folder / 'molecule_data.pkl'
    csv_path = output_folder / 'molecule_data.tsv'
    df.to_pickle(str(pkl_path))
    df.to_csv(str(csv_path), sep='\t', index=False)

    return df


# --------------------------------------------------
# CLI Definition using Click and click-didyoumean
# --------------------------------------------------

@click.group(cls=DYMGroup)
def cli():
    """
    Main entry point for the PTU processing CLI.
    """
    pass


@cli.command()
@click.option(
    '--ptu-pattern', '-p',
    default='*.ptu',
    show_default=True,
    type=str,
    help='Wildcard pattern to match PTU files under --output-dir (e.g., "*.ptu").'
)
@click.option(
    '--irf-file', '-i',
    'irf_file',
    default='../H2O_2.ptu',
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to the PTU file used for Instrument Response Function.'
)
@click.option(
    '--detector-chs', '-d',
    'detector_chs',
    default=[2, 0],
    show_default=True,
    type=int,
    multiple=True,
    help='Detector channels: even=parallel, odd=perpendicular.'
)
@click.option(
    '--micro-time-range', '-r',
    'micro_time_range',
    default=(0, 256),
    show_default=True,
    type=int,
    nargs=2,
    help='Micro-time bin range (start stop) after binning (e.g., -r 0 256).'
)
@click.option(
    '--micro-time-binning', '-b',
    'micro_time_binning',
    default=32,
    show_default=True,
    type=int,
    help='Factor by which raw micro-time data is binned (e.g., 32).'
)
@click.option(
    '--normalize-counts', '-n',
    'normalize_counts',
    default=0,
    show_default=True,
    type=click.IntRange(0, 3),
    help=(
        'Normalization mode for Jordi vectors:\n'
        '0 = no normalization, 1 = normalize by average rate,\n'
        '2 = normalize each channel to unit area, 3 = normalize by acq. time'
    )
)
@click.option(
    '--threshold', '-t',
    'threshold',
    default=-1.0,
    show_default=True,
    type=float,
    help='Threshold fraction to zero out low-intensity bins.'
)
@click.option(
    '--minlength',
    'minlength',
    default=-1,
    show_default=True,
    type=int,
    help='Minimum histogram length for Jordi vectors; -1 for auto.'
)
@click.option(
    '--shift-sp',
    'shift_sp',
    default=0.0,
    show_default=True,
    type=float,
    help='Fractional shift to apply to the parallel IRF component.'
)
@click.option(
    '--shift-ss',
    'shift_ss',
    default=0.0,
    show_default=True,
    type=float,
    help='Fractional shift to apply to the perpendicular IRF component.'
)
@click.option(
    '--irf-threshold-fraction',
    'irf_threshold_fraction',
    default=0.08,
    show_default=True,
    type=float,
    help='Zero out any IRF bin < fraction of IRF_max.'
)
@click.option(
    '--fit-initial-values',
    'fit_initial_values',
    default=(1.0, 0.0, 0.38, 1.0),
    show_default=True,
    type=float,
    nargs=4,
    help='Initial guess for Fit23 decay parameters: (tau0, gamma0, rho0, scale0).'
)
@click.option(
    '--fit-fixed-flags',
    'fit_fixed_flags',
    default=(0, 0, 1, 0),
    show_default=True,
    type=int,
    nargs=4,
    help='Flags to fix Fit23 parameters: (tau, gamma, rho, scale); 0=free, 1=fixed.'
)
@click.option(
    '--l1',
    'l1',
    default=0.04,
    show_default=True,
    type=float,
    help='Fit23 model parameter l1 (scattering/loss).'
)
@click.option(
    '--l2',
    'l2',
    default=0.04,
    show_default=True,
    type=float,
    help='Fit23 model parameter l2 (scattering/loss).'
)
@click.option(
    '--twoi-star/--no-twoi-star',
    'twoi_star',
    default=True,
    show_default=True,
    help='Enable or disable the twoI* flag in Fit23.'
)
@click.option(
    '--bifl-scatter/--no-bifl-scatter',
    'bifl_scatter',
    default=False,
    show_default=True,
    help='Enable or disable the bifurcation scatter flag in Fit23.'
)
@click.option(
    '--output-dir', '-o',
    'output_dir',
    default='.',
    show_default=True,
    type=click.Path(file_okay=False, writable=True),
    help='Directory in which to search for PTU files and save outputs.'
)
@click.option(
    '--output-file', '-f',
    'output_file',
    default='combined_molecule_data.tsv',
    show_default=True,
    type=click.Path(dir_okay=False, writable=True),
    help='Path for the combined TSV of all molecules.'
)
@click.option(
    '--seg-sigma',
    'seg_sigma',
    default=1.0,
    show_default=True,
    type=float,
    help='Gaussian smoothing sigma for image segmentation.'
)
@click.option(
    '--seg-threshold',
    'seg_threshold',
    default=-1.0,
    show_default=True,
    type=float,
    help='Fixed intensity threshold; if <0, use Otsu.'
)
@click.option(
    '--peak-footprint-size',
    'peak_footprint_size',
    default=6,
    show_default=True,
    type=int,
    help='Side length of square footprint for peak_local_max.'
)
def run(
        ptu_pattern: str,
        irf_file: str,
        detector_chs: tuple[int, ...],
        micro_time_range: tuple[int, int],
        micro_time_binning: int,
        normalize_counts: int,
        threshold: float,
        minlength: int,
        shift_sp: float,
        shift_ss: float,
        irf_threshold_fraction: float,
        fit_initial_values: tuple[float, float, float, float],
        fit_fixed_flags: tuple[int, int, int, int],
        l1: float,
        l2: float,
        twoi_star: bool,
        bifl_scatter: bool,
        output_dir: str,
        output_file: str,
        seg_sigma: float,
        seg_threshold: float,
        peak_footprint_size: int
) -> None:
    """
    Main command to process PTU files and extract molecule-level data.
    """
    detector_chs_list = list(detector_chs)
    micro_time_range_tuple = (micro_time_range[0], micro_time_range[1])
    fit_initial = np.array(fit_initial_values, dtype=float)
    fit_fixed = np.array(fit_fixed_flags, dtype=int)

    base_path = Path(output_dir).resolve()
    if not base_path.exists():
        click.echo(f"Error: Output directory {base_path} does not exist.", err=True)
        sys.exit(1)
    click.echo(f"Using output directory: {base_path}")

    click.echo(f"Loading IRF from: {irf_file}")
    irf_full, raw_irf, irf_tttr = load_irf(
        irf_file,
        detector_chs_list,
        micro_time_range_tuple,
        micro_time_binning,
        shift_sp,
        shift_ss,
        irf_threshold_fraction
    )
    click.echo("IRF loaded and normalized.")

    all_dfs = []
    for ptu_path in base_path.glob(ptu_pattern):
        click.echo(f"Processing PTU file: {ptu_path.name}")
        df_ptu = process_ptu_file(
            ptu_path,
            irf_full,
            raw_irf,
            irf_tttr,
            detector_chs_list,
            micro_time_range_tuple,
            micro_time_binning,
            normalize_counts,
            threshold,
            minlength,
            l1,
            l2,
            twoi_star,
            bifl_scatter,
            fit_initial,
            fit_fixed,
            seg_sigma,
            seg_threshold,
            peak_footprint_size
        )
        if not df_ptu.empty:
            all_dfs.append(df_ptu)
            click.echo(f" --> {len(df_ptu)} molecule(s) found in {ptu_path.name}")
        else:
            click.echo(f" --> No CLSM image data in {ptu_path.name}; skipping.")

    if len(all_dfs) > 0:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = Path(output_file)
        if not combined_path.is_absolute():
            combined_path = (base_path / combined_path).resolve()
        combined_df.to_csv(str(combined_path), sep='\t', index=False)
        click.echo(f"Molecule data saved to: {combined_path}")
    else:
        click.echo("No PTU files produced molecule data; combined TSV not generated.")

    click.echo("Processing complete.")


if __name__ == '__main__':
    run()
