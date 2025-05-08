from collections import OrderedDict
from typing import Dict, Tuple

import chisurf
import numpy as np
import pathlib
import pandas as pd

import tttrlib


def write_mti_summary(
        filename: pathlib.Path,
        analysis_dir: pathlib.Path,
        max_macro_time,
        append: bool = True
):
    """
    Creates or appends to an MTI file in the 'Info' folder. If any MTI file exists, appends to it;
    otherwise, creates a new one based on the given filename.

    Args:
        filename (pathlib.Path): Path to the original file (e.g., '.ht3').
        analysis_dir (pathlib.Path): Directory where the 'Info' folder will be created if missing.
        max_macro_time: The maximum macro time (last photon time) to log.
        append (bool): If True, appends to the first existing MTI file found, otherwise creates a new file.

    Example:
        Creates or appends to an MTI file in:
        c:/analysis_directory/Info/Split_60_132_tween0p00001-0000.mti
        With entry:
        c:/data/Split_60_132_tween0p00001-0000.ht3   74860.905977
    """
    # Create the 'Info' directory if it doesn't exist
    parent_directory = analysis_dir / 'Info'
    parent_directory.mkdir(exist_ok=True, parents=True)

    # Search for any existing .mti files in the 'Info' folder
    existing_mti_files = list(parent_directory.glob("*.mti"))

    # If there are any existing .mti files, append to the first one found
    if existing_mti_files and append:
        mti_filename = existing_mti_files[0]
        mode = 'a'
    else:
        # If no .mti files are found or append is False, create a new file based on the filename
        mti_filename = parent_directory / f"{filename.stem}.mti"
        mode = 'w'

    # Write the filename and max_macro_time to the .mti file
    with open(mti_filename, mode) as mti_file:
        mti_file.write(f"{filename}\t{max_macro_time:.6f}\n")


def write_bv4_analysis(df: pd.DataFrame, analysis_folder: str = "analysis"):
    """
    Writes Burst Variance Analysis (BVA) results to .bv4 files in a 'bv4' subfolder inside the
    specified analysis folder. Each TTTR file will have a corresponding .bv4 file containing
    the mean and standard deviation of the proximity ratio for each burst.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing burst data with columns 'First File', 'Proximity Ratio Mean',
        and 'Proximity Ratio Std'.
    analysis_folder : str, optional
        The path to the folder where the 'bv4' subfolder will be created. Default is 'analysis'.
    """

    # Use pathlib to create the analysis/bv4 folder if it doesn't exist
    bv4_folder = pathlib.Path(analysis_folder) / "bv4"
    bv4_folder.mkdir(parents=True, exist_ok=True)

    # Iterate through the DataFrame and write results to individual .bv4 files
    for _, row in df.iterrows():
        # Create the corresponding .bv4 file name based on the 'First File' column
        tttr_stem = pathlib.Path(row['First File']).stem
        bv4_filename = bv4_folder / f"{tttr_stem}.bv4"

        # Prepare a mini DataFrame with the required columns for the .bv4 file
        data = {
            'Mean Proximity Ratio': [row['Proximity Ratio Mean']],
            'Standard Deviation': [row['Proximity Ratio Std']]  # Standard deviation
        }
        bv4_df = pd.DataFrame(data)

        # Write the mini DataFrame to a .bv4 file using tab as the separator
        bv4_df.to_csv(bv4_filename, sep='\t', index=False)

    chisurf.logging.info(f"BVA results have been written to .bv4 files in the '{bv4_folder}' directory.")


def get_indices_in_ranges(rout, mt, chs, micro_time_ranges):
    # Create a boolean mask for the rout values in chs
    rout_mask = np.isin(rout, chs)

    # Create a boolean mask for the mt values in micro_time_ranges
    mt_mask = np.zeros(mt.shape, dtype=bool)
    for start, end in micro_time_ranges:
        mt_mask |= (mt >= start) & (mt <= end)

    # Get indices where both masks are true
    indices = np.where(rout_mask & mt_mask)[0]

    return indices.tolist()


def write_bur_file_old(bur_filename, start_stop, filename, tttr, windows, detectors):
    """
    Write burst summary information to a TSV file (tab-separated),
    using a vectorized approach for efficiency.

    Modified to match a format where zero rows are interleaved between
    the computed rows and an extra (empty) column is added at the end.
    Ensures that even if there are no bursts, the output file contains
    a header row with the expected columns.

    :param bur_filename: Output filename for the TSV summary.
    :param start_stop: List of tuples (start_index, stop_index) defining bursts.
    :param filename: String representing the file name.
    :param tttr: A TTTR-like object with:
                 - macro_times
                 - micro_times
                 - routing_channel
                 - header.macro_time_resolution
    :param windows: Dictionary {window_name: (r_start, r_stop)}
    :param detectors: Dictionary {det_name: {"chs": [...], "micro_time_ranges": [(mt_start, mt_stop), ...]}}
    """
    import numpy as np
    import pandas as pd
    from collections import OrderedDict

    # Unpack arrays and resolution
    n_ph = len(tttr)
    macro_times = tttr.macro_times
    micro_times = tttr.micro_times
    routing_channels = tttr.routing_channel
    res = tttr.header.macro_time_resolution

    # ---------------------------------------------------------
    # Precompute the full list of column headers
    # ---------------------------------------------------------
    static_cols = [
        "First Photon", "Last Photon", "Duration (ms)", "Mean Macro Time (ms)",
        "Number of Photons", "Count Rate (KHz)", "First File", "Last File"
    ]
    det_cols = []
    for det_name in detectors:
        det_cols += [
            f"First Photon ({det_name})", f"Last Photon ({det_name})",
            f"Duration ({det_name}) (ms)", f"Mean Macrotime ({det_name}) (ms)",
            f"Number of Photons ({det_name})", f"{det_name.capitalize()} Count Rate (KHz)"
        ]
    window_cols = []
    for window_name, (r_start, r_stop) in windows.items():
        for det_name in detectors:
            window_cols.append(
                f"S {window_name} {det_name} (kHz) | {r_start}-{r_stop}"
            )
    # extra empty column
    header_keys = static_cols + det_cols + window_cols + [""]

    summary_rows = []

    # Helper: create a zero row dict
    def create_zero_row(keys):
        row = OrderedDict()
        for key in keys:
            row[key] = "" if key == "" else 0
        return row

    # ---------------------------------------------------------
    # Iterate and build rows
    # ---------------------------------------------------------
    for start_idx, stop_idx in start_stop:
        if stop_idx > n_ph or stop_idx < 0:
            continue

        burst_macro = macro_times[start_idx:stop_idx]
        burst_micro = micro_times[start_idx:stop_idx]
        burst_rout = routing_channels[start_idx:stop_idx]

        if stop_idx <= start_idx:
            duration = mean_macro_time = n_photons = 0
        else:
            duration = (macro_times[stop_idx] - macro_times[start_idx]) * res
            mean_macro_time = ((macro_times[stop_idx] + macro_times[start_idx]) / 2.0) * res
            n_photons = stop_idx - start_idx
        count_rate = (n_photons / duration) if duration > 0 else np.nan

        # base row data
        row_data = OrderedDict([
            ("First Photon", start_idx),
            ("Last Photon", stop_idx),
            ("Duration (ms)", duration * 1e3),
            ("Mean Macro Time (ms)", mean_macro_time * 1e3),
            ("Number of Photons", n_photons),
            ("Count Rate (KHz)", count_rate / 1e3),
            ("First File", filename),
            ("Last File", filename),
        ])

        # detector masks and per-detector stats
        detector_masks = {}
        for det_name, det_info in detectors.items():
            ch_mask = np.isin(burst_rout, det_info["chs"])
            mt_mask = np.zeros(len(burst_micro), bool)
            for mt_start, mt_stop in det_info["micro_time_ranges"]:
                mt_mask |= (burst_micro >= mt_start) & (burst_micro < mt_stop)
            detector_masks[det_name] = ch_mask & mt_mask

        for det_name, mask in detector_masks.items():
            idxs = np.nonzero(mask)[0]
            if len(idxs) == 0:
                row_data.update({
                    f"First Photon ({det_name})": -1,
                    f"Last Photon ({det_name})": -1,
                    f"Duration ({det_name}) (ms)": -1.0,
                    f"Mean Macrotime ({det_name}) (ms)": -1.0,
                    f"Number of Photons ({det_name})": 0,
                    f"{det_name.capitalize()} Count Rate (KHz)": -1.0,
                })
            else:
                first_i, last_i = idxs[0], idxs[-1]
                dur_ms = (burst_macro[last_i] - burst_macro[first_i]) * res * 1e3
                mean_mt_ms = ((burst_macro[last_i] + burst_macro[first_i]) / 2.0) * res * 1e3
                rate_khz = (len(idxs) / dur_ms) if dur_ms > 0 else np.nan
                row_data.update({
                    f"First Photon ({det_name})": start_idx + first_i,
                    f"Last Photon ({det_name})": start_idx + last_i,
                    f"Duration ({det_name}) (ms)": dur_ms,
                    f"Mean Macrotime ({det_name}) (ms)": mean_mt_ms,
                    f"Number of Photons ({det_name})": len(idxs),
                    f"{det_name.capitalize()} Count Rate (KHz)": rate_khz,
                })

        # per-window, per-detector stats
        for window_name, (r_start, r_stop) in windows.items():
            w_mask = (burst_micro >= r_start) & (burst_micro < r_stop)
            for det_name in detectors:
                combined = detector_masks[det_name] & w_mask
                idxs = np.nonzero(combined)[0]
                key = f"S {window_name} {det_name} (kHz) | {r_start}-{r_stop}"
                if len(idxs) == 0:
                    row_data[key] = -1.0
                else:
                    dur_win_ms = (burst_macro[idxs[-1]] - burst_macro[idxs[0]]) * res * 1e3
                    row_data[key] = (len(idxs) / dur_win_ms) if dur_win_ms > 0 else np.nan

        # append empty column
        row_data[""] = ""

        # interleave zero rows
        if not summary_rows:
            summary_rows.append(create_zero_row(header_keys))
        summary_rows.append(row_data)
        summary_rows.append(create_zero_row(header_keys))

    # ---------------------------------------------------------
    # Build DataFrame and write TSV, ensuring header is always present
    # ---------------------------------------------------------
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
    else:
        summary_df = pd.DataFrame(columns=header_keys)
    summary_df.to_csv(bur_filename, sep='\t', index=False)


def write_bur_file_fast(bur_filename, start_stop, filename, tttr, windows, detectors):
    """Much faster burst summary writer by:
       1) precomputing global detector/window masks,
       2) building fixed-length lists instead of OrderedDict,
       3) appending to a list of lists and dumping to pandas once."""
    file_name_only = pathlib.Path(filename).name

    # unpack
    macro = tttr.macro_times
    micro = tttr.micro_times
    rout  = tttr.routing_channel
    res   = tttr.header.macro_time_resolution
    n_ph  = len(tttr)

    # build column list
    static_cols = [
        "First Photon", "Last Photon", "Duration (ms)", "Mean Macro Time (ms)",
        "Number of Photons", "Count Rate (KHz)", "First File", "Last File",
    ]
    det_cols = []
    for d in detectors:
        det_cols += [
            f"First Photon ({d})", f"Last Photon ({d})",
            f"Duration ({d}) (ms)", f"Mean Macrotime ({d}) (ms)",
            f"Number of Photons ({d})", f"{d.capitalize()} Count Rate (KHz)",
        ]
    win_cols = []
    for w,(r0,r1) in windows.items():
        for d in detectors:
            win_cols.append(f"S {w} {d} (kHz) | {r0}-{r1}")
    # extra blank column
    cols = static_cols + det_cols + win_cols + [""]

    # map col→index for fast assignment
    idx = {c:i for i,c in enumerate(cols)}
    n_cols = len(cols)

    # precompute global masks so we don't remake them per-burst
    det_global = {}
    for d,info in detectors.items():
        chm = np.isin(rout, info["chs"])
        mtm = np.zeros(n_ph, bool)
        for r0,r1 in info["micro_time_ranges"]:
            mtm |= (micro >= r0) & (micro < r1)
        det_global[d] = chm & mtm

    win_global = {
        w: (micro >= r0) & (micro < r1)
        for w,(r0,r1) in windows.items()
    }

    # helper zero-row
    zero_row = [0]*n_cols
    zero_row[-1] = ""  # last col blank string

    out = []
    # only add the leading zero‐row when there's at least one burst
    try:
        has_bursts = len(start_stop) > 0
    except TypeError:
        # fallback if start_stop isn’t sized like a sequence
        has_bursts = bool(start_stop)
    if has_bursts:
        out.append(zero_row.copy())

    for start, stop in start_stop:
        if stop <= start or stop>n_ph or start<0:
            continue

        # allocate a fresh row
        row = zero_row.copy()

        # static stats
        dur   = (macro[stop] - macro[start]) * res * 1e3
        meanm = ((macro[stop] + macro[start]) / 2) * res * 1e3
        npix  = stop - start
        crate = (npix / dur)/1e3 if dur>0 else np.nan

        row[idx["First Photon"]]          = start
        row[idx["Last Photon"]]           = stop
        row[idx["Duration (ms)"]]         = dur
        row[idx["Mean Macro Time (ms)"]]  = meanm
        row[idx["Number of Photons"]]     = npix
        row[idx["Count Rate (KHz)"]]      = crate
        row[idx["First File"]]            = file_name_only
        row[idx["Last File"]]             = file_name_only

        # slice views
        sl = slice(start, stop)
        for d in detectors:
            mask = det_global[d][sl]
            idxs = np.nonzero(mask)[0]
            col0 = f"First Photon ({d})"
            if idxs.size == 0:
                # these get -1 or 0 per your original logic
                row[idx[col0]]                             = -1
                row[idx[f"Last Photon ({d})"]]            = -1
                row[idx[f"Duration ({d}) (ms)"]]           = -1.0
                row[idx[f"Mean Macrotime ({d}) (ms)"]]     = -1.0
                row[idx[f"Number of Photons ({d})"]]       = 0
                row[idx[f"{d.capitalize()} Count Rate (KHz)"]] = -1.0
            else:
                i0, i1 = idxs[0], idxs[-1]
                abs0, abs1 = start + i0, start + i1
                d_ms = (macro[abs1] - macro[abs0]) * res * 1e3
                m_ms = ((macro[abs1] + macro[abs0]) / 2) * res * 1e3
                rate = (idxs.size / d_ms) if d_ms>0 else np.nan

                row[idx[col0]]                             = abs0
                row[idx[f"Last Photon ({d})"]]            = abs1
                row[idx[f"Duration ({d}) (ms)"]]           = d_ms
                row[idx[f"Mean Macrotime ({d}) (ms)"]]     = m_ms
                row[idx[f"Number of Photons ({d})"]]       = idxs.size
                row[idx[f"{d.capitalize()} Count Rate (KHz)"]] = rate

        # now per-window, per-detector
        for w in windows:
            wmask = win_global[w][sl]
            for d in detectors:
                combined = det_global[d][sl] & wmask
                idxs = np.nonzero(combined)[0]
                key = f"S {w} {d} (kHz) | {windows[w][0]}-{windows[w][1]}"
                if idxs.size == 0:
                    row[idx[key]] = -1.0
                else:
                    abs0, abs1 = start+idxs[0], start+idxs[-1]
                    d_ms = (macro[abs1] - macro[abs0]) * res * 1e3
                    row[idx[key]] = (idxs.size / d_ms) if d_ms>0 else np.nan

        # blank column already set to ""
        out.append(row)
        out.append(zero_row.copy())

    # build and write
    df = pd.DataFrame(out, columns=cols)
    df.to_csv(bur_filename, sep="\t", index=False)

write_bur_file = write_bur_file_fast

def read_burst_analysis(
        paris_path: pathlib.Path,
        tttr_file_type: str,
        pattern: str = 'b*4*',
        row_stride: int = 1
) -> (pd.DataFrame, Dict[str, tttrlib.TTTR]):
    """
    Reads and processes burst analysis data files from a specified directory,
    constructs a pandas DataFrame with the concatenated data, and populates a
    dictionary containing TTTR (Time Tagging and Time Resolved) data.

    This function supports TTTR file types as defined by the tttrlib library and
    allows for flexible reading of files based on a specified glob pattern. It
    handles data from multiple files and can accommodate files generated by
    Seidel software, which may require skipping additional rows.

    Parameters:
    ----------
    paris_path : pathlib.Path
        The path to the directory containing the burst analysis data files.
    tttr_file_type : str
        The file type for TTTR processing (e.g., 'PTU', 'HDF5', etc.), as supported by tttrlib.
    pattern : str, optional
        A glob pattern to match files in the directory. The default is 'b*4*',
        which will match files that start with 'b', contain '4', and have any extension.
    row_stride : int, optional
        The number of rows to skip between reads. The default is 1.
        If the data files are created by Seidel software (e.g., PARIS software),
        set `row_stride` to 2 to account for additional header rows that need to be skipped.

    Returns:
    -------
    Tuple[pd.DataFrame, Dict[str, tttrlib.TTTR]]
        A tuple containing:
        - A pandas DataFrame with the concatenated data from all matched files.
        - A dictionary with keys as filenames (from the 'First File' column) and values
          as TTTR data objects corresponding to those files.

    Raises:
    ------
    FileNotFoundError
        If the specified `paris_path` does not exist or is not a directory.
    ValueError
        If any conversion to numeric fails for columns after the first file is processed.

    Examples:
    --------
    >>> df, tttrs = read_burst_analysis(pathlib.Path('/path/to/data'), 'PTU', pattern='data*')
    >>> print(df.head())
    >>> print(tttrs.keys())
    """

    def update_tttr_dict(data_path, tttrs: Dict[str, tttrlib.TTTR] = dict()):
        for ff, fl in zip(df['First File'], df['Last File']):
            try:
                tttr = tttrs[ff]
            except KeyError:
                fn = str(data_path / ff)
                tttr = tttrlib.TTTR(fn, tttr_file_type)
                tttrs[ff] = tttr
        return tttrs

    info_path = paris_path / 'Info'
    data_path = paris_path.parent

    dfs = list()
    is_first_file = True  # Flag to track the first file
    for path in paris_path.glob(pattern):
        frames = list()
        for fn in sorted(path.glob('*')):
            with open(fn) as f:
                t = f.readlines()
                t = [line.rstrip('\n') for line in t]  # Remove trailing newlines
                h = t[0].split('\t')
                d = [[x for x in l.split('\t')] for l in t[2::row_stride]]
                frames.append(pd.DataFrame(d, columns=h))
        dfs.append(pd.concat(frames))
    df = pd.concat(dfs, axis=1)

    # Loop through each column and attempt to convert to numeric
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            if not is_first_file:  # Ignore conversion errors only for the first file
                chisurf.logging.warning(f"read_burst_analysis: Could not convert {column} to numeric")
        is_first_file = False  # After processing the first file, set flag to False

    tttrs = dict()
    update_tttr_dict(data_path, tttrs)
    return df, tttrs

