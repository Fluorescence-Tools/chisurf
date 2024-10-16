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


def write_bur_file(bur_filename, start_stop, filename, tttr, windows, detectors):
    # Initialize a list to store each row of summary data
    summary_data = []

    # Read the TTTR file
    res = tttr.header.macro_time_resolution

    # Iterate through the list of (start, stop) tuples
    for start_index, stop_index in start_stop:
        burst = tttr[start_index:stop_index]
        duration = (tttr.macro_times[stop_index] - tttr.macro_times[start_index]) * res
        mean_macro_time = (tttr.macro_times[stop_index] + tttr.macro_times[start_index]) / 2.0 * res
        n_photons = abs(stop_index - start_index)
        count_rate = n_photons / duration

        # Create the initial OrderedDict
        row_data = OrderedDict([
            ("First Photon", start_index),
            ("Last Photon", stop_index),
            ("Duration (ms)", duration * 1000.0),
            ("Mean Macro Time (ms)", mean_macro_time * 1000.0),
            ("Number of Photons", n_photons),
            ("Count Rate (KHz)", count_rate / 1000.0),
            ("First File", filename),
            ("Last File", filename),
        ])

        for window in windows:
            r_start, r_stop = windows[window][0]
            for det in detectors:
                # Create selection mask
                chs = detectors[det]["chs"]
                micro_time_ranges = detectors[det]["micro_time_ranges"]

                mt = burst.micro_times
                mT = burst.macro_times
                rout = burst.routing_channel

                # Signal in Detector
                idx = get_indices_in_ranges(rout, mt, chs, micro_time_ranges)
                nbr_ph_color = len(idx)
                if nbr_ph_color == 0:
                    first, last = -1, -1
                    duration_color = -1
                    mean_macro_time_color = -1
                    count_rate_color = -1
                else:
                    first, last = idx[0], idx[-1]
                    nbr_ph_color = len(idx)
                    duration_color = (mT[idx[-1]] - mT[idx[0]]) * res * 1000.0
                    mean_macro_time_color = (mT[last] + mT[first]) / 2.0 * res  * 1000.0
                    count_rate_color = nbr_ph_color / duration_color

                # Signal in Window
                idx_window = get_indices_in_ranges(rout, mt, chs, [(r_start, r_stop)])
                nbr_ph_window = len(idx_window)
                if nbr_ph_window == 0:
                    count_rate_window = -1.0
                else:
                    nbr_ph_window = len(idx_window)
                    duration_window = (mT[idx_window[-1]] - mT[idx_window[0]]) * res * 1000.0
                    count_rate_window = nbr_ph_window / duration_window

                # Create the update dict as an OrderedDict
                c = OrderedDict([
                    (f"First Photon ({det})", first + start_index),
                    (f"Last Photon ({det})", last + start_index),
                    (f"Duration ({det}) (ms)", duration_color),
                    (f"Mean Macrotime ({det}) (ms)", mean_macro_time_color),
                    (f"Number of Photons ({det})", nbr_ph_color),
                    (f"{det}".capitalize() + " Count Rate (KHz)", count_rate_color),
                    (f'S {window} {det} (kHz) | {r_start}-{r_stop}', count_rate_window)
                ])

                # Update row_data with c
                row_data.update(c)

        summary_data.append(row_data)  # Add the row data to the summary list

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(bur_filename, sep='\t', index=False)


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

