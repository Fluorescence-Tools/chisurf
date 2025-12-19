import pathlib
import typing

import numpy as np
import pandas as pd

from typing import TypedDict

from chisurf.gui import QtGui


QValidator = QtGui.QValidator


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


def create_mti_summary(
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


def create_bur_summary(start_stop, filename, tttr, windows, detectors):
    """
    A vectorized approach to compute burst summary information.

    :param start_stop: List of tuples (start_index, stop_index)
    :param filename: String representing the file name
    :param tttr: TTTR object with attributes:
                 - macro_times
                 - micro_times
                 - routing_channel
                 - header.macro_time_resolution
    :param windows: Dictionary {window_name: [(r_start, r_stop), ...]}
    :param detectors: Dictionary {det_name: {"chs": [...], "micro_time_ranges": [(mt_start, mt_stop), ...]}}
    :return: A pandas DataFrame with one row per burst, including detector + window stats.
    """

    # Extract main arrays once
    n_max = len(tttr)
    macro_times = tttr.macro_times
    micro_times = tttr.micro_times
    routing_channels = tttr.routing_channel
    res = tttr.header.macro_time_resolution

    summary_data = []

    # Loop over each burst defined by (start_index, stop_index)
    for start_index, stop_index in start_stop:
        if start_index > n_max or stop_index > n_max:
            continue

        # Slice once for this burst
        burst_rout = routing_channels[start_index:stop_index]
        burst_micro = micro_times[start_index:stop_index]
        burst_macro = macro_times[start_index:stop_index]

        # Compute global burst stats
        if stop_index <= start_index:
            # Handle corner cases (e.g., empty slice)
            duration = 0.0
            mean_macro_time = 0.0
            n_photons = 0
        else:
            duration = (macro_times[stop_index] - macro_times[start_index]) * res
            mean_macro_time = ((macro_times[stop_index] + macro_times[start_index]) / 2.0) * res
            n_photons = stop_index - start_index

        # Avoid division by zero
        count_rate = n_photons / duration if duration > 0 else np.nan

        # Initialize the row dictionary
        row_data = {
            "First Photon": start_index,
            "Last Photon": stop_index,
            "Duration (ms)": duration * 1e3,
            "Mean Macro Time (ms)": mean_macro_time * 1e3,
            "Number of Photons": n_photons,
            "Count Rate (KHz)": count_rate / 1e3,
            "First File": filename,
            "Last File": filename
        }

        # ----------------------------------------------------------------------
        # 1) Precompute a boolean mask for each detector (channels + micro_time_ranges)
        # ----------------------------------------------------------------------
        detector_masks = {}
        for det_name, det_info in detectors.items():
            chs = det_info["chs"]
            micro_time_ranges = det_info["micro_time_ranges"]

            # (a) channel mask (is routing_channel in the allowed channels?)
            ch_mask = np.isin(burst_rout, chs)

            # (b) microtime mask (is micro_time within one of the specified ranges?)
            micro_mask = np.zeros(len(burst_micro), dtype=bool)
            for (mts, mtp) in micro_time_ranges:
                micro_mask |= (burst_micro >= mts) & (burst_micro < mtp)

            # Combined detector mask
            det_mask = ch_mask & micro_mask
            detector_masks[det_name] = det_mask

        # ----------------------------------------------------------------------
        # 2) Compute per-detector stats for the entire burst
        # ----------------------------------------------------------------------
        for det_name, mask in detector_masks.items():
            idx = np.nonzero(mask)[0]
            if len(idx) == 0:
                # No photons in this detector for the entire burst
                row_data[f"First Photon ({det_name})"] = -1
                row_data[f"Last Photon ({det_name})"] = -1
                row_data[f"Duration ({det_name}) (ms)"] = -1.0
                row_data[f"Mean Macrotime ({det_name}) (ms)"] = -1.0
                row_data[f"Number of Photons ({det_name})"] = 0
                row_data[f"{det_name.capitalize()} Count Rate (KHz)"] = -1.0
            else:
                first_idx = idx[0]
                last_idx = idx[-1]
                num_ph = len(idx)
                dur_color_ms = (burst_macro[last_idx] - burst_macro[first_idx]) * res * 1e3
                mean_mt_color_ms = ((burst_macro[last_idx] + burst_macro[first_idx]) / 2.0) * res * 1e3
                rate_color_khz = (num_ph / dur_color_ms) if dur_color_ms > 0 else np.nan

                row_data[f"First Photon ({det_name})"] = start_index + first_idx
                row_data[f"Last Photon ({det_name})"] = start_index + last_idx
                row_data[f"Duration ({det_name}) (ms)"] = dur_color_ms
                row_data[f"Mean Macrotime ({det_name}) (ms)"] = mean_mt_color_ms
                row_data[f"Number of Photons ({det_name})"] = num_ph
                row_data[f"{det_name.capitalize()} Count Rate (KHz)"] = rate_color_khz

        # ----------------------------------------------------------------------
        # 3) Compute per-detector, per-window stats
        # ----------------------------------------------------------------------
        for window_name, w_ranges in windows.items():
            # If you only need the first (r_start, r_stop) in windows[window_name]:
            (r_start, r_stop) = w_ranges[0]

            # Build a mask for the window's microtime range
            w_mask = (burst_micro >= r_start) & (burst_micro < r_stop)

            for det_name in detectors:
                # Intersection of detector mask with the window mask
                combined_mask = detector_masks[det_name] & w_mask
                idx = np.nonzero(combined_mask)[0]
                if len(idx) == 0:
                    row_data[f"S {window_name} {det_name} (kHz) | {r_start}-{r_stop}"] = -1.0
                else:
                    num_ph = len(idx)
                    dur_window_ms = (burst_macro[idx[-1]] - burst_macro[idx[0]]) * res * 1e3
                    rate_window_khz = (num_ph / dur_window_ms) if dur_window_ms > 0 else np.nan
                    row_data[f"S {window_name} {det_name} (kHz) | {r_start}-{r_stop}"] = rate_window_khz

        # Append row data
        summary_data.append(row_data)

    # Build dataframe
    summary_df = pd.DataFrame(summary_data)
    return summary_df


class CommaSeparatedIntegersValidator(QValidator):

    def validate(self, input_str, pos):
        # Allow empty input
        if not input_str:
            return QValidator.Intermediate, input_str, pos

        # Split the input by commas
        parts = input_str.split(',')

        for part in parts:
            part = part.strip()

            # Allow empty parts (for intermediate states like entering a comma)
            if part == '':
                continue

            # Check if the part is a digit
            if not part.isdigit():
                return QValidator.Intermediate, input_str, pos

            # Convert to integer and check range
            num = int(part)
            if num < 0 or num > 255:
                return QValidator.Invalid, input_str, pos

        # If the input ends with a comma, allow it as intermediate input
        if input_str.endswith(','):
            return QValidator.Intermediate, input_str, pos

        # If all parts are valid, return Acceptable
        return QValidator.Acceptable, input_str, pos

    def fixup(self, input_str):
        # Remove trailing commas
        input_str = input_str.rstrip(',')

        # Optionally, fix invalid parts (in case there are out-of-range numbers)
        parts = input_str.split(',')
        valid_parts = []

        for part in parts:
            part = part.strip()
            if part.isdigit():
                num = int(part)
                if 0 <= num <= 255:
                    valid_parts.append(str(num))

        return ', '.join(valid_parts)


def fill_small_gaps_in_array(arr, max_gap):
    # Identify where the array changes from 1 to 0 and 0 to 1
    is_burst = np.diff(arr, prepend=0, append=0)
    starts = np.where(is_burst == 1)[0]
    stops = np.where(is_burst == -1)[0]

    # Calculate gap sizes between consecutive bursts
    gaps = starts[1:] - stops[:-1] - 1

    # Identify which gaps are small enough to fill
    small_gaps = np.where(gaps <= max_gap)[0]

    # Fill small gaps by setting the values in those gaps to 1
    for idx in small_gaps:
        arr[stops[idx]:starts[idx + 1]] = 1

    return arr


def find_bursts(arr, max_gap=0):
    # Find where the array changes from 0 to 1 (start of burst) and 1 to 0 (end of burst)
    is_burst = np.diff(arr, prepend=0, append=0)
    starts = np.where(is_burst == 1)[0]
    stops = np.where(is_burst == -1)[0]

    # If max_gap is greater than 0, merge small gaps
    if max_gap > 0:
        merged_starts = [starts[0]]  # Initialize with the first start
        merged_stops = []

        for i in range(1, len(starts)):
            # Check if the gap between current stop and next start is small enough to merge
            if starts[i] - stops[i - 1] - 1 <= max_gap:
                continue  # Skip this start, effectively merging
            else:
                merged_stops.append(stops[i - 1])
                merged_starts.append(starts[i])

        # Append the final stop
        merged_stops.append(stops[-1])

        # Convert merged lists to NumPy arrays
        starts = np.array(merged_starts)
        stops = np.array(merged_stops)

    # Stack the starts and stops into a 2D array
    bursts = np.column_stack((starts, stops - 1))  # stop is exclusive, so subtract 1

    return bursts


class CountRateFilterSettings(TypedDict):
    n_ph_max: int
    time_window: float
    invert: bool


class DeltaMacroTimeFilterSettings(TypedDict):
    dT_min: float
    dT_max: float
    dT_min_active: bool
    dT_max_active: bool


class PhotonFilterSettings(TypedDict):
    count_rate_filter: CountRateFilterSettings
    delta_macro_time: DeltaMacroTimeFilterSettings
