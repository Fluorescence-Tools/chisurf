"""
Lifetime fitting functionality.

This module provides classes and functions for fitting fluorescence lifetime data.
"""

from __future__ import annotations
import typing
import json
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from numba import njit

from .convolve import convolve_lifetime_spectrum, add_pile_up_to_model
from .optimization.leastsqbound import leastsqbound
from .scaling import scale_model_to_data
from .settings import get_default_settings


#@njit(nopython=True, nogil=True)
def _shift_irf_numba(original_irf: np.ndarray, shift_ch: float) -> np.ndarray:
    """
    Shift the IRF by a given number of channels.

    Parameters
    ----------
    original_irf : numpy.ndarray
        The original IRF
    shift_ch : float
        The shift in channels (can be fractional)

    Returns
    -------
    numpy.ndarray
        The shifted IRF
    """
    n = len(original_irf)
    shifted = np.zeros_like(original_irf)

    # Split the shift into integer and fractional parts
    int_shift = int(shift_ch)
    frac = shift_ch - int_shift

    if frac == 0:
        # Integer shift - simple case
        for i in range(n):
            dst = i + int_shift
            if 0 <= dst < n:
                shifted[dst] = original_irf[i]
    else:
        # Fractional shift - linear interpolation
        # int_shift is negative, so -int_shift is positive
        for i in range(-int_shift, n):
            dst = i + int_shift
            if 0 <= dst < n:
                # note: idx-1 also in range because i starts at -int_shift
                shifted[dst] = (1.0 - abs(frac)) * original_irf[i] + abs(frac) * original_irf[i - 1]

    return shifted


class Decay:
    """
    Class representing a fluorescence decay.

    Parameters
    ----------
    decay : numpy.ndarray
        The decay data
    irf : numpy.ndarray
        The instrument response function
    time_axis : numpy.ndarray
        The time axis
    """

    def __init__(
            self,
            decay: np.ndarray = None,
            irf: np.ndarray = None,
            time_axis: np.ndarray = None,
    ):
        self.decay = decay
        self._original_irf = irf
        self.time_axis = time_axis

        # Parameters
        self.lifetime_spectrum = np.array([1.0, 1.0])  # [amplitude, lifetime]
        self.irf_shift = 0.0
        self.irf_background = 0.0
        self.decay_background = 0.0
        self.apply_irf_shift = True  # Whether to apply IRF shift

        # Analysis range
        self.start = 0
        self.stop = len(decay) - 1 if decay is not None else 0

        # Model decay
        self.model_decay = None

        # Fit results
        self.fit_result = None

        # Pile-up correction parameters
        self.correct_pile_up = False
        self.rep_rate = 80.0  # MHz
        self.dead_time = 85.0  # ns
        self.measurement_time = 60.0  # seconds


    @property
    def irf(self) -> np.ndarray:
        """
        Returns the IRF shifted by self.irf_shift (seconds) along self.time_axis,
        using linear interpolation for non‐integer shifts. Accelerated with Numba.
        """
        if self._original_irf is None:
            return None

        if not self.apply_irf_shift:
            return self._original_irf.copy()

        # compute how many "channels" to shift by
        shift_ch = self.irf_shift / 2.0

        # call the JIT‐compiled routine
        return _shift_irf_numba(self._original_irf, shift_ch)


    def set_analysis_range(self, start: int, stop: int):
        """
        Set the analysis range.

        Parameters
        ----------
        start : int
            Start index
        stop : int
            Stop index
        """
        self.start = start
        self.stop = stop

    def get_analysis_range(
            self,
            count_threshold: float = 10.0,
            area: float = 0.999,
            start_at_peak: bool = True,
            start_fraction: float = 0.1,
            skip_first_channels: int = 0,
            skip_last_channels: int = 0,
            verbose: bool = True
    ) -> typing.Tuple[int, int]:
        """
        Get the analysis range based on the decay data.

        Parameters
        ----------
        count_threshold : float
            Count threshold for determining the analysis range
        area : float
            Fraction of the total area to include in the analysis range
        start_at_peak : bool
            Whether to start the analysis range at the peak of the decay
        start_fraction : float
            Fraction of the peak value to use as the start threshold
        skip_first_channels : int
            Number of channels to skip at the beginning
        skip_last_channels : int
            Number of channels to skip at the end
        verbose : bool
            Whether to print verbose output

        Returns
        -------
        tuple
            (start, stop) indices for the analysis range
        """
        if self.decay is None:
            return (0, 0)

        # Find the peak of the decay
        peak_idx = np.argmax(self.decay)
        peak_value = self.decay[peak_idx]

        # Determine start index
        if start_at_peak:
            start_idx = peak_idx
        else:
            # Find where the decay rises above the start_fraction of the peak
            start_threshold = peak_value * start_fraction
            for i in range(peak_idx, 0, -1):
                if self.decay[i] < start_threshold:
                    start_idx = i
                    break
            else:
                start_idx = 0

        # Skip first channels if requested
        start_idx = max(start_idx, skip_first_channels)

        # Determine stop index based on area
        if area < 1.0:
            # Calculate cumulative sum
            cum_sum = np.cumsum(self.decay[start_idx:])
            total_sum = cum_sum[-1]

            # Find where the cumulative sum reaches the desired area
            target_sum = total_sum * area
            for i, s in enumerate(cum_sum):
                if s >= target_sum:
                    stop_idx = start_idx + i
                    break
            else:
                stop_idx = len(self.decay) - 1
        else:
            stop_idx = len(self.decay) - 1

        # Apply count threshold
        for i in range(stop_idx, start_idx, -1):
            if self.decay[i] >= count_threshold:
                stop_idx = i
                break

        # Skip last channels if requested
        stop_idx = min(stop_idx, len(self.decay) - skip_last_channels - 1)

        # Set the analysis range
        self.start = start_idx
        self.stop = stop_idx + 1  # +1 because stop is exclusive

        if verbose:
            print(f"Analysis range: {self.start} to {self.stop}")
            print(f"Time range: {self.time_axis[self.start]:.2f} to {self.time_axis[self.stop-1]:.2f}")

        return (self.start, self.stop)

    def estimate_background(
            self,
            irf_fwhm_range: typing.Tuple[int, int] = None,
            average_window: int = 10,
            verbose: bool = True
    ) -> float:
        """
        Estimate the background level in the decay.

        Parameters
        ----------
        irf_fwhm_range : tuple
            (start, stop) indices for the IRF FWHM range
        average_window : int
            Number of channels to average for background estimation
        verbose : bool
            Whether to print verbose output

        Returns
        -------
        float
            Estimated background level
        """
        if self.decay is None:
            return 0.0

        # Use the last average_window channels to estimate background
        bg_start = max(0, len(self.decay) - average_window)
        bg_end = len(self.decay)

        # Calculate average background
        bg = np.mean(self.decay[bg_start:bg_end])

        # Set the background
        self.decay_background = bg

        if verbose:
            print(f"Estimated decay background: {bg:.2f}")

        return bg

    def estimate_irf_shift(
            self,
            irf_fwhm_range: typing.Tuple[int, int] = None,
            irf_time_shift_scan_range: typing.Tuple[float, float] = (-8.0, 8.0),
            irf_time_shift_scan_n_steps: int = 20,
            verbose: bool = True
    ) -> float:
        """
        Estimate the IRF shift by scanning a range of shifts and finding the one
        that minimizes the chi-square between the model and the data.

        Parameters
        ----------
        irf_fwhm_range : tuple
            (start, stop) indices for the IRF FWHM range
        irf_time_shift_scan_range : tuple
            (min, max) range for the IRF shift scan in nanoseconds
        irf_time_shift_scan_n_steps : int
            Number of steps for the IRF shift scan
        verbose : bool
            Whether to print verbose output

        Returns
        -------
        float
            Estimated IRF shift
        """
        if self.decay is None or self.irf is None:
            return 0.0

        # Create a range of shifts to scan
        shifts = np.linspace(
            irf_time_shift_scan_range[0],
            irf_time_shift_scan_range[1],
            irf_time_shift_scan_n_steps
        )

        # Initialize scores array
        scores = np.zeros(len(shifts))

        # Scan through shifts
        for i, shift in enumerate(shifts):
            # Set the shift
            self.irf_shift = shift

            # Calculate model decay
            self.calculate_model_decay()

            # Calculate chi-square
            residuals = self.decay[self.start:self.stop] - self.model_decay[self.start:self.stop]
            weights = 1.0 / np.sqrt(np.maximum(self.decay[self.start:self.stop], 1.0))
            chi_square = np.sum((residuals * weights) ** 2)

            # Store score
            scores[i] = chi_square

        # Find the shift with the minimum chi-square
        best_idx = np.argmin(scores)
        best_shift = shifts[best_idx]

        if verbose:
            print(f"Estimated IRF shift: {best_shift:.3f} ns")

        self.irf_shift = best_shift
        return best_shift

    def calculate_model_decay(self):
        """
        Calculate the model decay based on the current parameters.
        """
        if self.decay is None or self.irf is None:
            return

        # Create model decay
        model = np.zeros_like(self.decay)

        # Apply IRF background to the IRF
        irf_with_background = self.irf.copy()
        irf_with_background += self.irf_background

        # 1. Convolution - Convolve lifetime spectrum with IRF
        convolve_lifetime_spectrum(
            model,
            self.lifetime_spectrum,
            irf_with_background,
            convolution_stop=self.stop,
            time_axis=self.time_axis
        )

        # 2. Scaling - Scale model to data
        scale = scale_model_to_data(
            model_decay=model,
            experimental_decay=self.decay,
            start=self.start,
            stop=self.stop,
            experimental_background=self.decay_background,
            use_weights=True  # Use weights for better scaling
        )

        # 3. Pile-up correction if enabled
        if self.correct_pile_up:
            add_pile_up_to_model(
                data=self.decay,
                model=model,
                rep_rate=self.rep_rate,
                dead_time=self.dead_time,
                measurement_time=self.measurement_time,
                modify_inplace=True
            )

        # 4. Background - Add background
        model += self.decay_background

        # 5. Ensure all values are non-negative
        model = np.maximum(model, 0)

        self.model_decay = model

    def objective_function(
            self,
            params: np.ndarray,
            fixed: typing.List[bool] = None
    ) -> np.ndarray:
        """
        Objective function for fitting.

        Parameters
        ----------
        params : numpy.ndarray
            Parameters to fit
        fixed : list of bool
            Whether each parameter is fixed

        Returns
        -------
        numpy.ndarray
            Weighted residuals
        """
        if fixed is None:
            fixed = [False] * len(params)

        # Extract parameters
        param_idx = 0

        # Lifetime spectrum
        n_lifetimes = len(self.lifetime_spectrum) // 2
        for i in range(n_lifetimes):
            if not fixed[param_idx]:
                self.lifetime_spectrum[2 * i] = params[param_idx]  # Amplitude
            param_idx += 1

            if not fixed[param_idx]:
                self.lifetime_spectrum[2 * i + 1] = params[param_idx]  # Lifetime
            param_idx += 1

        # Normalize amplitudes to sum to 1
        amplitude_sum = np.sum(self.lifetime_spectrum[::2])
        if amplitude_sum != 0:
           self.lifetime_spectrum[::2] /= amplitude_sum

        # IRF shift
        if not fixed[param_idx]:
            self.irf_shift = params[param_idx]
        param_idx += 1

        # IRF background
        if not fixed[param_idx]:
            self.irf_background = params[param_idx]
        param_idx += 1

        # Decay background
        if not fixed[param_idx]:
            self.decay_background = params[param_idx]
        param_idx += 1

        # Calculate model decay
        self.calculate_model_decay()

        # Calculate weighted residuals
        residuals = self.decay[self.start:self.stop] - self.model_decay[self.start:self.stop]
        weights = 1.0 / np.sqrt(np.maximum(self.decay[self.start:self.stop], 1.0))
        weighted_residuals = residuals * weights

        return weighted_residuals

    def find_optimal_lifetime_spectrum(
            self,
            maximum_number_of_lifetimes: int = 6,
            prob_threshold: float = 0.95,
            verbose: bool = False,
            plot_probabilities: bool = True,
            plot_weighted_residuals: bool = True,
            min_lifetime: float = 0.5,
            max_lifetime: float = 5.0,
            selection_mode: str = 'lower',
            save_intermediate_results: bool = True,
            intermediate_results_base_filename: str = None
    ) -> dict:
        """
        Find the optimal number of fluorescence lifetimes.

        Parameters
        ----------
        maximum_number_of_lifetimes : int
            Maximum number of lifetimes to try
        prob_threshold : float
            Probability threshold for selecting the best number of lifetimes
        verbose : bool
            Whether to print verbose output
        plot_probabilities : bool
            Whether to plot the probabilities
        plot_weighted_residuals : bool
            Whether to plot the weighted residuals
        min_lifetime : float
            Minimum lifetime value in nanoseconds
        max_lifetime : float
            Maximum lifetime value in nanoseconds
        selection_mode : str
            Mode for selecting the best number of lifetimes ('lower' or 'higher')
        save_intermediate_results : bool
            Whether to save intermediate results to JSON files (default: True)
        intermediate_results_base_filename : str
            Base filename for intermediate results (defaults to output_file without extension if not provided)

        Returns
        -------
        dict
            Results of the optimal lifetime spectrum search
        """
        if self.decay is None or self.irf is None:
            return {}

        # Store original parameters
        original_lifetime_spectrum = self.lifetime_spectrum.copy()
        original_irf_shift = self.irf_shift
        original_irf_background = self.irf_background
        original_decay_background = self.decay_background

        # Initialize arrays to store results
        n_lifetimes_tried = list(range(1, maximum_number_of_lifetimes + 1))
        scores = []
        best_params = []

        # Try different numbers of lifetimes
        for n in n_lifetimes_tried:
            if verbose:
                print(f"Trying {n} lifetimes...")

            # Fit with n lifetimes
            self.fit(
                n_lifetimes=n,
                verbose=verbose,
                randomize_initial_values=True,
                min_lifetime=min_lifetime,
                max_lifetime=max_lifetime
            )

            # Store results
            scores.append(self.fit_result['reduced_chi_square'])
            best_params.append({
                'lifetime_spectrum': self.lifetime_spectrum.copy(),
                'irf_shift': self.irf_shift,
                'irf_background': self.irf_background,
                'decay_background': self.decay_background
            })

            # Save intermediate results if requested
            if save_intermediate_results and intermediate_results_base_filename is not None:
                # Create a temporary fit_result with the current parameters
                temp_fit_result = {
                    'n_lifetimes': n,
                    'lifetime_spectrum': self.lifetime_spectrum.tolist(),
                    'irf_shift': float(self.irf_shift),
                    'irf_background': float(self.irf_background),
                    'decay_background': float(self.decay_background),
                    'chi_square': float(self.fit_result['chi_square']),
                    'reduced_chi_square': float(self.fit_result['reduced_chi_square']),
                    'dof': int(self.fit_result['dof']),
                    'time_range': {
                        'start': float(self.time_axis[self.start]),
                        'stop': float(self.time_axis[self.stop-1]),
                        'start_idx': int(self.start),
                        'stop_idx': int(self.stop)
                    }
                }

                # Extract lifetimes and amplitudes for easier access
                lifetimes = []
                for i in range(n):
                    lifetimes.append({
                        'amplitude': float(self.lifetime_spectrum[2*i]),
                        'lifetime': float(self.lifetime_spectrum[2*i+1])
                    })
                temp_fit_result['lifetimes'] = lifetimes

                # Create filename for this intermediate result
                intermediate_filename = f"{intermediate_results_base_filename}_n{n}.json"

                # Save to file
                with open(intermediate_filename, 'w') as f:
                    f.write(json.dumps(temp_fit_result, indent=2))

                if verbose:
                    print(f"Saved intermediate result to {intermediate_filename}")

        # Using ucfret logic to select the optimal number of lifetimes
        # Convert scores to probabilities using F-test
        probs = []
        # First probability is always 1.0 (for n=1)
        probs.append(1.0)

        # Calculate probabilities for n > 1
        for i in range(1, len(scores)):
            # Calculate degrees of freedom for F-test
            # df1 = difference in number of parameters between models (2 parameters per lifetime)
            df1 = 2
            # df2 = number of data points - number of parameters in the more complex model
            # For a model with i lifetimes, we have 2*i + 3 parameters
            # (2 per lifetime + 3 additional: irf_shift, irf_background, decay_background)
            df2 = (self.stop - self.start) - (2*n_lifetimes_tried[i] + 3)

            f_value = scores[i-1] / scores[i]
            p = scipy.stats.f.cdf(f_value, df1, df2)
            probs.append(p)

        # Find the best number of lifetimes using ucfret's approach
        best_idx = 0
        search_range = range(1, maximum_number_of_lifetimes)

        if selection_mode == 'upper':  # Similar to ucfret's 'upper' mode
            # Search from highest to lowest number of lifetimes
            search_range = reversed(search_range)
            for i in search_range:
                if (probs[i] > prob_threshold) and (scores[i] < scores[i-1]):
                    best_idx = i
                    break
        else:  # 'lower' mode
            # Search from lowest to highest number of lifetimes
            for i in search_range:
                if (probs[i] < prob_threshold) and (scores[i] < scores[i-1]):
                    best_idx = i - 1
                    break

        # Set the best parameters
        best_n_lifetimes = n_lifetimes_tried[best_idx]
        self.lifetime_spectrum = best_params[best_idx]['lifetime_spectrum']
        self.irf_shift = best_params[best_idx]['irf_shift']
        self.irf_background = best_params[best_idx]['irf_background']
        self.decay_background = best_params[best_idx]['decay_background']

        # Calculate model decay with the best parameters
        self.calculate_model_decay()

        if verbose:
            print(f"Best number of lifetimes: {best_n_lifetimes}")
            for i in range(best_n_lifetimes):
                print(f"Lifetime {i+1}: {self.lifetime_spectrum[2*i+1]:.3f} ns, Amplitude: {self.lifetime_spectrum[2*i]:.3f}")

        # Plot probabilities if requested
        if plot_probabilities:
            plt.figure(figsize=(10, 6))
            plt.bar(n_lifetimes_tried, probs, alpha=0.7)
            plt.axhline(y=prob_threshold, color='r', linestyle='--', label=f'Threshold ({prob_threshold})')
            plt.axvline(x=best_n_lifetimes, color='g', linestyle='--', label=f'Selected ({best_n_lifetimes})')
            plt.xlabel('Number of Lifetimes')
            plt.ylabel('Probability')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Plot weighted residuals if requested
        if plot_weighted_residuals:
            plt.figure(figsize=(12, 8))
            for i, n in enumerate(n_lifetimes_tried):
                # Set parameters
                self.lifetime_spectrum = best_params[i]['lifetime_spectrum']
                self.irf_shift = best_params[i]['irf_shift']
                self.irf_background = best_params[i]['irf_background']
                self.decay_background = best_params[i]['decay_background']

                # Calculate model decay
                self.calculate_model_decay()

                # Calculate weighted residuals
                residuals = self.decay[self.start:self.stop] - self.model_decay[self.start:self.stop]
                weights = 1.0 / np.sqrt(np.maximum(self.decay[self.start:self.stop], 1.0))
                weighted_residuals = residuals * weights

                # Plot
                plt.subplot(len(n_lifetimes_tried), 1, i + 1)
                plt.plot(self.time_axis[self.start:self.stop], weighted_residuals)
                plt.ylabel(f'n={n}')
                plt.grid(True)
                if i == 0:
                    plt.title('Weighted Residuals')
                if i == len(n_lifetimes_tried) - 1:
                    plt.xlabel('Time (ns)')

            plt.tight_layout()
            plt.show()

        # Generate decay curve plot for the selected fit
        self.plot_decay_curve(best_n_lifetimes)

        # Restore the best parameters
        self.lifetime_spectrum = best_params[best_idx]['lifetime_spectrum']
        self.irf_shift = best_params[best_idx]['irf_shift']
        self.irf_background = best_params[best_idx]['irf_background']
        self.decay_background = best_params[best_idx]['decay_background']
        self.calculate_model_decay()

        return {
            'best_number_of_lifetimes': best_n_lifetimes,
            'n_lifetimes': n_lifetimes_tried,
            'scores': scores,
            'probabilities': probs,
            'best_idx': best_idx
        }

    def fit(
            self,
            n_lifetimes: int = 1,
            fixed: typing.List[bool] = None,
            verbose: bool = False,
            randomize_initial_values: bool = False,
            min_lifetime: float = 0.5,
            max_lifetime: float = 5.0,
            amplitude_variation: float = 0.5,
            find_optimal: bool = False,
            maximum_number_of_lifetimes: int = 6,
            prob_threshold: float = 0.95,
            plot_probabilities: bool = True,
            plot_weighted_residuals: bool = True,
            selection_mode: str = 'lower',
            save_intermediate_results: bool = True,
            intermediate_results_base_filename: str = None
    ) -> dict:
        """
        Fit the decay data.

        Parameters
        ----------
        n_lifetimes : int
            Number of lifetimes to fit (ignored if find_optimal is True)
        fixed : list of bool
            Whether each parameter is fixed
        verbose : bool
            Whether to print verbose output
        randomize_initial_values : bool
            Whether to randomize initial values for lifetimes and amplitudes
        min_lifetime : float
            Minimum lifetime value in nanoseconds when randomizing
        max_lifetime : float
            Maximum lifetime value in nanoseconds when randomizing
        amplitude_variation : float
            Factor controlling how much the amplitudes can vary from equal distribution (0-1)
        find_optimal : bool
            Whether to find the optimal number of lifetimes automatically
        maximum_number_of_lifetimes : int
            Maximum number of lifetimes to try when finding optimal
        prob_threshold : float
            Probability threshold for selecting the best number of lifetimes
        plot_probabilities : bool
            Whether to plot the probabilities when finding optimal
        plot_weighted_residuals : bool
            Whether to plot the weighted residuals when finding optimal
        selection_mode : str
            Mode for selecting the best number of lifetimes ('lower' or 'higher')
        save_intermediate_results : bool
            Whether to save intermediate results to JSON files when finding optimal (default: True)
        intermediate_results_base_filename : str
            Base filename for intermediate results (defaults to output_file without extension if not provided)

        Returns
        -------
        dict
            Fit results
        """
        if self.decay is None or self.irf is None:
            return {}

        # If find_optimal is True, use the find_optimal_lifetime_spectrum method
        if find_optimal:
            if verbose:
                print("Finding optimal number of lifetimes...")

            # Find the optimal number of lifetimes
            optimal_result = self.find_optimal_lifetime_spectrum(
                maximum_number_of_lifetimes=maximum_number_of_lifetimes,
                prob_threshold=prob_threshold,
                verbose=verbose,
                plot_probabilities=plot_probabilities,
                plot_weighted_residuals=plot_weighted_residuals,
                min_lifetime=min_lifetime,
                max_lifetime=max_lifetime,
                selection_mode=selection_mode,
                save_intermediate_results=save_intermediate_results,
                intermediate_results_base_filename=intermediate_results_base_filename
            )

            # Update n_lifetimes with the best number found
            n_lifetimes = optimal_result['best_number_of_lifetimes']

            if verbose:
                print(f"Optimal number of lifetimes found: {n_lifetimes}")

            # Calculate chi-square and other metrics for the final result
            residuals = self.decay[self.start:self.stop] - self.model_decay[self.start:self.stop]
            weights = 1.0 / np.sqrt(np.maximum(self.decay[self.start:self.stop], 1.0))
            chi_square = np.sum((residuals * weights) ** 2)
            dof = len(residuals) - (2 * n_lifetimes + 3) + sum(fixed) if fixed else 0
            reduced_chi_square = chi_square / max(1, dof)

            # Store fit results
            self.fit_result = {
                'n_lifetimes': n_lifetimes,
                'lifetime_spectrum': self.lifetime_spectrum.tolist(),
                'irf_shift': float(self.irf_shift),
                'irf_background': float(self.irf_background),
                'decay_background': float(self.decay_background),
                'chi_square': float(chi_square),
                'reduced_chi_square': float(reduced_chi_square),
                'dof': int(dof),
                'optimal_fitting': {
                    'scores': optimal_result['scores'],
                    'probabilities': optimal_result['probabilities'],
                    'n_lifetimes_tried': optimal_result['n_lifetimes']
                }
            }

            if verbose:
                print("Fit results:")
                for i in range(n_lifetimes):
                    print(f"Lifetime {i+1}: {self.lifetime_spectrum[2*i+1]:.3f} ns, Amplitude: {self.lifetime_spectrum[2*i]:.3f}")
                print(f"IRF shift: {self.irf_shift:.3f} ns")
                print(f"IRF background: {self.irf_background:.2f}")
                print(f"Decay background: {self.decay_background:.2f}")
                print(f"Chi-square: {chi_square:.2f}")
                print(f"Reduced chi-square: {reduced_chi_square:.2f}")
                print(f"Degrees of freedom: {dof}")

            return self.fit_result

        # Standard fitting with a fixed number of lifetimes
        # Initialize lifetime spectrum
        self.lifetime_spectrum = np.zeros(2 * n_lifetimes)

        if randomize_initial_values:
            # Distribute lifetimes systematically from large to small within the specified range
            if n_lifetimes > 1:
                # Calculate evenly spaced lifetime values from max to min
                lifetime_values = np.linspace(max_lifetime, min_lifetime, n_lifetimes)
                for i in range(n_lifetimes):
                    # Assign lifetime values in descending order (large to small)
                    self.lifetime_spectrum[2 * i + 1] = lifetime_values[i]
            else:
                # If there's only one lifetime, use the average of min and max
                self.lifetime_spectrum[1] = (min_lifetime + max_lifetime) / 2.0

            # Randomize amplitudes with controlled variation
            base_amplitude = 1.0 / n_lifetimes
            amplitudes = []
            for i in range(n_lifetimes):
                # Vary amplitude around the base value
                variation = random.uniform(1.0 - amplitude_variation, 1.0 + amplitude_variation)
                amplitudes.append(base_amplitude * variation)

            # Normalize amplitudes to sum to 1
            total = sum(amplitudes)
            for i in range(n_lifetimes):
                self.lifetime_spectrum[2 * i] = amplitudes[i] / total

            if verbose:
                print("Randomized initial values:")
                for i in range(n_lifetimes):
                    print(f"  Lifetime {i+1}: {self.lifetime_spectrum[2*i+1]:.3f} ns, Amplitude: {self.lifetime_spectrum[2*i]:.3f}")
        else:
            # Use deterministic initialization
            for i in range(n_lifetimes):
                self.lifetime_spectrum[2 * i] = 1.0 / n_lifetimes  # Equal amplitudes
                self.lifetime_spectrum[2 * i + 1] = 1.0 + i * 2.0  # Increasing lifetimes

        # Initialize parameters
        params = []
        for i in range(n_lifetimes):
            params.append(self.lifetime_spectrum[2 * i])  # Amplitude
            params.append(self.lifetime_spectrum[2 * i + 1])  # Lifetime

        params.append(self.irf_shift)
        params.append(self.irf_background)
        params.append(self.decay_background)

        params = np.array(params)

        # Initialize fixed parameters
        if fixed is None:
            fixed = [False] * len(params)

        # Create bounds for the parameters
        bounds = []
        param_idx = 0

        # Bounds for lifetime spectrum
        for i in range(n_lifetimes):
            # Amplitude bounds: (0, 1)
            bounds.append((0.0, 1.0))
            param_idx += 1

            # Lifetime bounds: (0, 10)
            bounds.append((0.0, 10.0))
            param_idx += 1

        # Bounds for IRF shift (no bounds)
        bounds.append((None, None))
        param_idx += 1

        # Bounds for IRF background (non-negative)
        bounds.append((0.0, None))
        param_idx += 1

        # Bounds for decay background (non-negative)
        bounds.append((0.0, None))

        # Fit using leastsqbound
        result = leastsqbound(
            lambda p: self.objective_function(p, fixed),
            params,
            args=(),
            bounds=bounds,
            full_output=True,
            ftol=1.49012e-8,
            xtol=1.49012e-8,
            gtol=0.0,
            maxfev=0,
            epsfcn=0.0,
            factor=1000,
            diag=None
        )
        # leastsqbound returns (x, cov_x, infodict, mesg, ier) when full_output=True
        params = result[0]  # x is the solution array

        # Extract parameters
        param_idx = 0

        # Lifetime spectrum
        for i in range(n_lifetimes):
            if not fixed[param_idx]:
                self.lifetime_spectrum[2 * i] = params[param_idx]  # Amplitude
            param_idx += 1

            if not fixed[param_idx]:
                self.lifetime_spectrum[2 * i + 1] = params[param_idx]  # Lifetime
            param_idx += 1

        # Take absolute value of amplitudes to ensure they are positive
        self.lifetime_spectrum[::2] = np.abs(self.lifetime_spectrum[::2])

        # Normalize amplitudes to sum to 1
        amplitude_sum = np.sum(self.lifetime_spectrum[::2])
        if amplitude_sum != 0:
            self.lifetime_spectrum[::2] /= amplitude_sum

        # IRF shift
        if not fixed[param_idx]:
            self.irf_shift = params[param_idx]
        param_idx += 1

        # IRF background
        if not fixed[param_idx]:
            self.irf_background = params[param_idx]
        param_idx += 1

        # Decay background
        if not fixed[param_idx]:
            self.decay_background = params[param_idx]
        param_idx += 1

        # Calculate model decay
        self.calculate_model_decay()

        # Calculate chi-square
        residuals = self.decay[self.start:self.stop] - self.model_decay[self.start:self.stop]
        weights = 1.0 / np.sqrt(np.maximum(self.decay[self.start:self.stop], 1.0))
        chi_square = np.sum((residuals * weights) ** 2)
        dof = len(residuals) - len(params) + sum(fixed)
        reduced_chi_square = chi_square / dof

        # Store fit results
        self.fit_result = {
            'n_lifetimes': n_lifetimes,
            'lifetime_spectrum': self.lifetime_spectrum.tolist(),
            'irf_shift': float(self.irf_shift),
            'irf_background': float(self.irf_background),
            'decay_background': float(self.decay_background),
            'chi_square': float(chi_square),
            'reduced_chi_square': float(reduced_chi_square),
            'dof': int(dof)
        }

        if verbose:
            print("Fit results:")
            for i in range(n_lifetimes):
                print(f"Lifetime {i+1}: {self.lifetime_spectrum[2*i+1]:.3f} ns, Amplitude: {self.lifetime_spectrum[2*i]:.3f}")
            print(f"IRF shift: {self.irf_shift:.3f} ns")
            print(f"IRF background: {self.irf_background:.2f}")
            print(f"Decay background: {self.decay_background:.2f}")
            print(f"Chi-square: {chi_square:.2f}")
            print(f"Reduced chi-square: {reduced_chi_square:.2f}")
            print(f"Degrees of freedom: {dof}")

        return self.fit_result

    def plot(self, filename: str = None):
        """
        Plot the decay data and fit.

        Parameters
        ----------
        filename : str
            Filename to save the plot to. If None, the plot will not be created.
        """
        if self.decay is None or filename is None:
            return

        plt.figure(figsize=(10, 8))

        # Plot decay and model
        plt.subplot(211)
        # Plot full data for context
        plt.semilogy(self.time_axis, self.decay, 'b-', label='Data')

        if self.model_decay is not None:
            # Plot model only for fit range
            plt.semilogy(self.time_axis[self.start:self.stop], self.model_decay[self.start:self.stop], 
                         'r-', label='Fit')

        if self.irf is not None:
            # Scale IRF to the maximum of the decay data in the fit range
            max_decay_in_range = np.max(self.decay[self.start:self.stop])
            max_irf = np.max(self.irf)
            plt.semilogy(self.time_axis, self.irf * max_decay_in_range / max_irf, 'g-', label='IRF')

        # Add vertical lines for analysis range
        plt.axvline(x=self.time_axis[self.start], color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=self.time_axis[self.stop-1], color='k', linestyle='--', alpha=0.5)

        plt.xlabel('Time (ns)')
        plt.ylabel('Counts')
        plt.legend()
        plt.grid(True)

        # Plot residuals
        if self.model_decay is not None:
            plt.subplot(212)
            # Calculate residuals only for fit range
            residuals = self.decay[self.start:self.stop] - self.model_decay[self.start:self.stop]
            weights = 1.0 / np.sqrt(np.maximum(self.decay[self.start:self.stop], 1.0))
            weighted_residuals = residuals * weights

            # Plot residuals only for fit range
            plt.plot(self.time_axis[self.start:self.stop], weighted_residuals, 'b-')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)

            # Add vertical lines for analysis range
            plt.axvline(x=self.time_axis[self.start], color='k', linestyle='--', alpha=0.5)
            plt.axvline(x=self.time_axis[self.stop-1], color='k', linestyle='--', alpha=0.5)

            plt.xlabel('Time (ns)')
            plt.ylabel('Weighted Residuals')
            plt.grid(True)

        plt.tight_layout()

        # Save to file
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_decay_curve(self, n_lifetimes: int = None):
        """
        Plot the decay curve for the selected fit, similar to ucfret's approach.

        Parameters
        ----------
        n_lifetimes : int
            Number of lifetimes in the fit. If None, uses the number from fit_result.
        """
        if self.decay is None or self.model_decay is None:
            return

        if n_lifetimes is None and self.fit_result is not None:
            n_lifetimes = self.fit_result.get('n_lifetimes', 1)

        # Create figure with two subplots
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot weighted residuals in the top subplot
        residuals = self.decay[self.start:self.stop] - self.model_decay[self.start:self.stop]
        weights = 1.0 / np.sqrt(np.maximum(self.decay[self.start:self.stop], 1.0))
        weighted_residuals = residuals * weights

        ax[0].plot(self.time_axis[self.start:self.stop], weighted_residuals, 'b-')
        ax[0].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax[0].set_ylabel('Weighted Residuals')
        ax[0].grid(True)

        # Calculate chi-square
        chi_square = np.sum(weighted_residuals**2)
        dof = len(weighted_residuals) - (2 * n_lifetimes + 3)  # Degrees of freedom
        reduced_chi_square = chi_square / max(1, dof)

        # Add chi-square information to the plot
        info_text = f"Chi² = {reduced_chi_square:.3f}\n"
        for i in range(n_lifetimes):
            info_text += f"τ{i+1} = {self.lifetime_spectrum[2*i+1]:.3f} ns, A{i+1} = {self.lifetime_spectrum[2*i]:.3f}\n"

        # Add text box with fit information
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax[0].text(0.02, 0.98, info_text, transform=ax[0].transAxes, 
                 verticalalignment='top', bbox=props, fontsize=9)

        # Plot decay data, model, and IRF in the bottom subplot (log scale)
        ax[1].semilogy(self.time_axis, self.decay, 'b-', label='Data')
        ax[1].semilogy(self.time_axis[self.start:self.stop], self.model_decay[self.start:self.stop], 
                     'r-', label='Fit')

        # Scale IRF to the maximum of the decay data in the fit range
        if self.irf is not None:
            max_decay_in_range = np.max(self.decay[self.start:self.stop])
            max_irf = np.max(self.irf)
            ax[1].semilogy(self.time_axis, self.irf * max_decay_in_range / max_irf, 'g-', label='IRF')

        # Add vertical lines for analysis range
        ax[1].axvline(x=self.time_axis[self.start], color='k', linestyle='--', alpha=0.5)
        ax[1].axvline(x=self.time_axis[self.stop-1], color='k', linestyle='--', alpha=0.5)

        ax[1].set_xlabel('Time (ns)')
        ax[1].set_ylabel('Counts')
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    def to_json(self, filename: str = None) -> str:
        """
        Convert fit results to JSON.

        Parameters
        ----------
        filename : str
            Filename to save the JSON to

        Returns
        -------
        str
            JSON string
        """
        if self.fit_result is None:
            return "{}"

        # Create a copy of the fit result
        result = self.fit_result.copy()

        # Add additional information
        result['time_range'] = {
            'start': float(self.time_axis[self.start]),
            'stop': float(self.time_axis[self.stop-1]),
            'start_idx': int(self.start),
            'stop_idx': int(self.stop)
        }

        # Extract lifetimes and amplitudes for easier access
        n_lifetimes = result['n_lifetimes']
        lifetimes = []
        for i in range(n_lifetimes):
            lifetimes.append({
                'amplitude': float(self.lifetime_spectrum[2*i]),
                'lifetime': float(self.lifetime_spectrum[2*i+1])
            })
        result['lifetimes'] = lifetimes

        # Add model decay between start and stop
        if hasattr(self, 'model_decay') and self.model_decay is not None:
            # Include time axis and model decay between start and stop
            result['model'] = {
                'time': [float(t) for t in self.time_axis[self.start:self.stop]],
                'decay': [float(d) for d in self.model_decay[self.start:self.stop]]
            }

        # Convert to JSON
        json_str = json.dumps(result, indent=2)

        # Save to file
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(json_str)

        return json_str


def fit_lifetime(
        decay_file: str,
        irf_file: str,
        n_lifetimes: int = 1,
        skiprows: int = 0,
        delimiter: str = None,
        time_column: int = 0,
        counts_column: int = 1,
        output_file: str = None,
        plot_file: str = None,
        verbose: bool = False,
        config: dict = None,
        save_intermediate_results: bool = True,
        intermediate_results_base_filename: str = None
) -> dict:
    """
    Fit a lifetime to decay data.

    Parameters
    ----------
    decay_file : str
        Path to the decay data file
    irf_file : str
        Path to the IRF data file
    n_lifetimes : int
        Number of lifetimes to fit
    skiprows : int
        Number of rows to skip in the data files
    delimiter : str
        Delimiter used in the data files
    time_column : int
        Column index for the time data
    counts_column : int
        Column index for the counts data
    output_file : str
        Path to save the output JSON file
    plot_file : str
        Path to save the plot file
    verbose : bool
        Whether to print verbose output
    config : dict
        Configuration dictionary
    save_intermediate_results : bool
        Whether to save intermediate results to JSON files when finding optimal (default: True)
    intermediate_results_base_filename : str
        Base filename for intermediate results (defaults to output_file without extension if not provided)

    Returns
    -------
    dict
        Fit results

    Raises
    ------
    FileNotFoundError
        If the decay or IRF file is not found
    ValueError
        If the data in the files is invalid or if fitting fails
    Exception
        For any other errors that occur during fitting
    """
    try:
        # Load configuration
        if config is None:
            # Default configuration
            config = {
                'verbose': verbose,
                'estimate_background_parameter': {
                    'enabled': True,
                    'initial_irf_background': 0.0,
                    'fit_irf_background': True,
                    'average_window': 10
                },
                'analysis_range_parameter': {
                    'count_threshold': 10.0,
                    'area': 0.999,
                    'start_at_peak': False,
                    'start_fraction': 0.1,
                    'skip_first_channels': 0,
                    'skip_last_channels': 0
                },
                'estimate_irf_shift_parameters': {
                    'enabled': True,
                    'apply_shift': True,
                    'irf_time_shift_scan_range': (-8.0, 8.0),
                    'irf_time_shift_scan_n_steps': 20
                },
                'lifetime_fit_parameter': {
                    'find_optimal': False,
                    'maximum_number_of_lifetimes': 6,
                    'prob_threshold': 0.68,
                    'selection_mode': 'lower',
                    'plot_probabilities': True,
                    'plot_weighted_residuals': True,
                    'randomize_initial_values': {
                        'enabled': False,
                        'min_lifetime': 0.5,
                        'max_lifetime': 5.0,
                        'amplitude_variation': 0.5
                    }
                },
                'pile_up_correction': {
                    'enabled': False,
                    'rep_rate': 80.0,
                    'dead_time': 85.0,
                    'measurement_time': 60.0
                },
                'plot_resulting_fit': True
            }
        else:
            # Merge provided configuration with default settings
            default_config = {
                'verbose': verbose,
                'estimate_background_parameter': {
                    'enabled': True,
                    'initial_irf_background': 0.0,
                    'fit_irf_background': True,
                    'average_window': 10
                },
                'analysis_range_parameter': {
                    'count_threshold': 10.0,
                    'area': 0.999,
                    'start_at_peak': False,
                    'start_fraction': 0.1,
                    'skip_first_channels': 0,
                    'skip_last_channels': 0
                },
                'estimate_irf_shift_parameters': {
                    'enabled': True,
                    'apply_shift': True,
                    'irf_time_shift_scan_range': (-8.0, 8.0),
                    'irf_time_shift_scan_n_steps': 20
                },
                'lifetime_fit_parameter': {
                    'find_optimal': False,
                    'maximum_number_of_lifetimes': 6,
                    'prob_threshold': 0.68,
                    'selection_mode': 'lower',
                    'plot_probabilities': True,
                    'plot_weighted_residuals': True,
                    'randomize_initial_values': {
                        'enabled': False,
                        'min_lifetime': 0.5,
                        'max_lifetime': 5.0,
                        'amplitude_variation': 0.5
                    }
                },
                'pile_up_correction': {
                    'enabled': False,
                    'rep_rate': 80.0,
                    'dead_time': 85.0,
                    'measurement_time': 60.0
                },
                'plot_resulting_fit': True
            }
            # Update default config with provided config (provided config takes precedence)
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        update_dict(d[k], v)
                    else:
                        d[k] = v
            update_dict(default_config, config)
            config = default_config

        # Override verbose with the value from the configuration if provided
        if 'verbose' in config:
            verbose = config['verbose']

        # Load decay data
        if verbose:
            print(f"Loading decay data from {decay_file}")

        try:
            decay_data = np.genfromtxt(
                decay_file,
                delimiter=delimiter,
                skip_header=skiprows,
                usecols=[time_column, counts_column]
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Decay file not found: {decay_file}")
        except Exception as e:
            raise ValueError(f"Error loading decay data: {str(e)}")

        if decay_data.size == 0:
            raise ValueError("Decay data is empty")

        time_axis = decay_data[:, 0]
        decay = decay_data[:, 1]

        # Load IRF data
        if verbose:
            print(f"Loading IRF data from {irf_file}")

        try:
            irf_data = np.genfromtxt(
                irf_file,
                delimiter=delimiter,
                skip_header=skiprows,
                usecols=[time_column, counts_column]
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"IRF file not found: {irf_file}")
        except Exception as e:
            raise ValueError(f"Error loading IRF data: {str(e)}")

        if irf_data.size == 0:
            raise ValueError("IRF data is empty")

        irf = irf_data[:, 1]

        # Create decay object
        decay_obj = Decay(decay=decay, irf=irf, time_axis=time_axis)

        # Set pile-up correction parameters
        pile_up_params = config.get('pile_up_correction', {})
        decay_obj.correct_pile_up = pile_up_params.get('enabled', False)
        decay_obj.rep_rate = pile_up_params.get('rep_rate', 80.0)
        decay_obj.dead_time = pile_up_params.get('dead_time', 85.0)
        decay_obj.measurement_time = pile_up_params.get('measurement_time', 60.0)

        # Set initial IRF background from configuration
        bg_params = config.get('estimate_background_parameter', {})
        decay_obj.irf_background = bg_params.get('initial_irf_background', 0.0)

        # Set analysis range
        analysis_range_params = config.get('analysis_range_parameter', {})
        decay_obj.get_analysis_range(
            count_threshold=analysis_range_params.get('count_threshold', 10.0),
            area=analysis_range_params.get('area', 0.999),
            start_at_peak=analysis_range_params.get('start_at_peak', False),
            start_fraction=analysis_range_params.get('start_fraction', 0.1),
            skip_first_channels=analysis_range_params.get('skip_first_channels', 0),
            skip_last_channels=analysis_range_params.get('skip_last_channels', 0),
            verbose=verbose
        )

        # Estimate background
        if bg_params.get('enabled', True):
            decay_obj.estimate_background(
                average_window=bg_params.get('average_window', 10),
                verbose=verbose
            )
        elif verbose:
            print("Background estimation disabled")

        # Get IRF shift parameters
        irf_shift_params = config.get('estimate_irf_shift_parameters', {})

        # Set whether to apply IRF shift
        decay_obj.apply_irf_shift = irf_shift_params.get('apply_shift', True)
        if not decay_obj.apply_irf_shift and verbose:
            print("IRF shift application disabled, using original IRF")

        # Estimate IRF shift if enabled
        if irf_shift_params.get('enabled', True):
            decay_obj.estimate_irf_shift(
                irf_time_shift_scan_range=irf_shift_params.get('irf_time_shift_scan_range', (-8.0, 8.0)),
                irf_time_shift_scan_n_steps=irf_shift_params.get('irf_time_shift_scan_n_steps', 20),
                verbose=verbose
            )
        elif verbose:
            print("IRF shift estimation disabled")

        # Get fitting parameters
        fit_params = config.get('lifetime_fit_parameter', {})

        # Check if we should find the optimal number of lifetimes
        find_optimal = fit_params.get('find_optimal', False)
        if find_optimal and verbose:
            print(f"Finding optimal number of lifetimes (max={fit_params.get('maximum_number_of_lifetimes', 6)})")
        elif verbose:
            # CLI parameter should always override the default lifetime settings
            print(f"Using n_lifetimes={n_lifetimes} from CLI parameter")

        # Get randomization parameters
        randomize_params = fit_params.get('randomize_initial_values', {})

        # Create fixed parameters list
        # For n_lifetimes, we have 2*n_lifetimes parameters for the lifetime spectrum
        # Plus 3 more parameters: irf_shift, irf_background, decay_background
        fixed = [False] * (2 * n_lifetimes + 3)

        # Set IRF background to fixed if not fitting
        if not bg_params.get('fit_irf_background', True):
            # IRF background is the second-to-last parameter
            fixed[-2] = True
            if verbose:
                print("IRF background fixed at:", decay_obj.irf_background)

        # Fit
        fit_result = decay_obj.fit(
            n_lifetimes=n_lifetimes,
            fixed=fixed,
            verbose=verbose,
            randomize_initial_values=randomize_params.get('enabled', False),
            min_lifetime=randomize_params.get('min_lifetime', 0.5),
            max_lifetime=randomize_params.get('max_lifetime', 5.0),
            amplitude_variation=randomize_params.get('amplitude_variation', 0.5),
            find_optimal=find_optimal,
            maximum_number_of_lifetimes=fit_params.get('maximum_number_of_lifetimes', 6),
            prob_threshold=fit_params.get('prob_threshold', 0.68),
            plot_probabilities=fit_params.get('plot_probabilities', True),
            plot_weighted_residuals=fit_params.get('plot_weighted_residuals', True),
            selection_mode=fit_params.get('selection_mode', 'lower'),
            save_intermediate_results=save_intermediate_results,
            intermediate_results_base_filename=intermediate_results_base_filename
        )

        # Plot
        if plot_file is not None:
            try:
                # Save plot to file
                decay_obj.plot(filename=plot_file)
            except Exception as e:
                if verbose:
                    print(f"Warning: Error creating plot: {str(e)}")

        # Save results
        if output_file is not None:
            try:
                decay_obj.to_json(filename=output_file)
            except Exception as e:
                if verbose:
                    print(f"Warning: Error saving results to file: {str(e)}")

        return fit_result
    except Exception as e:
        # Re-raise the exception to be caught by the caller
        raise
