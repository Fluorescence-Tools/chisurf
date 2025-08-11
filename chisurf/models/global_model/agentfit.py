from __future__ import annotations
from chisurf import typing

import numpy as np
import random
import enum
import collections
import warnings
from typing import TYPE_CHECKING

import chisurf.decorators
import chisurf.parameter
import chisurf.plots
from chisurf import logging

if TYPE_CHECKING:
    from chisurf.fitting.fit import Fit

# Conditionally import scikit-learn
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.exceptions import ConvergenceWarning
    SKLEARN_AVAILABLE = True
    # Suppress sklearn warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. AgentFitModel will use basic optimization only.")


from chisurf.curve import Curve
from chisurf.models import model
from chisurf.fitting.parameter import GlobalFittingParameter
from chisurf.models.global_model.globalfit import GlobalFitModel


class AgentState(enum.Enum):
    """
    States for the agent-based fitter state machine.
    """
    EXPLORE = 0    # Explore parameter space broadly
    EXPLOIT = 1    # Focus on parameters that have shown improvement
    REFINE = 2     # Fine-tune the best parameters
    LOCAL_FIT = 3  # Perform local fitting on individual fits
    GLOBAL_FIT = 4 # Perform global fitting
    ML_PREDICT = 5 # Use machine learning to predict parameter changes


class AgentFitModel(GlobalFitModel):
    """
    Agent-based fitter for chisurf that operates on fits assigned to the agent in a global fit.
    The agent can fit individual fits and global fits, optimizing the global chi-squared value.
    It can change parameter values and will move back if values are too far out of bounds.

    This implementation uses a state machine approach to make the agent smarter:
    - Explores parameter space intelligently
    - Prioritizes parameters that have more impact on the chi-squared value
    - Uses adaptive step sizes based on parameter sensitivity
    - Schedules individual and global fits based on optimization progress

    Enhanced with machine learning capabilities (when scikit-learn is available):
    - Learns from parameter change history to predict which parameters will have the most impact
    - Uses random forest models to predict optimal step sizes for each parameter
    - Adaptively switches between heuristic-based and ML-based optimization strategies
    - Becomes smarter over time as it collects more training data
    """
    name = "Agent-based Fit"

    def __init__(
            self,
            fit: Fit,
            fits: typing.List[Fit] = None,
            max_iterations: int = None,
            step_size: float = None,
            cooling_rate: float = None,
            tolerance: float = None,
            explore_exploit_ratio: float = None,
            sensitivity_memory: int = None,
            fit_frequency: int = None,
            settings_file: str = None,
            *args,
            **kwargs
    ):
        """
        Initialize the agent-based fitter.

        Args:
            fit: The global fit object
            fits: List of individual fits to include
            max_iterations: Maximum number of iterations for the agent
            step_size: Initial step size for parameter changes
            cooling_rate: Rate at which step size decreases
            tolerance: Convergence tolerance for chi-squared improvement
            explore_exploit_ratio: Ratio of exploration vs exploitation (0-1)
            sensitivity_memory: Number of parameter changes to remember for sensitivity analysis
            fit_frequency: How often to perform a fit (every N iterations)
            settings_file: Path to the YAML settings file
        """
        super().__init__(fit, fits, *args, **kwargs)

        # Default settings file path
        if settings_file is None:
            import os
            from chisurf.settings.path_utils import get_path
            # Use the user settings path (~/.chisurf/agentfit_settings.yaml)
            settings_file = os.path.join(get_path('settings'), "agentfit_settings.yaml")
            # If the file doesn't exist in the user settings path, it will be copied there
            # when ChiSurf starts via the copy_settings_to_user_folder() function
        self.settings_file = settings_file

        # Load settings from YAML file if it exists
        settings = self._load_settings()

        # Use provided parameters if specified, otherwise use settings from file
        self.max_iterations = max_iterations if max_iterations is not None else settings.get('max_iterations', 100)
        self.step_size = step_size if step_size is not None else settings.get('step_size', 0.1)
        self.cooling_rate = cooling_rate if cooling_rate is not None else settings.get('cooling_rate', 0.95)
        self.tolerance = tolerance if tolerance is not None else settings.get('tolerance', 1e-6)
        self.explore_exploit_ratio = explore_exploit_ratio if explore_exploit_ratio is not None else settings.get('explore_exploit_ratio', 0.3)
        self.sensitivity_memory = sensitivity_memory if sensitivity_memory is not None else settings.get('sensitivity_memory', 10)
        self.fit_frequency = fit_frequency if fit_frequency is not None else settings.get('fit_frequency', 5)

        # State machine initialization
        self.current_state = AgentState.EXPLORE
        self.state_transitions = 0

        # Tracking variables
        self.current_iteration = 0
        self.best_chi2 = float('inf')
        self.best_parameters = []
        self.action_history = []
        self.chi2_history = []
        self.wres_history = []

        # Parameter performance tracking
        self.parameter_sensitivities = {}  # Parameter name -> sensitivity score
        self.parameter_improvements = {}   # Parameter name -> recent improvements
        self.parameter_step_sizes = {}     # Parameter name -> adaptive step size
        self.no_improvement_count = 0      # Counter for iterations without improvement

        # Machine learning enhancements
        self.use_ml = SKLEARN_AVAILABLE and settings.get('use_ml', True)
        self.min_samples_for_ml = settings.get('min_samples_for_ml', 20)  # Minimum samples before ML kicks in
        self.ml_prediction_frequency = settings.get('ml_prediction_frequency', 10)  # How often to use ML predictions

        # ML models (initialized when enough data is available)
        self.parameter_impact_model = None  # Predicts parameter impact on chi2
        self.step_size_model = None  # Predicts optimal step size
        self.ml_feature_data = []  # Stores feature data for training
        self.ml_target_data = []  # Stores target data for training

        if self.use_ml:
            logging.info("Machine learning enhancements enabled for AgentFitModel")
        else:
            if not SKLEARN_AVAILABLE:
                logging.warning("Machine learning disabled: scikit-learn not available")
            else:
                logging.info("Machine learning disabled by configuration")

    def perform_fit_on_individual(self, fit_index):
        """
        Perform fitting on an individual fit.

        Args:
            fit_index: Index of the fit to perform fitting on

        Returns:
            Dictionary containing the fitting action details
        """
        if fit_index < 0 or fit_index >= len(self.fits):
            logging.error(f"Invalid fit index: {fit_index}")
            return None

        fit = self.fits[fit_index]
        logging.info(f"Performing fit on individual fit {fit_index}: {fit.name}")

        # Store original chi2
        old_chi2 = fit.chi2
        old_global_chi2 = self.calculate_global_chi2()

        # Perform the fit
        try:
            fit.run()
            new_chi2 = fit.chi2
            new_global_chi2 = self.calculate_global_chi2()

            # Create action record
            action = {
                'iteration': self.current_iteration,
                'action_type': 'individual_fit',
                'fit_index': fit_index,
                'fit_name': fit.name,
                'old_chi2': old_chi2,
                'new_chi2': new_chi2,
                'old_global_chi2': old_global_chi2,
                'new_global_chi2': new_global_chi2,
                'accepted': new_global_chi2 < old_global_chi2
            }

            # If the global chi2 got worse, revert the fit
            if new_global_chi2 >= old_global_chi2:
                logging.info(f"  REJECTED: Global chi2 did not improve ({old_global_chi2} vs {new_global_chi2}), reverting fit")
                # Revert by loading the previous result
                if len(fit.results) > 1:
                    fit.previous_result()
            else:
                logging.info(f"  ACCEPTED: Global chi2 improved from {old_global_chi2} to {new_global_chi2}")
                # Update best parameters and chi2
                self.best_parameters = self.get_current_parameters()
                self.best_chi2 = new_global_chi2

            return action

        except Exception as e:
            logging.error(f"Error performing fit on individual fit {fit_index}: {str(e)}")
            return None

    def perform_global_fit(self):
        """
        Perform fitting on the global fit.

        Returns:
            Dictionary containing the fitting action details
        """
        logging.info("Performing global fit")

        # Store original chi2
        old_chi2 = self.calculate_global_chi2()

        # Perform the fit
        try:
            # Use the FitGroup's run method to perform the global fit
            self.fit.run(local_first=False)

            new_chi2 = self.calculate_global_chi2()

            # Create action record
            action = {
                'iteration': self.current_iteration,
                'action_type': 'global_fit',
                'old_chi2': old_chi2,
                'new_chi2': new_chi2,
                'accepted': new_chi2 < old_chi2
            }

            # If the chi2 got worse, revert the fit
            if new_chi2 >= old_chi2:
                logging.info(f"  REJECTED: Chi2 did not improve ({old_chi2} vs {new_chi2}), reverting global fit")
                # Revert by loading the previous result
                if len(self.fit.results) > 1:
                    self.fit.previous_result()
            else:
                logging.info(f"  ACCEPTED: Chi2 improved from {old_chi2} to {new_chi2}")
                # Update best parameters and chi2
                self.best_parameters = self.get_current_parameters()
                self.best_chi2 = new_chi2

            return action

        except Exception as e:
            logging.error(f"Error performing global fit: {str(e)}")
            return None

    def run_agent(self):
        """
        Run the agent-based optimization process using a state machine approach.
        This implementation is smarter than the original random approach:
        - Always starts with a fit (individual and global) to establish a good baseline
        - Uses a state machine to guide the optimization process
        - Prioritizes parameters based on their sensitivity to chi-squared
        - Uses adaptive step sizes for each parameter
        - Intelligently schedules individual and global fits
        - Performs fits at regular intervals based on the fit_frequency parameter

        Machine learning enhancements (when scikit-learn is available):
        - Collects training data from parameter changes and their effects
        - Trains random forest models to predict parameter impact and optimal step sizes
        - Periodically enters ML_PREDICT state to use machine learning predictions
        - Adapts to the specific optimization problem by learning from past iterations
        """
        logging.info("AgentFitModel.run_agent started with state machine approach")
        self.current_iteration = 0
        self.action_history = []
        self.chi2_history = []
        self.wres_history = []
        self.parameter_sensitivities = {}
        self.parameter_improvements = {}
        self.parameter_step_sizes = {}
        self.no_improvement_count = 0
        self.current_state = AgentState.EXPLORE
        self.state_transitions = 0

        logging.info(f"Histories reset, starting optimization with {len(self.fits)} fits")
        logging.info(f"Initial state: {self.current_state.name}")
        logging.info(f"Agent settings: max_iterations={self.max_iterations}, "
                     f"step_size={self.step_size}, "
                     f"cooling_rate={self.cooling_rate}, "
                     f"tolerance={self.tolerance}, "
                     f"explore_exploit_ratio={self.explore_exploit_ratio}, "
                     f"sensitivity_memory={self.sensitivity_memory}, "
                     f"fit_frequency={self.fit_frequency}")

        # Initialize with current parameters
        self.best_parameters = self.get_current_parameters()
        logging.info(f"Initial parameters: {self.best_parameters}")
        self.best_chi2 = self.calculate_global_chi2()
        logging.info(f"Initial chi2: {self.best_chi2}")

        self.chi2_history.append(self.best_chi2)
        self.wres_history.append(self.get_all_wres())
        logging.info("Initial state recorded in histories")

        # Always start with a fit
        logging.info("Starting with an initial fit as requested")
        if len(self.fits) > 0:
            # Start with an individual fit on the worst fit
            fit_chi2s = [(i, f.chi2) for i, f in enumerate(self.fits)]
            fit_chi2s.sort(key=lambda x: x[1], reverse=True)  # Sort by chi2 (highest first)
            worst_fit_index = fit_chi2s[0][0]

            logging.info(f"Performing initial fit on worst fit (index {worst_fit_index})")
            fit_action = self.perform_fit_on_individual(worst_fit_index)
            if fit_action:
                self.action_history.append(fit_action)
                if fit_action['accepted']:
                    logging.info("Initial individual fit improved chi2")
                else:
                    logging.info("Initial individual fit did not improve chi2")

        # Also try a global fit
        logging.info("Performing initial global fit")
        global_fit_action = self.perform_global_fit()
        if global_fit_action:
            self.action_history.append(global_fit_action)
            if global_fit_action['accepted']:
                logging.info("Initial global fit improved chi2")
            else:
                logging.info("Initial global fit did not improve chi2")

        # Update histories after initial fits
        self.chi2_history.append(self.best_chi2)
        self.wres_history.append(self.get_all_wres())

        current_step_size = self.step_size

        # Main optimization loop
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            logging.info(f"Starting iteration {self.current_iteration}/{self.max_iterations}, current chi2: {self.best_chi2}")
            logging.info(f"Current state: {self.current_state.name}")

            # Handle special states first
            if self.current_state == AgentState.LOCAL_FIT:
                # Perform local fitting on a fit
                if len(self.fits) > 0:
                    # Select a fit to optimize - prioritize fits with worse chi2
                    fit_chi2s = [(i, f.chi2) for i, f in enumerate(self.fits)]
                    fit_chi2s.sort(key=lambda x: x[1], reverse=True)  # Sort by chi2 (highest first)

                    # Try to fit the worst fits first
                    improved = False
                    for fit_index, _ in fit_chi2s[:max(1, len(fit_chi2s) // 2)]:  # Try top half of worst fits
                        fit_action = self.perform_fit_on_individual(fit_index)
                        if fit_action and fit_action['accepted']:
                            self.action_history.append(fit_action)
                            logging.info(f"Individual fit action recorded for fit {fit_index}")
                            improved = True
                            break  # Stop after first successful fit

                    # Update state based on whether we improved
                    self.update_state(improved)
                else:
                    # No fits to optimize, move to next state
                    self.update_state(False)

                # Record chi2 and weighted residuals for this iteration
                self.chi2_history.append(self.best_chi2)
                self.wres_history.append(self.get_all_wres())
                continue  # Skip to next iteration

            elif self.current_state == AgentState.GLOBAL_FIT:
                # Perform global fitting
                global_fit_action = self.perform_global_fit()
                if global_fit_action:
                    self.action_history.append(global_fit_action)
                    logging.info("Global fit action recorded")

                    # Update state based on whether we improved
                    self.update_state(global_fit_action['accepted'])
                else:
                    # Failed to perform global fit, move to next state
                    self.update_state(False)

                # Record chi2 and weighted residuals for this iteration
                self.chi2_history.append(self.best_chi2)
                self.wres_history.append(self.get_all_wres())
                continue  # Skip to next iteration

            # For EXPLORE, EXPLOIT, and REFINE states, optimize parameters
            # Select parameters to optimize based on current state
            param_indices = self.select_parameters_to_optimize()

            # Try to improve selected parameters
            iteration_improved = False

            for param_idx in param_indices:
                param = self.parameters[param_idx]
                if param.fixed or param.is_linked:
                    continue

                # Store original value
                original_value = param.value
                logging.debug(f"  Optimizing parameter {param.name} (current value: {original_value})")

                # Get adaptive step size for this parameter
                if self.current_state == AgentState.ML_PREDICT and self.use_ml and SKLEARN_AVAILABLE and self.step_size_model is not None:
                    # In ML_PREDICT state, use ML to predict optimal step size
                    try:
                        param_step_size = self.predict_optimal_step_size(param_idx, current_step_size)
                        logging.debug(f"  Using ML-predicted step size for {param.name}: {param_step_size}")
                    except Exception as e:
                        logging.error(f"Error predicting step size with ML: {str(e)}")
                        param_step_size = self.get_step_size_for_parameter(param.name, current_step_size)
                else:
                    param_step_size = self.get_step_size_for_parameter(param.name, current_step_size)

                # Determine step direction and size based on state
                if self.current_state == AgentState.EXPLORE:
                    # In explore state, use larger random steps
                    step = (random.random() * 2 - 1) * param_step_size * abs(original_value)
                elif self.current_state == AgentState.EXPLOIT:
                    # In exploit state, use more controlled steps
                    # Use smaller random steps with bias toward direction that improved before
                    direction = 1 if random.random() < 0.7 else -1  # Bias toward positive direction
                    step = direction * random.random() * param_step_size * abs(original_value)
                elif self.current_state == AgentState.ML_PREDICT:
                    # In ML_PREDICT state, use more precise steps based on ML predictions
                    # Use smaller random component to fine-tune the ML prediction
                    direction = 1 if random.random() < 0.8 else -1  # Stronger bias toward positive direction
                    random_factor = 0.5 + random.random() * 0.5  # 0.5-1.0 random factor
                    step = direction * random_factor * param_step_size * abs(original_value)
                else:  # REFINE state
                    # In refine state, use very small steps for fine-tuning
                    step = (random.random() * 2 - 1) * param_step_size * 0.5 * abs(original_value)

                new_value = original_value + step
                logging.debug(f"  Trying new value: {new_value} (step: {step}, step_size: {param_step_size})")

                # Check bounds
                if param.bounds_on:
                    lb, ub = param.bounds
                    logging.debug(f"  Checking bounds: {lb} <= {new_value} <= {ub}")
                    if (lb is not None and new_value < lb) or (ub is not None and new_value > ub):
                        # Value out of bounds, try a smaller step in the opposite direction
                        step = -step * 0.5
                        new_value = original_value + step
                        logging.debug(f"  Value out of bounds, trying opposite direction: {new_value} (step: {step})")

                        # If still out of bounds, skip this parameter
                        if param.bounds_on:
                            if (lb is not None and new_value < lb) or (ub is not None and new_value > ub):
                                logging.debug(f"  Still out of bounds, skipping parameter {param.name}")
                                continue

                # Apply the new value
                param.value = new_value
                logging.debug(f"  Applying new value: {new_value}")
                self.update_model()

                # Calculate new chi2
                new_chi2 = self.calculate_global_chi2()
                logging.debug(f"  New chi2: {new_chi2} (previous: {self.best_chi2})")

                # Update parameter sensitivity
                self.update_parameter_sensitivity(param.name, self.best_chi2, new_chi2)

                # Record the action
                action = {
                    'iteration': self.current_iteration,
                    'parameter': param.name,
                    'old_value': original_value,
                    'new_value': new_value,
                    'old_chi2': self.best_chi2,
                    'new_chi2': new_chi2,
                    'accepted': False
                }
                logging.debug(f"  Recording action for parameter {param.name}")

                # Accept if improved
                accepted = new_chi2 < self.best_chi2
                if accepted:
                    self.best_chi2 = new_chi2
                    self.best_parameters = self.get_current_parameters()
                    iteration_improved = True
                    action['accepted'] = True
                    logging.info(f"  ACCEPTED: Chi2 improved from {action['old_chi2']} to {new_chi2}")
                else:
                    # Revert to original value
                    param.value = original_value
                    self.update_model()
                    logging.debug(f"  REJECTED: Chi2 did not improve ({action['old_chi2']} vs {new_chi2}), reverting to {original_value}")

                self.action_history.append(action)

                # Collect training data for machine learning
                if self.use_ml:
                    self.collect_ml_training_data(
                        param_name=param.name,
                        old_value=original_value,
                        new_value=new_value,
                        old_chi2=self.best_chi2,
                        new_chi2=new_chi2,
                        accepted=accepted
                    )
                logging.debug(f"  Action recorded, history size: {len(self.action_history)}")

                # In REFINE state, stop after first improvement to avoid over-optimization
                if self.current_state == AgentState.REFINE and iteration_improved:
                    break

            # Record chi2 and weighted residuals for this iteration
            self.chi2_history.append(self.best_chi2)
            self.wres_history.append(self.get_all_wres())
            logging.info(f"Iteration {self.current_iteration} completed, best chi2: {self.best_chi2}")

            # Check if we should perform a fit based on fit_frequency
            if self.fit_frequency > 0 and self.current_iteration % self.fit_frequency == 0:
                logging.info(f"Performing scheduled fit at iteration {self.current_iteration} (fit_frequency={self.fit_frequency})")

                # Alternate between individual and global fits
                if self.current_iteration % (self.fit_frequency * 2) == 0:
                    # Perform global fit
                    logging.info("Performing scheduled global fit")
                    global_fit_action = self.perform_global_fit()
                    if global_fit_action:
                        self.action_history.append(global_fit_action)
                        if global_fit_action['accepted']:
                            logging.info("Scheduled global fit improved chi2")
                            iteration_improved = True
                        else:
                            logging.info("Scheduled global fit did not improve chi2")
                else:
                    # Perform individual fit on worst fit
                    if len(self.fits) > 0:
                        fit_chi2s = [(i, f.chi2) for i, f in enumerate(self.fits)]
                        fit_chi2s.sort(key=lambda x: x[1], reverse=True)
                        worst_fit_index = fit_chi2s[0][0]

                        logging.info(f"Performing scheduled individual fit on worst fit (index {worst_fit_index})")
                        fit_action = self.perform_fit_on_individual(worst_fit_index)
                        if fit_action:
                            self.action_history.append(fit_action)
                            if fit_action['accepted']:
                                logging.info("Scheduled individual fit improved chi2")
                                iteration_improved = True
                            else:
                                logging.info("Scheduled individual fit did not improve chi2")

            # Update state machine based on improvement
            self.update_state(iteration_improved)

            # Cool down the global step size
            current_step_size *= self.cooling_rate
            logging.debug(f"Cooling global step size to {current_step_size}")

            # Check for convergence
            if len(self.chi2_history) > 1:
                improvement = abs(self.chi2_history[-2] - self.chi2_history[-1])
                logging.debug(f"Checking convergence: improvement = {improvement}, tolerance = {self.tolerance}")
                if improvement < self.tolerance and self.current_state == AgentState.REFINE:
                    logging.info(f"Converged! Improvement {improvement} < tolerance {self.tolerance}")
                    break

            # Update any associated plots
            logging.debug("Updating plots")
            self.update_plots()

        # Ensure we're using the best parameters found
        logging.info(f"Optimization completed after {self.current_iteration} iterations")
        logging.info(f"Setting best parameters: {self.best_parameters}")
        self.set_parameters(self.best_parameters)
        self.update_model()
        logging.info(f"Final chi2: {self.best_chi2}")
        logging.info(f"State transitions: {self.state_transitions}")

        # Log parameter sensitivities
        if self.parameter_sensitivities:
            logging.info("Parameter sensitivities:")
            for param_name, sensitivity in sorted(self.parameter_sensitivities.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"  {param_name}: {sensitivity:.6g}")

        return self.best_chi2, self.action_history

    def get_current_parameters(self):
        """Get the current parameter values."""
        params = [param.value for param in self.parameters]
        logging.debug(f"get_current_parameters: {len(params)} parameters retrieved")
        return params

    def set_parameters(self, values):
        """Set parameter values."""
        logging.debug(f"set_parameters: Setting {len(values)} parameter values")
        for param, value in zip(self.parameters, values):
            param.value = value
            logging.debug(f"  Set {param.name} = {value}")

    def calculate_global_chi2(self):
        """Calculate the global chi-squared value."""
        chi2 = 0
        logging.debug(f"calculate_global_chi2: Calculating chi2 for {len(self.fits)} fits")
        for i, fit in enumerate(self.fits):
            fit_chi2 = fit.chi2
            chi2 += fit_chi2
            logging.debug(f"  Fit {i}: chi2 = {fit_chi2}")
        logging.debug(f"  Total chi2 = {chi2}")
        return chi2

    def get_all_wres(self):
        """Get weighted residuals for all fits."""
        logging.debug(f"get_all_wres: Getting weighted residuals for {len(self.fits)} fits")
        wres_list = []
        for i, fit in enumerate(self.fits):
            wres = fit.model.weighted_residuals
            wres_list.append(wres)
            logging.debug(f"  Fit {i}: {len(wres)} weighted residuals")
        return wres_list

    def update_plots(self):
        """Update any associated plots."""
        logging.debug("update_plots: Updating associated plots")
        if hasattr(self.fit, 'plots'):
            logging.debug(f"  Found {len(self.fit.plots)} plots")
            for i, plot in enumerate(self.fit.plots):
                logging.debug(f"  Checking plot {i}: {plot.__class__.__name__}")
                if isinstance(plot, chisurf.plots.agent_fit.AgentFitPlot):
                    logging.debug(f"  Updating AgentFitPlot")
                    plot.update()
        else:
            logging.debug("  No plots found")

    def update_parameter_sensitivity(self, param_name, old_chi2, new_chi2):
        """
        Update the sensitivity score for a parameter based on its impact on chi-squared.

        Args:
            param_name: Name of the parameter
            old_chi2: Chi-squared value before parameter change
            new_chi2: Chi-squared value after parameter change
        """
        if param_name not in self.parameter_improvements:
            self.parameter_improvements[param_name] = collections.deque(maxlen=self.sensitivity_memory)

        # Calculate improvement (negative means chi2 decreased, which is good)
        improvement = new_chi2 - old_chi2
        self.parameter_improvements[param_name].append(improvement)

        # Calculate sensitivity as the average improvement
        if len(self.parameter_improvements[param_name]) > 0:
            avg_improvement = sum(self.parameter_improvements[param_name]) / len(self.parameter_improvements[param_name])
            # Negative avg_improvement means parameter tends to improve chi2
            # We want higher sensitivity for parameters that improve chi2 more
            self.parameter_sensitivities[param_name] = -avg_improvement
        else:
            self.parameter_sensitivities[param_name] = 0.0

        # Update adaptive step size based on sensitivity
        if param_name not in self.parameter_step_sizes:
            self.parameter_step_sizes[param_name] = self.step_size
        else:
            # If parameter consistently improves chi2, increase step size
            # If parameter consistently worsens chi2, decrease step size
            if avg_improvement < 0:  # Improvement
                self.parameter_step_sizes[param_name] *= 1.2  # Increase step size
            else:  # No improvement or worsening
                self.parameter_step_sizes[param_name] *= 0.8  # Decrease step size

            # Ensure step size stays within reasonable bounds
            self.parameter_step_sizes[param_name] = max(0.001 * self.step_size, 
                                                       min(5.0 * self.step_size, 
                                                           self.parameter_step_sizes[param_name]))

    def select_parameters_to_optimize(self):
        """
        Select parameters to optimize based on the current state and parameter sensitivities.
        When in ML_PREDICT state, uses machine learning to predict parameter impact.

        Returns:
            List of parameter indices to optimize
        """
        params = self.parameters
        param_indices = list(range(len(params)))

        # Filter out fixed and linked parameters
        param_indices = [i for i in param_indices if not (params[i].fixed or params[i].is_linked)]

        if not param_indices:
            return []

        if self.current_state == AgentState.EXPLORE:
            # In explore state, select parameters randomly
            random.shuffle(param_indices)
            return param_indices

        elif self.current_state == AgentState.EXPLOIT:
            # In exploit state, prioritize parameters with higher sensitivity
            if not self.parameter_sensitivities:
                return param_indices  # If no sensitivities yet, return all parameters

            # Sort parameters by sensitivity (highest first)
            param_names = [params[i].name for i in param_indices]
            sensitivities = [self.parameter_sensitivities.get(name, 0.0) for name in param_names]
            sorted_indices = [i for _, i in sorted(zip(sensitivities, param_indices), reverse=True)]

            # Return all parameters, but prioritize those with higher sensitivity
            return sorted_indices

        elif self.current_state == AgentState.REFINE:
            # In refine state, focus only on the most sensitive parameters
            if not self.parameter_sensitivities:
                return param_indices[:1]  # If no sensitivities yet, return just the first parameter

            # Sort parameters by sensitivity (highest first)
            param_names = [params[i].name for i in param_indices]
            sensitivities = [self.parameter_sensitivities.get(name, 0.0) for name in param_names]
            sorted_indices = [i for _, i in sorted(zip(sensitivities, param_indices), reverse=True)]

            # Return only the top 20% most sensitive parameters
            top_n = max(1, int(len(sorted_indices) * 0.2))
            return sorted_indices[:top_n]

        elif self.current_state == AgentState.ML_PREDICT:
            # In ML_PREDICT state, use machine learning to predict parameter impact
            if not self.use_ml or not SKLEARN_AVAILABLE or self.parameter_impact_model is None:
                # Fall back to sensitivity-based selection if ML not available
                logging.warning("ML_PREDICT state but ML not available, falling back to sensitivity-based selection")
                return self.select_parameters_to_optimize_by_sensitivity(param_indices)

            try:
                # Predict impact for each parameter
                impacts = [self.predict_parameter_impact(i) for i in param_indices]

                # Sort parameters by predicted impact (highest first)
                sorted_indices = [i for _, i in sorted(zip(impacts, param_indices), reverse=True)]

                # Log the predictions
                logging.info("ML parameter impact predictions:")
                for i, idx in enumerate(sorted_indices[:5]):  # Log top 5
                    param = params[idx]
                    logging.info(f"  {i+1}. {param.name}: {impacts[param_indices.index(idx)]:.6g}")

                # Return top parameters based on predicted impact
                top_n = max(1, int(len(sorted_indices) * 0.3))  # Use more parameters than in REFINE
                return sorted_indices[:top_n]

            except Exception as e:
                logging.error(f"Error in ML parameter selection: {str(e)}")
                # Fall back to sensitivity-based selection
                return self.select_parameters_to_optimize_by_sensitivity(param_indices)

        else:
            # For other states, return all parameters
            return param_indices

    def select_parameters_to_optimize_by_sensitivity(self, param_indices):
        """
        Helper method to select parameters based on sensitivity.
        Used as a fallback when ML is not available.

        Args:
            param_indices: List of parameter indices to consider

        Returns:
            Sorted list of parameter indices
        """
        params = self.parameters

        if not self.parameter_sensitivities:
            return param_indices  # If no sensitivities yet, return all parameters

        # Sort parameters by sensitivity (highest first)
        param_names = [params[i].name for i in param_indices]
        sensitivities = [self.parameter_sensitivities.get(name, 0.0) for name in param_names]
        sorted_indices = [i for _, i in sorted(zip(sensitivities, param_indices), reverse=True)]

        return sorted_indices

    def update_state(self, improved):
        """
        Update the state machine based on the current state and optimization progress.

        Args:
            improved: Whether the last iteration improved the chi-squared value
        """
        # Update no_improvement_count
        if improved:
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # State transitions
        if self.current_state == AgentState.EXPLORE:
            # After some iterations or if we've found improvements, switch to EXPLOIT
            if self.current_iteration > self.max_iterations * 0.2 or self.no_improvement_count > 5:
                self.current_state = AgentState.EXPLOIT
                self.state_transitions += 1
                logging.info(f"State transition: EXPLORE -> EXPLOIT (iteration {self.current_iteration})")

        elif self.current_state == AgentState.EXPLOIT:
            # If ML is available and we have enough data, periodically use ML predictions
            if (self.use_ml and SKLEARN_AVAILABLE and 
                len(self.ml_feature_data) >= self.min_samples_for_ml and
                self.current_iteration % self.ml_prediction_frequency == 0):
                self.current_state = AgentState.ML_PREDICT
                self.state_transitions += 1
                logging.info(f"State transition: EXPLOIT -> ML_PREDICT (iteration {self.current_iteration})")
            # If no improvement for a while, try LOCAL_FIT
            elif self.no_improvement_count > 3:
                self.current_state = AgentState.LOCAL_FIT
                self.state_transitions += 1
                logging.info(f"State transition: EXPLOIT -> LOCAL_FIT (iteration {self.current_iteration})")
            # If we're getting close to the end, switch to REFINE
            elif self.current_iteration > self.max_iterations * 0.7:
                self.current_state = AgentState.REFINE
                self.state_transitions += 1
                logging.info(f"State transition: EXPLOIT -> REFINE (iteration {self.current_iteration})")

        elif self.current_state == AgentState.REFINE:
            # If ML is available and we have enough data, periodically use ML predictions
            if (self.use_ml and SKLEARN_AVAILABLE and 
                len(self.ml_feature_data) >= self.min_samples_for_ml and
                self.current_iteration % self.ml_prediction_frequency == 0):
                self.current_state = AgentState.ML_PREDICT
                self.state_transitions += 1
                logging.info(f"State transition: REFINE -> ML_PREDICT (iteration {self.current_iteration})")
            # If no improvement for a while, try GLOBAL_FIT
            elif self.no_improvement_count > 3:
                self.current_state = AgentState.GLOBAL_FIT
                self.state_transitions += 1
                logging.info(f"State transition: REFINE -> GLOBAL_FIT (iteration {self.current_iteration})")

        elif self.current_state == AgentState.LOCAL_FIT:
            # After local fit, go back to EXPLOIT
            self.current_state = AgentState.EXPLOIT
            self.state_transitions += 1
            logging.info(f"State transition: LOCAL_FIT -> EXPLOIT (iteration {self.current_iteration})")

        elif self.current_state == AgentState.GLOBAL_FIT:
            # After global fit, go back to REFINE
            self.current_state = AgentState.REFINE
            self.state_transitions += 1
            logging.info(f"State transition: GLOBAL_FIT -> REFINE (iteration {self.current_iteration})")

        elif self.current_state == AgentState.ML_PREDICT:
            # After ML prediction, go back to previous state based on iteration progress
            if self.current_iteration > self.max_iterations * 0.7:
                self.current_state = AgentState.REFINE
                logging.info(f"State transition: ML_PREDICT -> REFINE (iteration {self.current_iteration})")
            else:
                self.current_state = AgentState.EXPLOIT
                logging.info(f"State transition: ML_PREDICT -> EXPLOIT (iteration {self.current_iteration})")
            self.state_transitions += 1

    def get_step_size_for_parameter(self, param_name, current_step_size):
        """
        Get the adaptive step size for a parameter.

        Args:
            param_name: Name of the parameter
            current_step_size: Current global step size

        Returns:
            Step size to use for this parameter
        """
        if param_name in self.parameter_step_sizes:
            return self.parameter_step_sizes[param_name]
        else:
            return current_step_size

    def _load_settings(self) -> dict:
        """
        Load settings from the YAML file.

        Returns:
            dict: The settings dictionary loaded from the YAML file, or an empty dict if the file doesn't exist.
        """
        import os
        import yaml

        settings = {}
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    settings = yaml.safe_load(f)
                logging.info(f"Loaded agent settings from {self.settings_file}")
            except Exception as e:
                logging.error(f"Error loading agent settings from {self.settings_file}: {str(e)}")
        else:
            logging.warning(f"Settings file {self.settings_file} not found, using default settings")

        return settings or {}

    def save_settings(self) -> None:
        """
        Save the current settings to the YAML file.
        Includes machine learning settings if scikit-learn is available.
        """
        import yaml

        settings = {
            'max_iterations': self.max_iterations,
            'step_size': self.step_size,
            'cooling_rate': self.cooling_rate,
            'tolerance': self.tolerance,
            'explore_exploit_ratio': self.explore_exploit_ratio,
            'sensitivity_memory': self.sensitivity_memory,
            'fit_frequency': self.fit_frequency,
            # Machine learning settings
            'use_ml': self.use_ml,
            'min_samples_for_ml': self.min_samples_for_ml,
            'ml_prediction_frequency': self.ml_prediction_frequency
        }

        try:
            with open(self.settings_file, 'w') as f:
                yaml.dump(settings, f, default_flow_style=False)
            logging.info(f"Saved agent settings to {self.settings_file}")
        except Exception as e:
            logging.error(f"Error saving agent settings to {self.settings_file}: {str(e)}")

    def update_model(self, **kwargs) -> None:
        """
        Override the update_model method from GlobalFitModel to always run without threads.
        This helps prevent crashes that can occur with threaded model updates.
        """
        logging.debug("AgentFitModel.update_model: Running without threads")
        # Always use the non-threaded approach
        for f in self.fits:
            f.model.update_model()

    def collect_ml_training_data(self, param_name, old_value, new_value, old_chi2, new_chi2, accepted):
        """
        Collect training data for machine learning models.

        Args:
            param_name: Name of the parameter that was changed
            old_value: Original parameter value
            new_value: New parameter value
            old_chi2: Chi-squared before parameter change
            new_chi2: Chi-squared after parameter change
            accepted: Whether the change was accepted

        Note:
            Includes preprocessing to handle infinite or extremely large values in parameter bounds
            and calculated metrics by capping them at ±1e30 and ensuring all values are finite.
            This prevents errors during model training with scikit-learn.
        """
        if not self.use_ml:
            return

        # Get parameter object
        param_idx = next((i for i, p in enumerate(self.parameters) if p.name == param_name), None)
        if param_idx is None:
            return

        param = self.parameters[param_idx]

        # Process bounds to avoid infinite values
        lower_bound = param.bounds[0] if param.bounds_on and param.bounds[0] is not None else -1e30
        upper_bound = param.bounds[1] if param.bounds_on and param.bounds[1] is not None else 1e30

        # Ensure bounds are finite
        if lower_bound == float('-inf') or lower_bound < -1e30:
            lower_bound = -1e30
        if upper_bound == float('inf') or upper_bound > 1e30:
            upper_bound = 1e30

        # Calculate value range safely
        if param.bounds_on and None not in param.bounds:
            value_range = min(1e30, max(-1e30, upper_bound - lower_bound))
        else:
            value_range = 1e30

        # Ensure relative change is finite
        try:
            relative_change = (new_value - old_value) / (abs(old_value) + 1e-10)
            if not np.isfinite(relative_change):
                relative_change = 0.0
        except:
            relative_change = 0.0

        # Extract features with safe values
        features = {
            'param_idx': param_idx,
            'current_iteration': self.current_iteration,
            'current_state': self.current_state.value,
            'old_value': old_value,
            'relative_change': relative_change,
            'old_chi2': old_chi2,
            'has_bounds': param.bounds_on,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'value_range': value_range,
            'sensitivity': self.parameter_sensitivities.get(param_name, 0.0),
            'no_improvement_count': self.no_improvement_count
        }

        # Extract targets with safe values
        chi2_change = new_chi2 - old_chi2
        if not np.isfinite(chi2_change):
            chi2_change = 0.0

        improvement = old_chi2 - new_chi2
        if not np.isfinite(improvement):
            improvement = 0.0

        targets = {
            'chi2_change': chi2_change,
            'accepted': 1 if accepted else 0,
            'improvement': improvement  # Positive means improvement
        }

        # Store data
        self.ml_feature_data.append(features)
        self.ml_target_data.append(targets)

        # Train models if we have enough data
        if len(self.ml_feature_data) >= self.min_samples_for_ml:
            self.train_ml_models()

    def train_ml_models(self):
        """
        Train machine learning models using collected data.

        This method includes preprocessing steps to handle infinite or extremely large values
        in the feature data, which can cause errors in scikit-learn models. Values are capped
        at ±1e30 to ensure they remain finite while still representing very large or small numbers.
        """
        if not self.use_ml or not SKLEARN_AVAILABLE:
            return

        if len(self.ml_feature_data) < self.min_samples_for_ml:
            logging.debug(f"Not enough data to train ML models: {len(self.ml_feature_data)}/{self.min_samples_for_ml}")
            return

        try:
            # Convert data to numpy arrays
            feature_names = ['param_idx', 'current_iteration', 'current_state', 'old_value', 
                            'relative_change', 'old_chi2', 'has_bounds', 'lower_bound', 
                            'upper_bound', 'value_range', 'sensitivity', 'no_improvement_count']

            # Preprocess feature data to handle infinite values
            processed_features = []
            for features in self.ml_feature_data:
                processed_feature = {}
                for name in feature_names:
                    value = features[name]
                    # Replace infinite values with large but finite numbers
                    if name in ['lower_bound', 'upper_bound', 'value_range']:
                        if value == float('inf') or value > 1e30:
                            value = 1e30
                        elif value == float('-inf') or value < -1e30:
                            value = -1e30
                    processed_feature[name] = value
                processed_features.append(processed_feature)

            # Create feature array with processed values
            X = np.array([[features[name] for name in feature_names] for features in processed_features])

            # Check for any remaining infinite or NaN values
            if not np.isfinite(X).all():
                logging.warning("Non-finite values found in feature data after preprocessing")
                # Replace any remaining non-finite values
                X = np.nan_to_num(X, nan=0.0, posinf=1e30, neginf=-1e30)

            # Target for parameter impact model (predicts improvement)
            y_impact = np.array([target['improvement'] for target in self.ml_target_data])
            y_impact = np.nan_to_num(y_impact, nan=0.0, posinf=1e30, neginf=-1e30)

            # Target for step size model (predicts whether change will be accepted)
            y_step = np.array([target['accepted'] for target in self.ml_target_data])
            y_step = np.nan_to_num(y_step, nan=0.0, posinf=1.0, neginf=0.0)

            # Train parameter impact model
            self.parameter_impact_model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=50, max_depth=5))
            ])
            self.parameter_impact_model.fit(X, y_impact)

            # Train step size model
            self.step_size_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestRegressor(n_estimators=50, max_depth=5))
            ])
            self.step_size_model.fit(X, y_step)

            logging.info(f"Trained ML models with {len(self.ml_feature_data)} samples")

        except Exception as e:
            logging.error(f"Error training ML models: {str(e)}")
            self.parameter_impact_model = None
            self.step_size_model = None

    def predict_parameter_impact(self, param_idx):
        """
        Predict the impact of changing a parameter on chi-squared.

        Args:
            param_idx: Index of the parameter

        Returns:
            Predicted impact score (higher means more likely to improve chi2)

        Note:
            Includes preprocessing to handle infinite or extremely large values in parameter bounds
            by capping them at ±1e30 to prevent errors in scikit-learn models.
        """
        if not self.use_ml or not SKLEARN_AVAILABLE or self.parameter_impact_model is None:
            # Fall back to sensitivity-based scoring
            param = self.parameters[param_idx]
            return self.parameter_sensitivities.get(param.name, 0.0)

        try:
            param = self.parameters[param_idx]

            # Process bounds to avoid infinite values
            lower_bound = param.bounds[0] if param.bounds_on and param.bounds[0] is not None else -1e30
            upper_bound = param.bounds[1] if param.bounds_on and param.bounds[1] is not None else 1e30

            # Ensure bounds are finite
            if lower_bound == float('-inf') or lower_bound < -1e30:
                lower_bound = -1e30
            if upper_bound == float('inf') or upper_bound > 1e30:
                upper_bound = 1e30

            # Calculate value range safely
            if param.bounds_on and None not in param.bounds:
                value_range = min(1e30, max(-1e30, upper_bound - lower_bound))
            else:
                value_range = 1e30

            # Create feature vector with safe values
            features = np.array([[
                param_idx,
                self.current_iteration,
                self.current_state.value,
                param.value,
                0.1,  # Default relative change
                self.best_chi2,
                param.bounds_on,
                lower_bound,
                upper_bound,
                value_range,
                self.parameter_sensitivities.get(param.name, 0.0),
                self.no_improvement_count
            ]])

            # Ensure all values are finite
            features = np.nan_to_num(features, nan=0.0, posinf=1e30, neginf=-1e30)

            # Predict impact
            impact = self.parameter_impact_model.predict(features)[0]
            return impact

        except Exception as e:
            logging.error(f"Error predicting parameter impact: {str(e)}")
            # Fall back to sensitivity-based scoring
            param = self.parameters[param_idx]
            return self.parameter_sensitivities.get(param.name, 0.0)

    def predict_optimal_step_size(self, param_idx, base_step_size):
        """
        Predict the optimal step size for a parameter.

        Args:
            param_idx: Index of the parameter
            base_step_size: Base step size to adjust

        Returns:
            Predicted optimal step size

        Note:
            Includes preprocessing to handle infinite or extremely large values in parameter bounds
            by capping them at ±1e30 to prevent errors in scikit-learn models.
        """
        if not self.use_ml or not SKLEARN_AVAILABLE or self.step_size_model is None:
            # Fall back to default step size
            param = self.parameters[param_idx]
            return self.get_step_size_for_parameter(param.name, base_step_size)

        try:
            param = self.parameters[param_idx]

            # Process bounds to avoid infinite values
            lower_bound = param.bounds[0] if param.bounds_on and param.bounds[0] is not None else -1e30
            upper_bound = param.bounds[1] if param.bounds_on and param.bounds[1] is not None else 1e30

            # Ensure bounds are finite
            if lower_bound == float('-inf') or lower_bound < -1e30:
                lower_bound = -1e30
            if upper_bound == float('inf') or upper_bound > 1e30:
                upper_bound = 1e30

            # Calculate value range safely
            if param.bounds_on and None not in param.bounds:
                value_range = min(1e30, max(-1e30, upper_bound - lower_bound))
            else:
                value_range = 1e30

            # Try different relative changes and predict acceptance probability
            relative_changes = [0.01, 0.05, 0.1, 0.2, 0.5]
            best_prob = 0
            best_change = 0.1  # Default

            for rel_change in relative_changes:
                # Create feature vector with safe values
                features = np.array([[
                    param_idx,
                    self.current_iteration,
                    self.current_state.value,
                    param.value,
                    rel_change,
                    self.best_chi2,
                    param.bounds_on,
                    lower_bound,
                    upper_bound,
                    value_range,
                    self.parameter_sensitivities.get(param.name, 0.0),
                    self.no_improvement_count
                ]])

                # Ensure all values are finite
                features = np.nan_to_num(features, nan=0.0, posinf=1e30, neginf=-1e30)

                # Predict acceptance probability
                prob = self.step_size_model.predict(features)[0]

                if prob > best_prob:
                    best_prob = prob
                    best_change = rel_change

            # Scale base step size by the best relative change
            return base_step_size * best_change / 0.1

        except Exception as e:
            logging.error(f"Error predicting optimal step size: {str(e)}")
            # Fall back to default step size
            param = self.parameters[param_idx]
            return self.get_step_size_for_parameter(param.name, base_step_size)
