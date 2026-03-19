"""
2D-MEM (Maximum Entropy Method) fitting functions for 2D-FLCS analysis.

This module implements the 2D-MEM fitting algorithm from MATLAB code
TK_FitF_2DMEM_07.m, adapted for Python/ChiSurf integration.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from scipy.optimize import minimize
import logging

from .mem_1d import OneDMEMFitter

class TwoDMEMFitter:
    """
    Performs 2D Maximum Entropy Method (MEM) fitting on 2D-FDC matrices.
    
    This class implements an optimization algorithm to decompose a 2D-FDC matrix
    into a distribution of components with specific fluorescence lifetimes and 
    transition rates. It uses the Maximum Entropy Method for regularization.
    
    The implementation is based on TK_FitF_2DMEM_07.m.
    """
    
    def __init__(self):
        """Initialize the 2D-MEM fitter."""
        self.logger = logging.getLogger(__name__)
        self.last_fit_result = None
    
    def fit_2d_mem(
        self,
        initial_mat_a: np.ndarray,
        initial_mat_g: np.ndarray,
        initial_y0: float,
        fix_mat_a: np.ndarray,
        fix_mat_g: np.ndarray, 
        fix_y0: int,
        regulator_const: float,
        mat_2dfdc_it: np.ndarray,
        mat_2dfdc: np.ndarray,
        mat_2dfdc_cor: np.ndarray,
        tau_values: np.ndarray,
        exp_curve: np.ndarray,
        mi_matrix: np.ndarray,
        max_iterations: int = 10000,
        tolerance: float = 1e-6,
        progress_callback=None
    ) -> Dict:
        """
        Perform 2D-MEM fitting on 2D-FDC data.
        """
        self.logger.info("Starting 2D-MEM fitting")
        
        n_components, n_states = initial_mat_a.shape
        
        # Prepare initial parameters
        initial_params = self._prepare_initial_parameters(
            initial_mat_a, initial_mat_g, initial_y0,
            fix_mat_a, fix_mat_g, fix_y0,
            n_components, n_states
        )
        
        # Prepare differential matrix for y0 calculation
        diff_mat_2dfdc_it = self._prepare_diff_matrix(mat_2dfdc_it)
        
        # Define objective function
        def objective_function(params):
            return self._expfun(
                params, initial_mat_a, initial_mat_g, initial_y0,
                fix_mat_a, fix_mat_g, fix_y0,
                mat_2dfdc, mat_2dfdc_cor, diff_mat_2dfdc_it,
                tau_values, exp_curve, mi_matrix,
                n_components, n_states, regulator_const
            )
        
        # Optimize with MATLAB-like settings
        self.logger.info(f"[2D-MEM][Progress] Starting MATLAB-style optimization with {max_iterations} max evaluations")
        
        # Calculate initial Q-value for reference
        initial_q = objective_function(initial_params)
        self.logger.info(f"[2D-MEM][Progress] Initial Q-value: {initial_q:.6f}")
        
        try:
            iteration_count = 0
            
            def matlab_style_objective(params):
                nonlocal iteration_count
                iteration_count += 1
                result = objective_function(params)
                
                if progress_callback is not None:
                    progress_value = min(iteration_count / max_iterations, 1.0)
                    progress_callback(progress_value)
                
                if iteration_count % 100 == 1:
                    progress_value = min(iteration_count / max_iterations, 1.0)
                    self.logger.info(f"[2D-MEM][Progress] Evaluation {iteration_count}/{max_iterations} ({progress_value*100:.1f}%), Q={result:.3f}")
                
                return result
            
            result = minimize(
                matlab_style_objective,
                initial_params,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iterations,
                    'xatol': tolerance,
                    'fatol': tolerance * 0.1,
                    'disp': False,
                    'adaptive': False
                }
            )
            
            self.logger.info(f"[2D-MEM][Progress] Optimization completed - Success: {result.success}, Q: {result.fun:.6f}, Evaluations: {iteration_count}")
            
        except Exception as e:
            self.logger.error(f"[2D-MEM][Error] Optimization failed: {e}")
            result = type('Result', (), {
                'success': False,
                'fun': np.inf,
                'nit': 0,
                'message': str(e),
                'x': initial_params
            })()
        
        if progress_callback is not None:
            progress_callback(1.0)
        
        final_mat_a, final_mat_g, final_y0 = self._extract_final_matrices(
            result.x, initial_mat_a, initial_mat_g, initial_y0,
            fix_mat_a, fix_mat_g, fix_y0,
            n_components, n_states
        )
        
        mat_m_2dflc = final_mat_a @ final_mat_g @ final_mat_a.T
        final_model = exp_curve @ mat_m_2dflc @ exp_curve.T + final_y0 * diff_mat_2dfdc_it
        
        final_chi2 = self._calculate_chi2(final_model, mat_2dfdc_cor, mat_2dfdc)
        final_entropy = self._calculate_entropy(final_mat_a, mi_matrix)
        final_q = final_chi2 - 2 * final_entropy / regulator_const
        
        fit_result = {
            'success': result.success,
            'message': result.message,
            'fun': result.fun,
            'nit': result.nit,
            'estimates': result.x,
            'mat_a': final_mat_a,
            'mat_g': final_mat_g,
            'y0': final_y0,
            'model': final_model,
            'chi2': final_chi2,
            'entropy': final_entropy,
            'q_value': final_q,
            'converged': result.success
        }
        
        self.last_fit_result = fit_result
        self.logger.info(f"2D-MEM fitting complete. Q-value: {final_q:.6f}")
        
        return fit_result
    
    def _prepare_initial_parameters(
        self,
        initial_mat_a: np.ndarray,
        initial_mat_g: np.ndarray,
        initial_y0: float,
        fix_mat_a: np.ndarray,
        fix_mat_g: np.ndarray,
        fix_y0: int,
        n_components: int,
        n_states: int
    ) -> np.ndarray:
        """Prepare initial parameter vector for optimization."""
        n_tau_values = initial_mat_a.shape[0]
        params = []
        
        for k in range(n_states):
            fix_val = fix_mat_a[k] if isinstance(fix_mat_a, np.ndarray) else fix_mat_a
            if fix_val == 1:
                continue
            elif fix_val == 3:
                params.append(1.0)
            else:
                for i in range(n_tau_values):
                    if fix_val == 2:
                        params.append(abs(initial_mat_a[i, k]))
                    else:
                        params.append(initial_mat_a[i, k])
        
        for k in range(n_states):
            for i in range(n_states):
                fix_val = fix_mat_g[i, k] if isinstance(fix_mat_g, np.ndarray) and fix_mat_g.ndim == 2 else fix_mat_g
                if fix_val == 1:
                    continue
                elif fix_val == 2:
                    params.append(abs(initial_mat_g[i, k]))
                elif fix_val == 0:
                    params.append(initial_mat_g[i, k])
                elif fix_val == 3 and i <= k:
                    params.append(abs(initial_mat_g[i, k]))
        
        if fix_y0 != 1:
            if fix_y0 == 2:
                params.append(abs(initial_y0))
            else:
                params.append(initial_y0)
        
        return np.array(params)
    
    def _prepare_diff_matrix(self, mat_2dfdc_it: np.ndarray) -> np.ndarray:
        """Prepare differential matrix for y0 calculation using kronecker product."""
        diff_mat = mat_2dfdc_it.copy()
        var = len(mat_2dfdc_it)
        diff_mat[1:var] = mat_2dfdc_it[1:var] - mat_2dfdc_it[0:var-1]
        diff_mat[0] = diff_mat[1] if var > 1 else 1.0
        return np.kron(diff_mat, diff_mat.reshape(-1, 1))
    
    def _expfun(
        self,
        params: np.ndarray,
        initial_mat_a: np.ndarray,
        initial_mat_g: np.ndarray,
        initial_y0: float,
        fix_mat_a: np.ndarray,
        fix_mat_g: np.ndarray,
        fix_y0: int,
        mat_2dfdc: np.ndarray,
        mat_2dfdc_cor: np.ndarray,
        diff_mat_2dfdc_it: np.ndarray,
        tau_values: np.ndarray,
        exp_curve: np.ndarray,
        mi_matrix: np.ndarray,
        n_components: int,
        n_states: int,
        regulator_const: float
    ) -> float:
        """Objective function for 2D-MEM optimization with error handling."""
        try:
            if hasattr(self, '_call_count'):
                self._call_count += 1
            else:
                self._call_count = 1
            
            if not np.all(np.isfinite(params)):
                return np.inf
            
            mat_a, mat_g, y0 = self._extract_final_matrices(
                params, initial_mat_a, initial_mat_g, initial_y0,
                fix_mat_a, fix_mat_g, fix_y0,
                n_components, n_states
            )
            
            if not np.all(np.isfinite(mat_a)) or not np.all(np.isfinite(mat_g)) or not np.isfinite(y0):
                return np.inf
            
            mat_a = np.abs(mat_a)
            y0 = np.abs(y0)
            
            var_range = np.max(mat_a) - np.min(mat_a)
            if var_range < 1e-10:
                var_range = 1.0
            mat_a = mat_a + (mat_a == 0) * (var_range * 1e-7)
            
            mat_m_2dflc = mat_a @ mat_g @ mat_a.T
            mat_model = exp_curve @ mat_m_2dflc @ exp_curve.T + y0 * diff_mat_2dfdc_it
            
            if not np.all(np.isfinite(mat_model)):
                return np.inf
            
            var1 = np.mean(mat_2dfdc)
            if var1 <= 0:
                var1 = 1.0
            var = ((mat_2dfdc_cor - mat_model) ** 2) / (mat_2dfdc + var1)
            imax = mat_2dfdc.shape[0]
            chi2 = np.sum(var) / (imax * imax)
            
            entropy = self._calculate_entropy(mat_a, mi_matrix)
            q_value = chi2 - 2 * entropy / regulator_const
            
            if not np.isfinite(q_value):
                return np.inf
            
            return q_value
            
        except Exception as e:
            return np.inf
    
    def _extract_final_matrices(
        self,
        params: np.ndarray,
        initial_mat_a: np.ndarray,
        initial_mat_g: np.ndarray,
        initial_y0: float,
        fix_mat_a: np.ndarray,
        fix_mat_g: np.ndarray,
        fix_y0: int,
        n_components: int,
        n_states: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Extract final matrices from parameter vector."""
        n_tau_values = initial_mat_a.shape[0]
        mat_a = np.zeros((n_tau_values, n_states))
        mat_g = np.zeros((n_states, n_states))
        param_idx = 0
        
        for k in range(n_states):
            fix_val = fix_mat_a[k] if isinstance(fix_mat_a, np.ndarray) else fix_mat_a
            if fix_val == 1:
                mat_a[:, k] = initial_mat_a[:, k]
            elif fix_val == 3:
                var_amp = abs(params[param_idx])
                mat_a[:, k] = initial_mat_a[:, k] * var_amp
                param_idx += 1
            else:
                for i in range(n_tau_values):
                    if fix_val == 2:
                        mat_a[i, k] = abs(params[param_idx])
                    else:
                        mat_a[i, k] = params[param_idx]
                    param_idx += 1
        
        for k in range(n_states):
            for i in range(n_states):
                fix_val = fix_mat_g[i, k] if isinstance(fix_mat_g, np.ndarray) and fix_mat_g.ndim == 2 else fix_mat_g
                if fix_val == 1:
                    mat_g[i, k] = initial_mat_g[i, k]
                elif fix_val == 2:
                    mat_g[i, k] = abs(params[param_idx])
                    param_idx += 1
                elif fix_val == 0:
                    mat_g[i, k] = params[param_idx]
                    param_idx += 1
                elif fix_val == 3:
                    if i <= k:
                        mat_g[i, k] = abs(params[param_idx])
                        mat_g[k, i] = abs(params[param_idx])
                        param_idx += 1
        
        if fix_y0 != 1:
            if fix_y0 == 2:
                y0 = abs(params[param_idx])
            else:
                y0 = params[param_idx]
        else:
            y0 = initial_y0
        
        return mat_a, mat_g, y0
    
    def _calculate_chi2(
        self,
        model: np.ndarray,
        data_cor: np.ndarray,
        data: np.ndarray
    ) -> float:
        """Calculate chi-squared statistic."""
        var1 = np.mean(data)
        var = ((data_cor - model) ** 2) / (data + var1)
        imax = data.shape[0]
        return np.sum(var) / (imax * imax)
    
    def _calculate_entropy(
        self,
        mat_a: np.ndarray,
        mi_matrix: np.ndarray
    ) -> float:
        """Calculate entropy for regularization - match MATLAB exactly."""
        entropy = 0.0
        n_tau_values, n_states = mat_a.shape
        
        for k in range(n_states):
            var = mat_a[:, k]
            var1 = np.sum(var)
            var_safe = (mat_a[:, k] + np.max(var) * 1e-10) / (mi_matrix[:, k] + np.max(mi_matrix[:, k]) * 1e-10)
            var = mat_a[:, k] * np.log(var_safe)
            var2 = np.sum(var)
            var3 = np.sum(mi_matrix[:, k])
            entropy += (var1 - var3 - var2)
        
        return entropy
    
    def fit_2d_mem_wrapper(
        self,
        mat_2dfdc: np.ndarray,
        mat_2dfdc_cor: np.ndarray,
        mat_2dfdc_it: np.ndarray,
        n_components: int,
        tau_range: tuple = (0.05, 5.05),
        tau_step: float = 0.05,
        regulator: float = 0.1,
        y0_initial: float = 1000.0,
        use_1d_init: bool = True,
        max_iterations: int = 10000,
        tolerance: float = 1e-6,
        progress_callback=None
    ) -> dict:
        """
        Wrapper method that converts high-level parameters to fit_2d_mem parameters.
        
        Optional use_1d_init pre-fits the marginals to initialize Mat_A accurately.
        """
        self.logger.info(f"Starting 2D-MEM fitting with {n_components} components")
        
        tau_min, tau_max = tau_range
        tau_values = np.arange(tau_min, tau_max + tau_step, tau_step)
        n_tau = len(tau_values)
        
        exp_curve = np.zeros((len(mat_2dfdc_it), n_tau))
        for i, tau in enumerate(tau_values):
            exp_curve[:, i] = np.exp(-mat_2dfdc_it / tau)
        
        n_states = n_components
        
        # 1D-MEM Initialization (The "Trick")
        if use_1d_init:
            self.logger.info("Initializing with 1D-MEM pre-fit on marginals...")
            # 1. Marginal 1D decay (sum of 2D matrix)
            # MATLAB: Mat_1DFDC_M = sum(Mat_2DFDC_Cor, 2)
            marginal_decay = np.sum(mat_2dfdc_cor, axis=1)
            
            # 2. Fit with OneDMEMFitter
            fitter_1d = OneDMEMFitter()
            # mi_matrix for 1D is flat ones
            mi_1d = np.ones((n_tau, 1))
            init_a_1d = np.ones((n_tau, 1)) * (np.mean(marginal_decay) / n_tau)
            
            fit_1d = fitter_1d.fit_1d_mem(
                initial_mat_a=init_a_1d,
                initial_y0=y0_initial / n_tau,
                fix_mat_a=0, fix_y0=0,
                regulator_const=regulator * 0.1, # Slightly less regularization for seed?
                mat_1dfdc_it=mat_2dfdc_it,
                mat_1dfdc=marginal_decay,
                mat_1dfdc_cor=marginal_decay,
                tau_values=tau_values,
                exp_curve=exp_curve,
                mi_matrix=mi_1d,
                max_iterations=2000 # Quick fit
            )
            
            if fit_1d['success']:
                self.logger.info("1D-MEM pre-fit successful.")
                dist_1d = fit_1d['mat_a'].flatten()
                # Initialize species with slight perturbations of the 1D fit
                initial_mat_a = np.zeros((n_tau, n_states))
                for k in range(n_states):
                    # Add 10% random noise or slight shift for species differentiation?
                    # Most simple: just copy it or scale it.
                    initial_mat_a[:, k] = dist_1d / n_states
            else:
                self.logger.warning("1D-MEM initialization failed, falling back to uniform.")
                use_1d_init = False

        if not use_1d_init:
            data_scale = np.mean(mat_2dfdc[mat_2dfdc > 0])
            initial_mat_a = np.ones((n_tau, n_states)) * (data_scale / n_tau)
            for k in range(n_states):
                tau_weight = tau_values[k] / np.mean(tau_values)
                initial_mat_a[:, k] *= tau_weight
        
        initial_mat_g = np.eye(n_states) * 0.5
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    tau_ratio = tau_values[i] / tau_values[j]
                    initial_mat_g[i, j] = 0.1 * min(tau_ratio, 1.0/tau_ratio)
        
        fix_mat_a = np.zeros(n_states, dtype=int)
        fix_mat_g = np.zeros((n_states, n_states), dtype=int)
        fix_y0 = 0
        
        mi_matrix = np.ones((n_tau, n_states))
        
        result = self.fit_2d_mem(
            initial_mat_a=initial_mat_a,
            initial_mat_g=initial_mat_g,
            initial_y0=y0_initial,
            fix_mat_a=fix_mat_a,
            fix_mat_g=fix_mat_g,
            fix_y0=fix_y0,
            regulator_const=regulator,
            mat_2dfdc_it=mat_2dfdc_it,
            mat_2dfdc=mat_2dfdc,
            mat_2dfdc_cor=mat_2dfdc_cor,
            tau_values=tau_values,
            exp_curve=exp_curve,
            mi_matrix=mi_matrix,
            max_iterations=max_iterations,
            tolerance=tolerance,
            progress_callback=progress_callback
        )
        
        return result
