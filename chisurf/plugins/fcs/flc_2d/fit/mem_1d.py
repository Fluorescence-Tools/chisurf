"""
1D-MEM (Maximum Entropy Method) fitting functions for fluorescence decays.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy.optimize import minimize
import logging

class OneDMEMFitter:
    """
    Performs 1D Maximum Entropy Method (MEM) fitting on fluorescence decays.
    
    Based on TK_FitF_1DMEM_02.m.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_fit_result = None

    def fit_1d_mem(
        self,
        initial_mat_a: np.ndarray,
        initial_y0: float,
        fix_mat_a: int,
        fix_y0: int,
        regulator_const: float,
        mat_1dfdc_it: np.ndarray,
        mat_1dfdc: np.ndarray,
        mat_1dfdc_cor: np.ndarray,
        tau_values: np.ndarray,
        exp_curve: np.ndarray,
        mi_matrix: np.ndarray,
        irf: Optional[np.ndarray] = None,
        irf_params: Optional[Dict[str, int]] = None,
        max_iterations: int = 10000,
        tolerance: float = 1e-6,
        progress_callback=None
    ) -> Dict:
        """
        Perform 1D-MEM fitting on 1D decay data.
        
        Parameters
        ----------
        ...
        irf : np.ndarray, optional
            Instrument Response Function.
        irf_params : dict, optional
            'rise_fl': signal start index, 'rise_irf': IRF start index, 
            'min_irf': window start, 'max_irf': window end.
        """
        n_components = initial_mat_a.shape[0]
        n_times = len(mat_1dfdc_it)
        
        # Basis set preparation (Convolution if IRF provided)
        if irf is not None and irf_params is not None:
            processed_exp_curve = np.zeros((n_times, exp_curve.shape[1]))
            rise_fl = irf_params.get('rise_fl', 0)
            rise_irf = irf_params.get('rise_irf', 0)
            min_irf = irf_params.get('min_irf', 0)
            max_irf = irf_params.get('max_irf', len(irf))
            
            idev = rise_irf - rise_fl
            
            for k in range(exp_curve.shape[1]):
                basis = exp_curve[:, k]
                convolved = np.zeros(n_times)
                for i in range(min_irf, max_irf):
                    if i < len(irf):
                        weight = irf[i]
                        start_idx = i - idev
                        if start_idx < n_times:
                            len_segment = min(n_times - start_idx, n_times)
                            if start_idx >= 0:
                                convolved[start_idx:start_idx+len_segment] += weight * basis[:len_segment]
                processed_exp_curve[:, k] = convolved / (np.max(convolved) if np.max(convolved) > 0 else 1.0)
        else:
            processed_exp_curve = exp_curve

        # Prepare diff matrix for baseline
        
        # Prepare diff matrix for baseline
        diff_mat_1dfdc_it = np.zeros_like(mat_1dfdc_it)
        diff_mat_1dfdc_it[1:] = mat_1dfdc_it[1:] - mat_1dfdc_it[:-1]
        diff_mat_1dfdc_it[0] = diff_mat_1dfdc_it[1] if len(diff_mat_1dfdc_it) > 1 else 1.0

        # Prepare parameters
        initial_params = []
        if fix_mat_a != 1:
            for i in range(n_components):
                initial_params.append(abs(initial_mat_a[i, 0]) if fix_mat_a == 2 else initial_mat_a[i, 0])
        if fix_y0 != 1:
            initial_params.append(abs(initial_y0) if fix_y0 == 2 else initial_y0)
        
        initial_params = np.array(initial_params)

        def objective_function(params):
            count = 0
            if fix_mat_a != 1:
                mat_a = np.zeros((n_components, 1))
                for i in range(n_components):
                    mat_a[i, 0] = abs(params[count]) if fix_mat_a == 2 else params[count]
                    count += 1
            else:
                mat_a = initial_mat_a
            
            if fix_y0 != 1:
                y0 = abs(params[count]) if fix_y0 == 2 else params[count]
                count += 1
            else:
                y0 = initial_y0
                
            # Avoid log(0)
            var_range = np.max(mat_a) - np.min(mat_a)
            if var_range < 1e-10: var_range = 1.0
            mat_a_safe = mat_a + (mat_a == 0) * (var_range * 1e-10)
            
            # Model: Model = (Mat_A' * ExpCurve')' + y0*Dif_Mat_1DFDC_It
            model = (mat_a_safe.T @ processed_exp_curve.T).T + y0 * diff_mat_1dfdc_it.reshape(-1, 1)
            model = model.flatten()
            
            # Chi2
            var1 = np.mean(mat_1dfdc)
            if var1 <= 0: var1 = 1.0
            var = ((mat_1dfdc_cor - model) ** 2) / (mat_1dfdc + var1)
            chi2 = np.sum(var) / len(mat_1dfdc)
            
            # Entropy
            entropy = 0.0
            var = mat_a_safe[:, 0]
            var1 = np.sum(var)
            var_safe = mat_a_safe[:, 0] / (mi_matrix[:, 0] + np.max(mi_matrix[:, 0]) * 1e-10)
            var_log = mat_a_safe[:, 0] * np.log(var_safe)
            var2 = np.sum(var_log)
            var3 = np.sum(mi_matrix[:, 0])
            entropy = (var1 - var3 - var2)
            
            q_value = chi2 - 2 * entropy / regulator_const
            return q_value

        res = minimize(
            objective_function,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': max_iterations, 'xatol': tolerance}
        )
        
        # Extract final
        count = 0
        final_mat_a = np.zeros((n_components, 1))
        if fix_mat_a != 1:
            for i in range(n_components):
                final_mat_a[i, 0] = abs(res.x[count]) if fix_mat_a == 2 else res.x[count]
                count += 1
        else:
            final_mat_a = initial_mat_a
        
        if fix_y0 != 1:
            final_y0 = abs(res.x[count]) if fix_y0 == 2 else res.x[count]
        else:
            final_y0 = initial_y0
            
        final_model = (final_mat_a.T @ processed_exp_curve.T).T + final_y0 * diff_mat_1dfdc_it.reshape(-1, 1)
        
        return {
            'success': res.success,
            'mat_a': final_mat_a,
            'y0': final_y0,
            'model': final_model.flatten(),
            'q_value': res.fun
        }
