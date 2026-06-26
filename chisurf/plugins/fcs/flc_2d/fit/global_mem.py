"""
Global 2D-MEM fitting functions for 2D-FLCS analysis.
"""

import numpy as np
from typing import Dict, List, Any
from scipy.optimize import minimize
import logging

class GlobalTwoDMEMFitter:
    """
    Performs Global 2D-MEM fitting across multiple dT datasets.
    
    Based on TK_GFitF_2DMEM_05.m.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fit_global_2d_mem(
        self,
        initial_mat_a: np.ndarray,      # Global (n_tau x n_states)
        initial_mat_g_list: list,      # dT-specific [n_states x n_states]
        initial_y0_list: list,         # dT-specific
        fix_mat_a: int,
        fix_mat_g: int,
        fix_y0: int,
        regulator_const: float,
        mat_2dfdc_it: np.ndarray,
        mat_2dfdc_list: list,
        mat_2dfdc_cor_list: list,
        tau_values: np.ndarray,
        exp_curve: np.ndarray,
        mi_matrix: np.ndarray,
        max_iterations: int = 10000,
        tolerance: float = 1e-6
    ) -> Dict:
        """
        Globally fit multiple 2D-FDC matrices with a shared state distribution (Mat_A).
        """
        n_dt = len(mat_2dfdc_list)
        n_tau, n_states = initial_mat_a.shape
        
        # Prepare diff matrix
        diff_mat_it = np.zeros_like(mat_2dfdc_it)
        diff_mat_it[1:] = mat_2dfdc_it[1:] - mat_2dfdc_it[:-1]
        diff_mat_it[0] = diff_mat_it[1]
        diff_mat_it_kron = np.kron(diff_mat_it, diff_mat_it.reshape(-1, 1))

        # Build initial parameters
        params = []
        # 1. Global Mat_A
        if fix_mat_a != 1:
            for k in range(n_states):
                for i in range(n_tau):
                    params.append(abs(initial_mat_a[i, k]) if fix_mat_a == 2 else initial_mat_a[i, k])
        
        # 2. Local Mat_G and y0 for each dT
        for t in range(n_dt):
            if fix_mat_g != 1:
                for i in range(n_states):
                    for k in range(n_states):
                        if fix_mat_g == 3: # symmetric
                            if i <= k: params.append(abs(initial_mat_g_list[t][i, k]))
                        else:
                            params.append(abs(initial_mat_g_list[t][i, k]) if fix_mat_g == 2 else initial_mat_g_list[t][i, k])
            
            if fix_y0 != 1:
                params.append(abs(initial_y0_list[t]) if fix_y0 == 2 else initial_y0_list[t])

        params = np.array(params)

        def objective(p):
            idx = 0
            # Extract Mat_A
            if fix_mat_a != 1:
                mat_a = np.zeros((n_tau, n_states))
                for k in range(n_states):
                    for i in range(n_tau):
                        mat_a[i, k] = abs(p[idx]) if fix_mat_a == 2 else p[idx]
                        idx += 1
            else:
                mat_a = initial_mat_a
            
            # Avoid log(0)
            range_a = np.max(mat_a) - np.min(mat_a)
            if range_a < 1e-10: range_a = 1.0
            mat_a_safe = mat_a + (mat_a == 0) * (range_a * 1e-7)
            
            total_chi2 = 0
            # For each dT
            for t in range(n_dt):
                if fix_mat_g != 1:
                    mat_g = np.zeros((n_states, n_states))
                    for i in range(n_states):
                        for k in range(n_states):
                            if fix_mat_g == 3:
                                if i <= k:
                                    val = abs(p[idx])
                                    mat_g[i, k] = val
                                    mat_g[k, i] = val
                                    idx += 1
                            else:
                                mat_g[i, k] = abs(p[idx]) if fix_mat_g == 2 else p[idx]
                                idx += 1
                else:
                    mat_g = initial_mat_g_list[t]
                
                if fix_y0 != 1:
                    y0 = abs(p[idx]) if fix_y0 == 2 else p[idx]
                    idx += 1
                else:
                    y0 = initial_y0_list[t]
                
                # Model
                mat_m_2dflc = mat_a_safe @ mat_g @ mat_a_safe.T
                mat_model = exp_curve @ mat_m_2dflc @ exp_curve.T + y0 * diff_mat_it_kron
                
                # Local Chi2
                m_data = mat_2dfdc_list[t]
                m_cor = mat_2dfdc_cor_list[t]
                var1 = np.mean(m_data)
                if var1 <= 0: var1 = 1.0
                err = ((m_cor - mat_model) ** 2) / (m_data + var1)
                total_chi2 += np.sum(err) / (m_data.shape[0]**2)
            
            # Global Entropy
            entropy = 0
            for k in range(n_states):
                v_a = mat_a_safe[:, k]
                v_mi = mi_matrix[:, k]
                v_sum = np.sum(v_a)
                v_safe = v_a / (v_mi + np.max(v_mi)*1e-10)
                v_log = v_a * np.log(v_safe)
                entropy += (v_sum - np.sum(v_mi) - np.sum(v_log))
                
            q_val = (total_chi2 / n_dt) - 2 * entropy / regulator_const
            return q_val

        res = minimize(
            objective, params, method='Nelder-Mead',
            options={'maxiter': max_iterations, 'xatol': tolerance}
        )

        return {'success': res.success, 'q_value': res.fun, 'params': res.x}
