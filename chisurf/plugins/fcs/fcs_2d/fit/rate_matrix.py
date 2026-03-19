"""
Rate matrix (population kinetics) fitting functions for 2D-FLCS analysis.
"""

import numpy as np
from typing import Dict, Any
from scipy.optimize import minimize
import logging

class RateMatrixFitter:
    """
    Fits transition rates between states to experimental correlation curves.
    
    Based on TK_FitF_CorrelationDecay_RateMat_16_NotRatio.m.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fit_rate_matrix(
        self,
        xdata: np.ndarray,              # dT values
        ydata: np.ndarray,              # 3D array [n_states x n_states x n_dt]
        state_assign: np.ndarray,       # Map components -> states
        initial_rate_matrix: np.ndarray,
        initial_brightness: np.ndarray, # epsilon * Q per state
        initial_y0: np.ndarray,         # [n_states x n_states]
        fix_rates: np.ndarray,          # 0=free, 1=fixed, 2=abs
        fix_brightness: np.ndarray,
        fix_y0: np.ndarray,
        max_iterations: int = 10000,
        tolerance: float = 1e-6
    ) -> Dict:
        """
        Extract kinetic rates by fitting correlation decays.
        """
        n_states = initial_rate_matrix.shape[0]
        n_dt = len(xdata)
        
        # Flatten parameters for minimize
        params = []
        # 1. Rates (off-diagonal)
        for i in range(n_states):
            for j in range(n_states):
                if fix_rates[i, j] != 1:
                    params.append(abs(initial_rate_matrix[i, j]) if fix_rates[i, j] == 2 else initial_rate_matrix[i, j])
        
        # 2. Brightness
        for i in range(n_states):
            if fix_brightness[i] != 1:
                params.append(abs(initial_brightness[i]) if fix_brightness[i] == 2 else initial_brightness[i])
        
        # 3. y0
        for i in range(n_states):
            for j in range(n_states):
                if fix_y0[i, j] != 1:
                    params.append(abs(initial_y0[i, j]) if fix_y0[i, j] == 2 else initial_y0[i, j])
        
        params = np.array(params)

        def objective(p):
            idx = 0
            # Reconstruct matrices
            k_mat = np.zeros((n_states, n_states))
            for i in range(n_states):
                for j in range(n_states):
                    if fix_rates[i, j] != 1:
                        k_mat[i, j] = abs(p[idx]) if fix_rates[i, j] == 2 else p[idx]
                        idx += 1
                    else:
                        k_mat[i, j] = initial_rate_matrix[i, j]
            
            bright = np.zeros(n_states)
            for i in range(n_states):
                if fix_brightness[i] != 1:
                    bright[i] = abs(p[idx]) if fix_brightness[i] == 2 else p[idx]
                    idx += 1
                else:
                    bright[i] = initial_brightness[i]
            
            y0 = np.zeros((n_states, n_states))
            for i in range(n_states):
                for j in range(n_states):
                    if fix_y0[i, j] != 1:
                        y0[i, j] = abs(p[idx]) if fix_y0[i, j] == 2 else p[idx]
                        idx += 1
                    else:
                        y0[i, j] = initial_y0[i, j]
            
            # Model calculation
            total_err = 0
            # P_eq
            row_sum = np.sum(k_mat, axis=0)
            M = k_mat - np.diag(row_sum)
            # Find equilibrium: M*P = 0
            eigvals, eigvecs = np.linalg.eig(M)
            p_eq = np.real(eigvecs[:, np.argmin(np.abs(eigvals))])
            p_eq = p_eq / np.sum(p_eq)
            
            # Diagonalization for evolution
            evals, evecs = np.linalg.eig(M)
            iev = np.linalg.inv(evecs)
            
            for t_idx, t in enumerate(xdata):
                exp_dt = evecs @ np.diag(np.exp(evals * t)) @ iev
                model_cor = np.zeros((n_states, n_states))
                for i in range(n_states):
                    for j in range(n_states):
                        # G_ij(t) = P_i(eq) * B_i * T_j|i(t) * B_j
                        model_cor[j, i] = p_eq[i] * bright[i] * exp_dt[j, i] * bright[j] + y0[j, i]
                
                err = np.sum((model_cor - ydata[:, :, t_idx])**2)
                total_err += err
            
            return total_err / (n_states * n_states * n_dt)

        res = minimize(objective, params, method='Nelder-Mead', options={'maxiter': max_iterations})
        
        return {'success': res.success, 'fun': res.fun, 'params': res.x}
