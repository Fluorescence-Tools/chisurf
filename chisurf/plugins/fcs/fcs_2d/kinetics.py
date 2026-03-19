"""
Population kinetics and rate equation solvers for 2D-FLCS.

This module provides tools for modeling the time-dependent populations of 
fluorophore states using transition rate matrices.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import expm

def solve_rate_equations(
    rate_matrix: np.ndarray, 
    time_axis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the rate equations dP/dt = K*P using matrix diagonalization/exponentials.
    
    This implementation follows the logic of TK_RateEq_MakeExpMatrix.m.
    It constructs the full transition matrix (including diagonal decay terms)
    and provides the eigenvalues and vectors for population evolution.

    Args:
        rate_matrix: Transition rate matrix (k_ij).
        time_axis: Time points to evaluate population at.

    Returns:
        Tuple containing:
        - eigenvalues: Eigenvalues of the modified rate matrix.
        - exp_matrix: Matrix exponential components.
        - eigen_vectors: Matrix of eigenvectors.
        - inv_eigen_vectors: Inverse of eigenvector matrix.
    """
    n = rate_matrix.shape[0]
    
    # Construct modified rate matrix (diagonal contains negative sum of outgoing rates)
    # ModifiedRateMatrix = RateMatrix - diag(sum(RateMatrix, axis=0))
    # This ensures sum(dP/dt) = 0 (population conservation)
    
    outgoing_rates = np.sum(rate_matrix, axis=0)
    modified_rate_matrix = rate_matrix - np.diag(outgoing_rates)
    
    # Diagonalization
    eigen_values, eigen_vectors = np.linalg.eig(modified_rate_matrix)
    inv_eigen_vectors = np.linalg.inv(eigen_vectors)
    
    # Check for complex values (standard rate equations usually have real eigvals)
    if not np.isrealobj(eigen_values):
        # We handle this by taking real part if it's just numerical noise
        # but warn if significant
        if np.any(np.abs(np.imag(eigen_values)) > 1e-10):
            import logging
            logging.getLogger(__name__).warning("Complex eigenvalues detected in rate matrix.")

    return eigen_values, None, eigen_vectors, inv_eigen_vectors

def calculate_population_evolution(
    rate_matrix: np.ndarray,
    initial_populations: np.ndarray,
    time_axis: np.ndarray
) -> np.ndarray:
    """
    Calculate the time evolution of state populations.
    
    Args:
        rate_matrix: n x n transition rate matrix.
        initial_populations: n x 1 initial population vector.
        time_axis: Time points (seconds or ticks).
        
    Returns:
        n x len(time_axis) array of populations over time.
    """
    eigvals, _, eigvecs, inv_eigvecs = solve_rate_equations(rate_matrix, time_axis)
    
    # P(t) = V * exp(D*t) * V^-1 * P(0)
    # Where V is eigvecs, D is diag(eigvals)
    
    n_states = len(initial_populations)
    n_times = len(time_axis)
    populations = np.zeros((n_states, n_times))
    
    for i, t in enumerate(time_axis):
        exp_d_t = np.diag(np.exp(eigvals * t))
        prop_matrix = eigvecs @ exp_d_t @ inv_eigvecs
        populations[:, i] = np.real(prop_matrix @ initial_populations)
        
    return populations

class KineticsModel:
    """
    Represents a multi-state kinetic model for 2D-FLCS analysis.
    """
    
    def __init__(self, n_states: int):
        self.n_states = n_states
        self.rate_matrix = np.zeros((n_states, n_states))
        self.state_lifetimes = np.ones(n_states)
        self.state_brightness = np.ones(n_states) # epsilon * Q
        
    def set_rate(self, from_state: int, to_state: int, rate: float):
        """Set transition rate k_ij (from j to i)."""
        self.rate_matrix[to_state, from_state] = rate
        
    def get_equilibrium_populations(self) -> np.ndarray:
        """Calculate steady-state populations."""
        # Solving (K - diag(sum(K))) * P = 0 with sum(P) = 1
        outgoing_rates = np.sum(self.rate_matrix, axis=0)
        M = self.rate_matrix - np.diag(outgoing_rates)
        
        # Add constraint sum(P) = 1
        M_constrained = np.vstack([M, np.ones(self.n_states)])
        b = np.zeros(self.n_states + 1)
        b[-1] = 1.0
        
        # Solve least squares for safety
        pop, _, _, _ = np.linalg.lstsq(M_constrained, b, rcond=None)
        return np.maximum(pop, 0.0) # Ensure non-negative
