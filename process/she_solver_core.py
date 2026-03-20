"""
she_solver_core.py
==================
Core physics and linear algebra solver for Single-Shot Spherical Harmonic Expansion.
Updated with Two-Threshold Progressive Lambda (Generalized Tikhonov) regularization.
"""

from __future__ import annotations
import numpy as np
import math
from scipy.special import spherical_jn, sph_harm_y
from utils import hankel1

# Attempt to import Speed of Sound and new solver configs
try:
    from config_process import SPEED_OF_SOUND
except ImportError:
    SPEED_OF_SOUND = None

def _solve_one_frequency(
    f_hz: float,
    P_complex: np.ndarray,
    coords_sph: tuple[np.ndarray, np.ndarray, np.ndarray],
    order_N: int,
    k_val: float | None = None,
    CONDITION_METRICS: bool = False,
    noise_floor_start_db: float | None = None,
    noise_floor_max_db: float | None = None,
    max_lambda: float | None = None
) -> tuple[np.ndarray, dict]:
    """
    Solves for SHE coefficients for a single frequency using a Dual Basis.
    Accepts optional overrides for progressive lambda thresholds.
    """
    # Wavenumber setup
    if k_val is not None:
        k = k_val
    else:
        if SPEED_OF_SOUND is None:
            raise ValueError("Speed of Sound not found and 'k_val' not provided.")
        k = 2 * math.pi * f_hz / SPEED_OF_SOUND

    r, th, ph = coords_sph
    kr = r * k
    base_norm = np.linalg.norm(P_complex)

    # Build Matrix
    A_cols = []
    final_N = order_N
    for n in range(order_N + 1):
        hn = hankel1(n, kr)
        jn = spherical_jn(n, kr)
        if not (np.all(np.isfinite(hn)) and np.all(np.isfinite(jn))):
            final_N = n - 1
            break
        for m_ord in range(-n, n + 1):
            Y = sph_harm_y(n, m_ord, th, ph)
            A_cols.extend([hn * Y, jn * Y])

    if final_N < 0:
        return np.array([]), {'residual_norm': base_norm, 'cond_pre': 0.0, 'cond_post': 0.0, 'final_N': -1, 'stop_reason': "Instability"}

    A_solve = np.column_stack(A_cols)
    b_solve = P_complex

    # Two-Threshold Progressive Lambda / Generalized Tikhonov Logic
    try:
        from config_process import NOISE_FLOOR_START_DB, NOISE_FLOOR_MAX_DB, MAX_LAMBDA
    except ImportError:
        NOISE_FLOOR_START_DB = -30.0  # Where damping begins
        NOISE_FLOOR_MAX_DB = -50.0    # Where damping reaches maximum
        MAX_LAMBDA = 0.01

    # Override config values with function arguments if they are explicitly provided
    active_start_db = noise_floor_start_db if noise_floor_start_db is not None else NOISE_FLOOR_START_DB
    active_max_db = noise_floor_max_db if noise_floor_max_db is not None else NOISE_FLOOR_MAX_DB
    active_max_lambda = max_lambda if max_lambda is not None else MAX_LAMBDA

    U, s_vals, Vh = np.linalg.svd(A_solve, full_matrices=False)
    
    s_max = s_vals[0]
    
    # Convert active dB floors to linear boundaries
    s_start = s_max * (10 ** (active_start_db / 20.0))
    s_max_damp = s_max * (10 ** (active_max_db / 20.0))
    
    w_progressive = np.zeros_like(s_vals)
    effective_lambda_max = 0.0
    
    for i, s in enumerate(s_vals):
        if s >= s_start:
            # Zone 1: Signal Zone
            lam_i = 0.0
        elif s <= s_max_damp:
            # Zone 2: Noise Zone
            lam_i = active_max_lambda
        else:
            # Zone 3: Transition Slope
            # Calculate progress from 0.0 (at s_start) to 1.0 (at s_max_damp)
            if s_start > s_max_damp: # Guard against identical thresholds
                progress = (s_start - s) / (s_start - s_max_damp)
                lam_i = active_max_lambda * (progress ** 2)
            else:
                lam_i = active_max_lambda
        
        # Generalized Tikhonov filter weight: s_i / (s_i^2 + lambda_i)
        w_progressive[i] = s / (s**2 + lam_i)
        effective_lambda_max = max(effective_lambda_max, lam_i)
            
    # Compute solution: x = V * W_progressive * U^H * b
    x = Vh.conj().T @ (w_progressive * (U.conj().T @ b_solve))
    
    lam = effective_lambda_max

    # Metrics
    cond_pre = (s_vals[0] / s_vals[-1]) if (s_vals[-1] > 0) else 0.0
    
    # Calculate the exact condition number of the applied pseudo-inverse operator
    # The condition number of the inverted system is max(w) / min(w)
    valid_w = w_progressive[w_progressive > 0]
    if len(valid_w) > 0:
        cond_post = np.max(valid_w) / np.min(valid_w)
    else:
        # Fallback to standard augmented matrix condition number
        if lam > 0:
            cond_post = math.sqrt((s_vals[0]**2 + lam) / (s_vals[-1]**2 + lam))
        else:
            cond_post = cond_pre

    resid_norm = np.linalg.norm(b_solve - (A_solve @ x))

    metrics = {
        'residual_norm': resid_norm,
        'cond_pre': cond_pre,
        'cond_post': cond_post,
        'final_N': final_N,
        'stop_reason': "Target N Reached" if final_N == order_N else "Numerical Instability (Truncated)"
    }
    return x, metrics