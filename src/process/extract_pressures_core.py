#!/usr/bin/env python3
"""
extract_pressures_core.py
=========================
Core mathematical engine for extracting FRD/complex pressures from SHE coefficients.
Includes CTA-2034-A Spinorama metrics generation and dynamic inverse coordinate translation.
"""

from __future__ import annotations

import math
import multiprocessing
import sys
from pathlib import Path
from typing import List, Tuple, Union, Dict

import h5py
import numpy as np
from scipy.special import spherical_jn, spherical_yn, sph_harm_y
import schema

from utils import (
    spherical_to_cartesian, cartesian_to_spherical, translate_coordinates,
    hankel2, load_she_h5
)

# -------------------------------------------------
# Worker Function (Updated for Dynamic Translation)
# -------------------------------------------------

def _worker_calc_chunk(args):
    (indices, freqs_sub, coeffs_sub, n_used_sub,
     r_base, theta_base, phi_base, origins_mm_sub, obs_mode, c_sound) = args
     
    num_pts = len(r_base)
    num_freqs_chunk = len(freqs_sub)
    pressures_chunk = np.zeros((num_freqs_chunk, num_pts), dtype=np.complex128)
    
    is_internal = obs_mode.lower() == "internal"
    is_external = obs_mode.lower() == "external"

    for local_idx, f in enumerate(freqs_sub):
        
        # --- 1. Inverse Coordinate Translation ---
        # Shift the global observation coordinates inversely to the acoustic origin
        origin_m = origins_mm_sub[local_idx] / 1000.0
        r_eval, theta_eval, phi_eval = translate_coordinates(
            r_base, theta_base, phi_base, origin_m, inverse=False
        )
        # -----------------------------------------

        N_max = int(n_used_sub[local_idx])
        k_wave = 2 * math.pi * f / c_sound
        kr = k_wave * r_eval

        n_indices_list = []
        m_indices_list = []
        for n in range(N_max + 1):
            deg_range = range(-n, n + 1)
            count = len(deg_range)
            n_indices_list.extend([n] * count)
            m_indices_list.extend(deg_range)
            
        n_vec = np.array(n_indices_list)
        m_vec = np.array(m_indices_list)
        coeffs_k = coeffs_sub[local_idx]
        num_modes = len(n_vec)
        
        C_coeffs = coeffs_k[0 : 2*num_modes : 2]
        D_coeffs = coeffs_k[1 : 2*num_modes : 2]

        Y_nm = sph_harm_y(n_vec[:, np.newaxis], m_vec[:, np.newaxis], 
                          theta_eval[np.newaxis, :], phi_eval[np.newaxis, :])
        h_n_val = hankel2(n_vec[:, np.newaxis], kr[np.newaxis, :])
        j_n_val = spherical_jn(n_vec[:, np.newaxis], kr[np.newaxis, :])
        
        if is_internal:
            term = C_coeffs[:, np.newaxis] * h_n_val * Y_nm
        elif is_external:
            term = D_coeffs[:, np.newaxis] * j_n_val * Y_nm
        else:
            term = (C_coeffs[:, np.newaxis] * h_n_val + 
                    D_coeffs[:, np.newaxis] * j_n_val) * Y_nm
            
        pressures_chunk[local_idx, :] = np.sum(term, axis=0)
        
    return indices, pressures_chunk

# -------------------------------------------------
# Core Evaluation Function
# -------------------------------------------------

def evaluate_she_field(
    coords_sph: List[Tuple[float, float, float]] | np.ndarray,
    she_input: Union[str, Path, Dict],
    obs_mode: str = "Internal",
    c_sound: float = 343.0,
    use_optimized_origins: bool = True
) -> Dict[str, np.ndarray]:
    
    data = load_she_h5(she_input)
    freqs = data[schema.FREQS]
    coeffs = data[schema.COEFFS]
    n_used = data[schema.N_USED]

    # Fallback if processing older files without origins
    if not use_optimized_origins:
        origins_mm = np.zeros((len(freqs), 3))
    else:
        origins_mm = data[schema.ORIGINS_MM]
        if origins_mm is None:
            origins_mm = np.zeros((len(freqs), 3))
    
    pts_sph = np.array(coords_sph, dtype=float)
    theta_in = np.radians(pts_sph[:, 0])
    phi_in   = np.radians(pts_sph[:, 1])
    r_in     = pts_sph[:, 2]

    # The received coordinates are now considered absolute.
    # The static offset is pre-applied in the calling script (stage5).
    r_base, theta_base, phi_base = r_in, theta_in, phi_in
    num_pts = len(r_base)
    num_freqs = len(freqs)
    
    num_cpus = multiprocessing.cpu_count()
    num_chunks = max(1, min(num_freqs, num_cpus * 4))
    indices = np.arange(num_freqs)
    chunks = np.array_split(indices, num_chunks)
    
    tasks = []
    for chunk_idx in chunks:
        if len(chunk_idx) == 0: continue
        # Pass the spherical base and the origin slice to the worker
        tasks.append((chunk_idx, freqs[chunk_idx], coeffs[chunk_idx], n_used[chunk_idx],
                      r_base, theta_base, phi_base, origins_mm[chunk_idx], obs_mode, c_sound))

    pressures_all = np.zeros((num_freqs, num_pts), dtype=np.complex128)
    print(f"Starting parallel solve on {num_cpus} cores ({num_pts} points)...")
    
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results_iter = pool.imap(func=_worker_calc_chunk, iterable=tasks)
        total_tasks = len(tasks)
        for i, result in enumerate(results_iter):
            idx_range, p_chunk = result
            pressures_all[idx_range, :] = p_chunk
            percent = ((i + 1) / total_tasks) * 100
            sys.stdout.write(f"\rProgress: {percent:5.1f}% complete")
            sys.stdout.flush()
            
    print("\nCalculation complete.")

    eps = np.finfo(float).eps
    mags_db = 20 * np.log10(np.abs(pressures_all) + eps)
    phase_deg = np.angle(pressures_all, deg=True)

    return {
        "freqs": freqs,
        "complex": pressures_all,
        "magnitude": mags_db,
        "phase": phase_deg
    }