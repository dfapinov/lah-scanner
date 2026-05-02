#!/usr/bin/env python3

"""
Stage 2 – Spherical Harmonic Expansion (SHE) Driver (Refactored)
==============================================================

Description:
------------
Loads Stage 1 complex pressure data, prepares coordinates and weights, 
and distributes the Single-Shot solve to parallel workers using the 
core '_solve_one_frequency' function.

Updated to support optional 'MANUAL_ORDER_TABLE' and per-frequency 
optimized acoustic origins via 'USE_OPTIMIZED_ORIGINS' from config.
"""

from __future__ import annotations

# --- Standard library imports ---
import argparse
import logging
import math
import os
import sys
import time
import importlib
import re
from datetime import datetime
from typing import Optional, Tuple, Dict
from multiprocessing import Pool, cpu_count

# --- Third-party imports ---
import h5py
import numpy as np
import schema

# --- Local Project imports ---
try:
    from she_solver_core import _solve_one_frequency
    from utils import translate_coordinates, load_and_parse_npz, get_grid_limit, get_kr_limit
    from viewers import plot_she_results
except ImportError as e:
    sys.exit(f"Error: Could not import required module. {e}")

TARGET_MAX_F = 6000.0 # Experimentaly determined setting - the frequency at which the max order N is reached by the power law.

# =============================================================================
# --- Table Lookup Engine ---
# =============================================================================
def _get_table_limit(f_hz: float, use_manual_table: bool, manual_order_table: dict) -> int:
    if not use_manual_table or not manual_order_table:
        return 999
    
    sorted_cuts = sorted(manual_order_table.keys())
    for cutoff in sorted_cuts:
        if f_hz <= cutoff:
            return manual_order_table[cutoff]
    return manual_order_table[sorted_cuts[-1]]

# =============================================================================
# --- Worker Wrapper ---
# =============================================================================
def _init_worker(log_file_path: Optional[str] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file_path:
        handlers.append(logging.FileHandler(log_file_path, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s", 
        handlers=handlers,
        force=True
    )

def _worker_wrapper(args: Tuple) \
        -> Tuple[int, np.ndarray, float, float, int, np.ndarray, str]:
    k_idx, f, Pk, r, th, ph, N_grid, cfg = args

    kw = 2 * math.pi * f / cfg['speed_of_sound']
    
    N_kr = get_kr_limit(f, r, cfg['speed_of_sound'], cfg['kr_offset'])
    N_table = _get_table_limit(f, cfg['use_manual_table'], cfg['manual_order_table'])
    
    N_target = min(N_kr, N_grid, N_table, cfg['target_n_max'])

    # If manual table is not used, enforce a minimum order of 2.
    # This prevents order from dropping too low when using a low kr_offset.
    if not cfg['use_manual_table']:
        N_target = max(2, N_target)

    coeffs, metrics = _solve_one_frequency(
        f_hz=f,
        P_complex=Pk,
        coords_sph=(r, th, ph),
        order_N=N_target,
        k_val=kw,
        CONDITION_METRICS=cfg['condition_metrics'],
        noise_floor_start_db=cfg['noise_floor_start_db'],
        noise_floor_max_db=cfg['noise_floor_max_db'],
        max_lambda=cfg['max_lambda']
    )
    
    resid_norm = metrics['residual_norm']
    final_N = metrics['final_N']
    core_stop = metrics['stop_reason']
    
    if final_N < N_target:
        stop_reason = core_stop 
    elif cfg['use_manual_table'] and final_N == N_table and N_table < N_kr and N_table < N_grid:
        stop_reason = "Manual Table"
    elif final_N == N_kr and N_kr < cfg['target_n_max']:
        stop_reason = "KR-Limit"
    elif final_N == N_grid and N_grid < cfg['target_n_max']:
        stop_reason = "Grid-Limit"
    else:
        stop_reason = "User Hard Cap Reached"

    base_norm = np.linalg.norm(Pk)
    pct_err = resid_norm / max(base_norm, 1e-20) * 100.0
    
    if cfg['condition_metrics']:
        log_msg = f"{f:6.2f} Hz N={final_N} Resid={pct_err:.2f}% Pre-Cond={metrics['cond_pre']:.2e} Post-Cond={metrics['cond_post']:.2e} -> {stop_reason}"
    else:
        log_msg = f"{f:6.2f} Hz N={final_N} Resid={pct_err:.2f}% -> {stop_reason}"

    return k_idx, coeffs, resid_norm, metrics['cond_post'], final_N, metrics['residual_vector'], log_msg

# =============================================================================
# --- Main Driver Logic ---
# =============================================================================
def run_she_solve(
    input_filename_she: str,
    output_filename_she: str,
    input_dir_she: str,
    output_dir_she: str,
    target_n_max: int,
    use_manual_table: bool = False,
    manual_order_table: dict = None,
    noise_floor_start_db: float = -30.0,
    noise_floor_max_db: float = -50.0,
    max_lambda: float = 0.000001,
    condition_metrics: bool = True,
    use_optimized_origins: bool = True,
    save_to_disk: bool = True,
    speed_of_sound: float = 343.0,
    kr_offset: float = 2.0,
    jobs: int = None,
    show_plot: bool = True
) -> Dict:
    if manual_order_table is None:
        manual_order_table = {}
    if jobs is None:
        jobs = max(1, cpu_count() // 2)

    input_npz_path = os.path.join(input_dir_she, input_filename_she)

    worker_cfg = {
        'target_n_max': target_n_max,
        'use_manual_table': use_manual_table,
        'manual_order_table': manual_order_table,
        'condition_metrics': condition_metrics,
        'speed_of_sound': speed_of_sound,
        'noise_floor_start_db': noise_floor_start_db,
        'noise_floor_max_db': noise_floor_max_db,
        'max_lambda': max_lambda,
        'kr_offset': kr_offset
    }

    log_file_path = None
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if save_to_disk:
        os.makedirs(output_dir_she, exist_ok=True)
        timestamp_file = datetime.now().strftime("%d-%m-%Y_%H-%M")
        log_filename = f"solve_{timestamp_file}.log"
        log_file_path = os.path.join(output_dir_she, log_filename)
        handlers.append(logging.FileHandler(log_file_path, encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s", 
        handlers=handlers,
        force=True
    )
    logging.info("="*60)
    logging.info(f"SOLVER SESSION START: {datetime.now().strftime('%A, %d %B %Y at %I:%M %p')}")
    if save_to_disk:
        logging.info(f"LOG FILE:           {log_filename}")
        logging.info(f"OUTPUT DIR:         {output_dir_she}")
    logging.info(f"OPTIMIZED ORIGINS:  {use_optimized_origins}")
    logging.info("="*60 + "\n")

    logging.info("--- SHE Solver Driver ---")
    if not os.path.exists(input_npz_path):
        raise FileNotFoundError(f"ERROR: Input file '{input_npz_path}' not found.")
        
    parsed_data = load_and_parse_npz(input_npz_path)
    f_all = parsed_data['freqs']
    data_dict = parsed_data['complex_data']
    keys = parsed_data['filenames']
    r_static = parsed_data['r_arr']
    th_static = parsed_data['th_arr']
    ph_static = parsed_data['ph_arr']
    
    # 2. Optimized Origin Loading
    origins_mm = np.zeros((len(f_all), 3))
    if use_optimized_origins:
        if parsed_data['origins_mm'] is not None:
            origins_mm = parsed_data['origins_mm']
            logging.info(f"Loaded optimized origin coordinates for {len(origins_mm)} frequencies.")
        else:
            logging.warning(f"USE_OPTIMIZED_ORIGINS is True, but '{schema.ORIGINS_MM}' array not found in NPZ. Defaulting to (0,0,0).")

    # 3. Frequency Selection
    idx = np.where((f_all >= f_all[1]) & (f_all <= 24000.0))[0]
    f_sel = f_all[idx]
    origins_sel_mm = origins_mm[idx]
    K = len(idx)

    # 4. Matrix & Weight Prep
    N_grid, M_unique = get_grid_limit(th_static, ph_static)
    logging.info(f"Points: {len(keys)} (Unique: {M_unique}) | Grid Limit: {N_grid}")

    P = np.empty((K, len(keys)), complex)
    for m, fname in enumerate(keys):
        P[:, m] = data_dict[fname][idx]

    # 5. Task Building (Injecting dynamic translations per frequency)
    J_max = 2 * (target_n_max + 1) ** 2
    res_coeffs = np.zeros((K, J_max), complex)
    res_resid, res_cond, res_N = np.zeros(K), np.zeros(K), np.zeros(K, int)
    res_resid_vec = np.zeros((K, len(keys)), complex)

    logging.info(f"Solving {K} bins on {jobs} physical cores...")
    tasks = []
    for k in range(K):
        # Calculate translated spherical coordinates for this specific frequency bin
        origin_m = origins_sel_mm[k] / 1000.0
        r_k, th_k, ph_k = translate_coordinates(r_static, th_static, ph_static, origin_m)
        tasks.append((k, f_sel[k], P[k], r_k, th_k, ph_k, N_grid, worker_cfg))
    
    # 6. Execution
    with Pool(processes=jobs, initializer=_init_worker, initargs=(log_file_path,)) as pool:
        for k, cfs, resid, cond, n_val, r_vec, log_msg in pool.imap_unordered(_worker_wrapper, tasks):
            res_coeffs[k, :len(cfs)] = cfs
            res_resid[k], res_cond[k], res_N[k] = resid, cond, n_val
            res_resid_vec[k, :] = r_vec
            logging.info(log_msg)

    bn = np.linalg.norm(P, axis=1)
    pct_error = res_resid / np.maximum(bn, 1e-20) * 100.0

    # 7. H5 Save (Appending origins for inverse translation later)
    if save_to_disk:
        output_h5 = os.path.join(output_dir_she, output_filename_she)
        with h5py.File(output_h5, "w") as h:
            h[schema.FREQS] = f_sel
            h[schema.COEFFS] = res_coeffs
            h[schema.N_USED] = res_N
            h[schema.RESIDUAL] = res_resid
            h[schema.COND] = res_cond
            h[schema.ORIGINS_MM] = origins_sel_mm # Saved so reconstruction scripts can extract and shift observation coords
            h[schema.PCT_ERROR] = pct_error
            
        logging.info(f"Complete. Results saved to {output_h5}")

        # 8. Plot Fit Error and Condition Metrics
        if show_plot:
            plot_she_results(f_sel, pct_error, res_cond, res_N, condition_metrics, P, res_resid_vec, save_path_prefix=save_prefix)

    return {
        "coeffs": res_coeffs,
        "freqs": f_sel,
        "pct_error": pct_error,
        "cond": res_cond,
        "N_used": res_N,
        "origins_mm": origins_sel_mm,
        "residual": res_resid,
        "P_measured": P,
        "residual_vector": res_resid_vec
    }

def main() -> None:
    # Script execution configuration parsing happens here, completely isolated from module imports
    try:
        from config_process import (
            INPUT_DIR_SHE, INPUT_FILENAME_SHE, OUTPUT_DIR_SHE, OUTPUT_FILENAME_SHE,
            SPEED_OF_SOUND, TARGET_N_MAX, CONDITION_METRICS
        )
    except ImportError as e:
        sys.exit(f"Error: Could not import required module. {e}")

    try:
        from config_process import USE_MANUAL_TABLE, MANUAL_ORDER_TABLE
    except ImportError:
        USE_MANUAL_TABLE, MANUAL_ORDER_TABLE = False, {}

    try:
        from config_process import USE_OPTIMIZED_ORIGINS
    except ImportError:
        USE_OPTIMIZED_ORIGINS = False

    try:
        from config_process import NOISE_FLOOR_START_DB, NOISE_FLOOR_MAX_DB, MAX_LAMBDA
    except ImportError:
        NOISE_FLOOR_START_DB, NOISE_FLOOR_MAX_DB, MAX_LAMBDA = -30.0, -50.0, 0.000001

    try:
        from config_process import KR_OFFSET
    except ImportError:
        KR_OFFSET = 2.0

    p = argparse.ArgumentParser(description="Stage 2 – SHE Solver Driver.")
    p.add_argument("-j", "--jobs", type=int, default=max(1, cpu_count() // 2), help="number of parallel processes")
    args = p.parse_args()

    project_root = os.path.dirname(importlib.import_module("config_process").__file__)
    resolved_input_dir = os.path.join(project_root, INPUT_DIR_SHE)
    resolved_output_dir = os.path.join(project_root, OUTPUT_DIR_SHE)

    start_all = time.time()
    
    run_she_solve(
        input_filename_she=INPUT_FILENAME_SHE,
        output_filename_she=OUTPUT_FILENAME_SHE,
        input_dir_she=resolved_input_dir,
        output_dir_she=resolved_output_dir,
        target_n_max=TARGET_N_MAX,
        manual_order_table=MANUAL_ORDER_TABLE,
        use_manual_table=USE_MANUAL_TABLE,
        noise_floor_start_db=NOISE_FLOOR_START_DB,
        noise_floor_max_db=NOISE_FLOOR_MAX_DB,
        max_lambda=MAX_LAMBDA,
        condition_metrics=CONDITION_METRICS,
        use_optimized_origins=USE_OPTIMIZED_ORIGINS,
        save_to_disk=True,
        speed_of_sound=SPEED_OF_SOUND,
        kr_offset=KR_OFFSET,
        jobs=args.jobs,
        show_plot=True
    )

    logging.info("Total solve time: %.2f s", time.time() - start_all)

if __name__ == "__main__":
    main()