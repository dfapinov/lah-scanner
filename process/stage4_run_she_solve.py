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
    from utils import translate_coordinates, load_and_parse_npz
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

def _get_power_law_limit(f_hz: float, use_power_law_limit: bool, power_law_start_f: float, target_n_max: int) -> int:
    if not use_power_law_limit or f_hz < power_law_start_f:
        return 999
    
    n_calc = int(round(target_n_max * (f_hz / TARGET_MAX_F)**0.5))
    return min(n_calc, target_n_max)

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
        -> Tuple[int, np.ndarray, float, float, int]:
    k_idx, f, Pk, r, th, ph, M_unique, cfg = args

    kw = 2 * math.pi * f / cfg['speed_of_sound']
    kr = r * kw
    
    N_grid = int(math.floor(math.sqrt(M_unique / 2) - 1))
    N_kr = int(math.floor(kr.max() + 2))
    N_power = _get_power_law_limit(f, cfg['use_power_law_limit'], cfg['power_law_start_f'], cfg['target_n_max'])
    N_table = _get_table_limit(f, cfg['use_manual_table'], cfg['manual_order_table'])
    
    N_target = min(N_kr, N_grid, N_power, N_table, cfg['target_n_max'])

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
    elif cfg['use_manual_table'] and final_N == N_table and N_table < N_kr and N_table < N_grid and N_table < N_power:
        stop_reason = "Manual Table"
    elif cfg['use_power_law_limit'] and final_N == N_power and N_power < N_kr and N_power < N_grid and N_power < N_table:
        stop_reason = "Power Law"
    elif final_N == N_kr and N_kr < cfg['target_n_max']:
        stop_reason = "KR-Limit"
    elif final_N == N_grid and N_grid < cfg['target_n_max']:
        stop_reason = "Grid-Limit"
    else:
        stop_reason = "User Hard Cap Reached"

    base_norm = np.linalg.norm(Pk)
    pct_err = resid_norm / max(base_norm, 1e-20) * 100.0
    
    if cfg['condition_metrics']:
        logging.info("%6.2f Hz N=%d Resid=%.2f%% Pre-Cond=%.2e Post-Cond=%.2e -> %s", f, final_N, pct_err, metrics['cond_pre'], metrics['cond_post'], stop_reason)
    else:
        logging.info("%6.2f Hz N=%d Resid=%.2f%% -> %s", f, final_N, pct_err, stop_reason)

    return k_idx, coeffs, resid_norm, metrics['cond_post'], final_N

# =============================================================================
# --- Main Driver Logic ---
# =============================================================================
def run_she_solve(
    input_filename_she: str,
    output_filename_she: str,
    input_dir_she: str,
    output_dir_she: str,
    target_n_max: int,
    hf_fmax: float = 20000.0,
    use_power_law_limit: bool = True,
    power_law_start_f: float = 500.0,
    use_manual_table: bool = False,
    manual_order_table: dict = None,
    noise_floor_start_db: float = -30.0,
    noise_floor_max_db: float = -50.0,
    max_lambda: float = 0.000001,
    condition_metrics: bool = True,
    use_optimized_origins: bool = True,
    save_to_disk: bool = True,
    speed_of_sound: float = 343.0,
    jobs: int = None
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
        'use_power_law_limit': use_power_law_limit,
        'power_law_start_f': power_law_start_f,
        'condition_metrics': condition_metrics,
        'speed_of_sound': speed_of_sound,
        'noise_floor_start_db': noise_floor_start_db,
        'noise_floor_max_db': noise_floor_max_db,
        'max_lambda': max_lambda
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
    idx = np.where((f_all >= f_all[1]) & (f_all <= hf_fmax))[0]
    f_sel = f_all[idx]
    origins_sel_mm = origins_mm[idx]
    K = len(idx)

    # 4. Matrix & Weight Prep
    coords_stacked = np.column_stack((np.round(th_static, 4), np.round(ph_static, 4)))
    M_unique = len(np.unique(coords_stacked, axis=0))
    logging.info(f"Points: {len(keys)} (Unique: {M_unique}) | Grid Limit: {int(math.floor(math.sqrt(M_unique / 2) - 1))}")

    P = np.empty((K, len(keys)), complex)
    for m, fname in enumerate(keys):
        P[:, m] = np.conj(data_dict[fname][idx])

    # 5. Task Building (Injecting dynamic translations per frequency)
    J_max = 2 * (target_n_max + 1) ** 2
    res_coeffs = np.zeros((K, J_max), complex)
    res_resid, res_cond, res_N = np.zeros(K), np.zeros(K), np.zeros(K, int)

    logging.info(f"Solving {K} bins on {jobs} cores...")
    tasks = []
    for k in range(K):
        # Calculate translated spherical coordinates for this specific frequency bin
        origin_m = origins_sel_mm[k] / 1000.0
        r_k, th_k, ph_k = translate_coordinates(r_static, th_static, ph_static, origin_m)
        tasks.append((k, f_sel[k], P[k], r_k, th_k, ph_k, M_unique, worker_cfg))
    
    # 6. Execution
    with Pool(processes=jobs, initializer=_init_worker, initargs=(log_file_path,)) as pool:
        for k, cfs, resid, cond, n_val in pool.imap_unordered(_worker_wrapper, tasks):
            res_coeffs[k, :len(cfs)] = cfs
            res_resid[k], res_cond[k], res_N[k] = resid, cond, n_val

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
        plot_she_results(f_sel, pct_error, res_cond, condition_metrics)

    return {
        "coeffs": res_coeffs,
        "freqs": f_sel,
        "pct_error": pct_error,
        "cond": res_cond,
        "N_used": res_N,
        "origins_mm": origins_sel_mm,
        "residual": res_resid
    }

def main() -> None:
    # Script execution configuration parsing happens here, completely isolated from module imports
    try:
        from config_process import (
            INPUT_DIR_SHE, INPUT_FILENAME_SHE, OUTPUT_DIR_SHE, OUTPUT_FILENAME_SHE,
            HF_FMAX, SPEED_OF_SOUND, TARGET_N_MAX, CONDITION_METRICS
        )
    except ImportError as e:
        sys.exit(f"Error: Could not import required module. {e}")

    try:
        from config_process import USE_POWER_LAW_LIMIT, POWER_LAW_START_F
    except ImportError:
        USE_POWER_LAW_LIMIT, POWER_LAW_START_F = False, 500.0

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
        hf_fmax=HF_FMAX,
        use_power_law_limit=USE_POWER_LAW_LIMIT,
        power_law_start_f=POWER_LAW_START_F,
        use_manual_table=USE_MANUAL_TABLE,
        noise_floor_start_db=NOISE_FLOOR_START_DB,
        noise_floor_max_db=NOISE_FLOOR_MAX_DB,
        max_lambda=MAX_LAMBDA,
        condition_metrics=CONDITION_METRICS,
        use_optimized_origins=USE_OPTIMIZED_ORIGINS,
        save_to_disk=True,
        speed_of_sound=SPEED_OF_SOUND,
        jobs=args.jobs
    )

    logging.info("Total solve time: %.2f s", time.time() - start_all)

if __name__ == "__main__":
    main()