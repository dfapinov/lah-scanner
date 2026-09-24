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
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from session_pool import borrow_pool
from pathlib import Path
from typing import List, Tuple, Union, Dict

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import h5py
import numpy as np
from scipy.special import spherical_jn, spherical_yn, sph_harm_y
import schema
from stage5_pressure_utils import (
    DEFAULT_IR_CAPTURE_PADDING_SAMPLES,
    apply_ir_padding_phase,
    get_min_phase_delay,
    get_tof_phasor,
)
from complex_to_ir_core import complex_to_ir

from utils import (
    spherical_to_cartesian, cartesian_to_spherical, translate_coordinates,
    hankel2, load_she_h5, apply_mic_calibration
)

# -------------------------------------------------
# User Adjustable Settings
# -------------------------------------------------
# Number of leading padding samples added during IR capture/deconvolution.
# Stage 5 subtracts this artificial delay from exported FRD/IR phase.
IR_CAPTURE_PADDING_SAMPLES = DEFAULT_IR_CAPTURE_PADDING_SAMPLES

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
    use_optimized_origins: bool = True,
    corr_ir_pad_phase: bool = True,
    ir_capture_padding_samples: int | None = None,
    use_process_pool: bool = True,
    process_pool=None,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    
    data = load_she_h5(she_input)
    freqs = data[schema.FREQS]
    coeffs = data[schema.COEFFS]
    n_used = data[schema.N_USED]
    fs_val = data.get(schema.FS)

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
    backend = "processes" if use_process_pool else "threads"
    if show_progress:
        print(f"Starting parallel solve with {num_cpus} {backend} ({num_pts} points)...")

    def consume_results(results_iter):
        total_tasks = len(tasks)
        for i, result in enumerate(results_iter):
            idx_range, p_chunk = result
            pressures_all[idx_range, :] = p_chunk
            if show_progress:
                percent = ((i + 1) / total_tasks) * 100
                sys.stdout.write(f"\rProgress: {percent:5.1f}% complete")
                sys.stdout.flush()

    if use_process_pool:
        if process_pool is None:
            process_pool = borrow_pool(num_cpus)
        if process_pool is not None:
            consume_results(process_pool.imap(func=_worker_calc_chunk, iterable=tasks))
        else:
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(processes=num_cpus) as pool:
                consume_results(pool.imap(func=_worker_calc_chunk, iterable=tasks))
    else:
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            consume_results(executor.map(_worker_calc_chunk, tasks))
            
    if show_progress:
        print("\nCalculation complete.")

    # -------------------------------------------------
    # Artificial Padding Phase Correction
    # -------------------------------------------------
    # The capture process splits the linear IR from the full Farina IR with
    # leading sample padding to avoid cutting off pre-ringing at the start of
    # the IR. This introduces an artificial delay into the entire system. We
    # mathematically remove that delay here by applying a phase advance,
    # ensuring all extracted responses (FRD and WAV) retain only their true
    # physical time-of-flight.
    if corr_ir_pad_phase:
        pad_samples = IR_CAPTURE_PADDING_SAMPLES if ir_capture_padding_samples is None else int(ir_capture_padding_samples)
        if pad_samples < 0:
            raise ValueError("IR capture padding samples must be zero or greater.")
        if fs_val is not None:
            fs_target = float(fs_val)
        else:
            fs_target = 44100.0 if freqs[-1] < 23000.0 else 48000.0
        if show_progress:
            print(f"Applying artificial padding phase correction (-{pad_samples} samples at {fs_target:.0f} Hz)...")
        pressures_all = apply_ir_padding_phase(pressures_all, freqs, pad_samples, fs_target)
    # -------------------------------------------------

    eps = np.finfo(float).eps
    mags_db = 20 * np.log10(np.abs(pressures_all) + eps)
    phase_deg = np.angle(pressures_all, deg=True)

    return {
        "freqs": freqs,
        "complex": pressures_all,
        "magnitude": mags_db,
        "phase": phase_deg,
        "fs": fs_val
    }


class PressureEvaluationSession:
    """Reusable Stage 5 evaluator with an optional persistent process pool.

    The coefficient data and worker processes live for the lifetime of the
    session, making repeated full-resolution preview evaluations inexpensive.
    Calls are serialized because a multiprocessing Pool should only have one
    active result consumer in this application.
    """

    def __init__(self, she_input, *, c_sound=None, use_process_pool=True, worker_count=None):
        self.data = load_she_h5(she_input)
        saved_speed = self.data.get(schema.SPEED_OF_SOUND_MPS)
        self.c_sound = float(c_sound if c_sound is not None else (saved_speed if saved_speed is not None else 343.0))
        self.use_process_pool = bool(use_process_pool)
        self.worker_count = int(worker_count or multiprocessing.cpu_count())
        self._lock = threading.Lock()
        self._closed = False
        self._pool = None
        if self.use_process_pool:
            ctx = multiprocessing.get_context('spawn')
            self._pool = borrow_pool(self.worker_count) or ctx.Pool(processes=self.worker_count)

    def evaluate_field(
        self,
        coords_sph,
        *,
        obs_mode="Internal",
        use_optimized_origins=True,
        corr_ir_pad_phase=True,
        ir_capture_padding_samples=None,
    ):
        with self._lock:
            if self._closed:
                raise RuntimeError("Pressure evaluation session is closed.")
            return evaluate_she_field(
                coords_sph=coords_sph,
                she_input=self.data,
                obs_mode=obs_mode,
                c_sound=self.c_sound,
                use_optimized_origins=use_optimized_origins,
                corr_ir_pad_phase=corr_ir_pad_phase,
                ir_capture_padding_samples=ir_capture_padding_samples,
                use_process_pool=self.use_process_pool,
                process_pool=self._pool,
                show_progress=False,
            )

    def evaluate_preview_response(
        self,
        coord_sph,
        *,
        obs_mode="Internal",
        use_optimized_origins=True,
        ir_capture_padding_samples=None,
        subtract_tof="Off",
        reference_coord_sph=None,
        reference_distance=None,
        apply_mic_cal=False,
        mic_cal_file=None,
        mic_cal_mode="subtract",
        mic_cal_fade_octaves=1.0,
        frd_db_offset=0.0,
    ):
        """Evaluate a full-resolution FRD and its physically timed IR.

        The selected-point IR is generated before physical TOF subtraction.
        The FRD phase correction and marker time use the separate reference
        coordinate (normally the on-axis observation).
        """
        reference_coord = coord_sph if reference_coord_sph is None else reference_coord_sph
        same_point = np.allclose(reference_coord, coord_sph)
        coords = [coord_sph] if same_point else [coord_sph, reference_coord]
        field = self.evaluate_field(
            coords,
            obs_mode=obs_mode,
            use_optimized_origins=use_optimized_origins,
            corr_ir_pad_phase=True,
            ir_capture_padding_samples=ir_capture_padding_samples,
        )
        freqs = np.asarray(field["freqs"], dtype=float)
        pressure_all = np.asarray(field["complex"], dtype=np.complex128)
        if apply_mic_cal:
            pressure_all = apply_mic_calibration(
                pressure_all,
                freqs,
                mic_cal_file,
                mic_cal_mode,
                float(mic_cal_fade_octaves),
            )

        selected_pressure = pressure_all[:, 0].copy()
        reference_pressure = selected_pressure if same_point else pressure_all[:, 1]
        mode = "Ref Origin" if subtract_tof is True else ("Off" if subtract_tof is False else str(subtract_tof))
        mode_lower = mode.lower()
        tof_distance = None
        if mode_lower == "ref origin":
            # Geometry-only mode: use the configured on-axis observation radius;
            # no response analysis or IR generation is needed to determine TOF.
            tof_distance = None if reference_distance is None else float(reference_distance)
        elif mode_lower == "ir peak":
            # IR Peak is the only mode that derives TOF in the time domain. Build
            # the full-resolution on-axis IR and use its earliest significant
            # peak; the selected off-axis IR never changes the batch reference.
            from fdw_smoothing_core import get_earliest_significant_peak

            ir_fs = float(field.get("fs") or (44100.0 if freqs[-1] < 23000.0 else 48000.0))
            reference_ir = complex_to_ir(reference_pressure, freqs, target_fs=ir_fs)
            peak_index = get_earliest_significant_peak(reference_ir, ir_fs, -12.0)
            tof_distance = float(peak_index / ir_fs * self.c_sound)
        elif mode_lower == "min phase ref":
            # Frequency-domain mode: estimate the on-axis response's robust
            # linear excess group delay relative to its minimum-phase response.
            tof_distance = get_min_phase_delay(reference_pressure, freqs, self.c_sound)
        # Off deliberately leaves tof_distance as None. Capture-padding phase
        # was still removed by evaluate_field; only physical TOF stays intact.

        frd_pressure = selected_pressure.copy()
        if tof_distance is not None:
            # Every point uses the same on-axis/reference delay so relative
            # inter-position timing remains intact throughout an observation set.
            frd_pressure *= get_tof_phasor(freqs, tof_distance, self.c_sound)

        ir_sample_rate = float(field.get("fs") or (44100.0 if freqs[-1] < 23000.0 else 48000.0))
        selected_ir = complex_to_ir(selected_pressure, freqs, target_fs=ir_sample_rate)
        ir_times = np.arange(len(selected_ir), dtype=float) / ir_sample_rate
        magnitude = 20.0 * np.log10(np.abs(frd_pressure) + np.finfo(float).eps) + float(frd_db_offset)
        return {
            "freqs": freqs,
            "complex": frd_pressure,
            "complex_pre_tof": selected_pressure,
            "magnitude": magnitude,
            "phase": np.angle(frd_pressure, deg=True),
            "frd_db_offset": float(frd_db_offset),
            "fs": field.get("fs"),
            "ir": selected_ir,
            "ir_times_s": ir_times,
            "tof_mode": mode,
            "tof_reference_distance": tof_distance,
            "tof_reference_time_s": None if tof_distance is None else float(tof_distance / self.c_sound),
        }

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
                self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
