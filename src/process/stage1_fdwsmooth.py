#!/usr/bin/env python3
"""
Frequency Dependent Windowing (FDW) and Complex Smoothing
=========================================================

This script processes a batch of Impulse Response (IR) .wav files using 
Frequency Dependent Windowing (FDW) and phase-aware Complex Smoothing.
The processed frequency responses are compiled into a single complex dataset
(<filename>.NPZ) for downstream Spherical Harmonic Expansion.

Usage
-----

Configured via:  config_process.py

Run from the command line:

    python stage1_fdwsmooth.py

Or import as a module:

    from stage1_fdwsmooth import fdwsmooth
    
Input Arguments:
    input_dir (str): Directory containing the raw .wav files.
    out_dir (str): Directory to save the output .npz files.
    output_filename (str): Name of the output .npz file.
    fdw_rft_ms (float): Reflection Free Time in milliseconds.
    fdw_oct_res (float): Target Octave Resolution.
    fdw_max_cap_ms (float): Maximum window length cap in milliseconds.
    enable_smoothing (bool): Toggle for applying complex smoothing.
    smoothing_oct_res (float): Octave resolution for smoothing.
    show_plot (bool): Launch the interactive FDW Viewer.
    save_to_disk (bool): Save results to .npz file.
    fdw_alpha_hf (float): High-frequency alpha taper (0.0 to 1.0).
    fdw_alpha_lf (float): Low-frequency alpha taper.
    fdw_f_min (float): Minimum frequency for plotting limits.
    fdw_windows_per_oct (int): Windows per octave to generate.
    peak_detect_threshold_db (float): Peak detection threshold.
    enable_auto_gain (bool): Auto-normalize batch to a target peak.
    target_peak_db (float): Target peak level if auto gain is enabled.
    keep_raw_and_smoothed (bool): Save both raw and smoothed files.

Returns:
    freqs (np.ndarray): Array of frequency bins.
    results_raw (dict): Raw complex frequency responses per file.
    results_smooth (dict): Smoothed complex frequency responses per file.
    meta (dict): Metadata (peaks, window sizes, schedules) per file.

Code Pipeline Overview
----------------------

1) SCAN: Reads all .wav files in the input directory, determines global max samples needed, and caches peaks.
2) SORT: Parses filenames using natural sorting to ensure deterministic ordering.
3) WINDOW CALCULATION: Determines frequency-dependent window lengths based on target octave resolution and reflection-free time.
4) PARALLEL PROCESSING: Distributes files and their cached peaks across multiple CPU cores.
5) FDW ANALYSIS: Windows the IR in the time domain, applies frequency-specific weights, and re-stitches in the frequency domain.
6) COMPLEX SMOOTHING: Optionally performs phase-aware Gaussian smoothing on the combined response.
7) OUTPUT: Saves the processed frequency bins, complex data, and metadata to an .npz archive, and launches an interactive visualizer.

"""

import os
import sys
import re
import time
import warnings
import numpy as np
import soundfile as sf
from scipy.fft import next_fast_len
import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import schema
from utils import natural_keys
from viewers import FDWViewer

from fdw_smoothing_core import (
    calculate_fdw_durations, get_earliest_significant_peak,
    process_single_ir, apply_complex_smoothing
)

# ─── 1. Core Logic Helper Functions ───────────────────────────────────────────

def load_and_prep_ir(file_path, crop_samples=None):
    """
    Loads a .wav file and extracts the first channel if multi-channel.
    Optionally crops to a specific sample length to save memory.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if crop_samples is not None:
            data, fs = sf.read(file_path, frames=crop_samples)
        else:
            data, fs = sf.read(file_path)
            
    if data.ndim > 1: 
        data = data[:, 0]
        
    return data, fs
# ─── 2. Parallel Worker ───────────────────────────────────────────────────────

def process_file_worker(file_info, ir_dir, gain, fs_common, n_fft, freqs, eff_smooth_res, crop_samples,
                        enable_smoothing, fdw_rft_ms, fdw_oct_res, fdw_max_cap_ms,
                        fdw_windows_per_oct, fdw_alpha_hf, fdw_alpha_lf):
    """
    Worker function to process a single IR file.
    Applies gain, zero padding to optimal FFT size, FDW analysis, and optional smoothing.
    """
    fname, t_peak_sec = file_info
    file_path = os.path.join(ir_dir, fname)
    
    # Load the specific crop size needed for the file based on its peak position
    data, _ = load_and_prep_ir(file_path, crop_samples)
    
    if len(data) < n_fft:
        data = np.pad(data, (0, n_fft - len(data)), mode='constant')
    elif len(data) > n_fft:
        data = data[:n_fft]
             
    # Perform frequency-dependent windowing (FDW)
    H_raw, m = process_single_ir(data * gain, fs_common, n_fft, freqs, t_peak_sec,
                                 fdw_rft_ms, fdw_oct_res, fdw_max_cap_ms,
                                 fdw_windows_per_oct, fdw_alpha_hf, fdw_alpha_lf)
    
    H_smooth = None
    if enable_smoothing:
        H_smooth = apply_complex_smoothing(freqs, H_raw, eff_smooth_res, m['t_peak'])
        
    return fname, H_raw, H_smooth, m

# ─── 3. Main Pipeline ─────────────────────────────────────────────────────────

def fdwsmooth(
    input_dir,
    out_dir,
    output_filename,
    fdw_rft_ms,
    fdw_oct_res,
    fdw_max_cap_ms,
    enable_smoothing,
    smoothing_oct_res,
    show_plot=False,
    save_to_disk=True,
    fdw_alpha_hf=0.2,
    fdw_alpha_lf=1.0,
    fdw_f_min=20.0,
    fdw_windows_per_oct=3,
    peak_detect_threshold_db=-12.0,
    enable_auto_gain=True,
    target_peak_db=-3.0,
    keep_raw_and_smoothed=False
):
    """
    Main pipeline function that orchestrates the scanning, parallel processing,
    and output consolidation of FDW and smoothed IR data.
    """
    start_time = time.time()

    ir_dir = input_dir
    wav_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith('.wav')], key=natural_keys)
    if not wav_files: 
        print("No .wav files found.")
        return None

    print("Scanning files to determine exact maximum crop length and caching peaks...")
    max_peak, fs_common = 0.0, None
    global_max_req_samples = 0
    peaks_cache = {}

    # ---- Pre-scan pass: determine peaks and the global maximum required sample length ----
    for f in wav_files:
        file_path = os.path.join(ir_dir, f)
        data, fs = load_and_prep_ir(file_path, None) 
        
        if fs_common is None:
            fs_common = fs
            max_window_sec = float(calculate_fdw_durations(fdw_f_min, fdw_oct_res, fdw_rft_ms, fdw_max_cap_ms))
            max_window_samples = int(max_window_sec * fs_common)

        # Identify the main impulse peak for causal windowing
        t_peak_idx = get_earliest_significant_peak(data, fs_common, peak_detect_threshold_db)
        t_peak_sec = t_peak_idx / fs_common
        peaks_cache[f] = t_peak_sec
        
        # Track the largest sample length required to fit the peak + maximum window
        req_samples = t_peak_idx + max_window_samples
        global_max_req_samples = max(global_max_req_samples, req_samples)
        
        # Keep track of global maximum amplitude for gain normalization
        max_peak = max(max_peak, np.max(np.abs(data)))
    
    n_fft = next_fast_len(global_max_req_samples)
    crop_samples = n_fft 
    freqs = np.fft.rfftfreq(n_fft, d=1/fs_common)
    
    print(f"Global max requirement: {global_max_req_samples} samples.")
    print(f"Optimal FFT size (next_fast_len): {n_fft} samples.")
    
    # Prepare worker settings
    gain = (10**(target_peak_db/20.0) / max_peak) if enable_auto_gain and max_peak > 0 else 1.0
    eff_smooth_res = smoothing_oct_res if smoothing_oct_res else fdw_oct_res
    num_workers = os.cpu_count() or 1
    
    print(f"Configuration: N_fft={n_fft}, Freq Bins={len(freqs)}, Gain={20*np.log10(gain):.2f}dB")
    print(f"Parallel Processing: Using {num_workers} processes.")

    results_raw, results_smooth, meta = {}, {}, {}
    
    worker_task = functools.partial(
        process_file_worker, 
        ir_dir=ir_dir, 
        gain=gain, 
        fs_common=fs_common, 
        n_fft=n_fft, 
        freqs=freqs, 
        eff_smooth_res=eff_smooth_res,
        crop_samples=crop_samples,
        enable_smoothing=enable_smoothing,
        fdw_rft_ms=fdw_rft_ms,
        fdw_oct_res=fdw_oct_res,
        fdw_max_cap_ms=fdw_max_cap_ms,
        fdw_windows_per_oct=fdw_windows_per_oct,
        fdw_alpha_hf=fdw_alpha_hf,
        fdw_alpha_lf=fdw_alpha_lf
    )

    print(f"Starting processing of {len(wav_files)} files...")
    
    file_infos = [(f, peaks_cache[f]) for f in wav_files]
    
    # ---- Parallel processing of individual files ----
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        for i, (fname, H_raw, H_smooth, m) in enumerate(executor.map(worker_task, file_infos)):
            sys.stdout.write(f"\rCompleted {i+1}/{len(wav_files)}: {fname}\033[K")
            sys.stdout.flush()
            
            results_raw[fname] = H_raw
            meta[fname] = m
            if H_smooth is not None:
                results_smooth[fname] = H_smooth

    print("\nProcessing complete.")

    if not enable_smoothing:
        plot_data = results_raw
        plot_smooth = None
        
    else:
        if keep_raw_and_smoothed:
            plot_data = results_raw
            plot_smooth = results_smooth
            
        else:
            plot_data = results_smooth 
            plot_smooth = None 

    # ---- Write to disk based on smoothing preferences ----
    if save_to_disk:
        print("Saving data to disk...")
        os.makedirs(out_dir, exist_ok=True)

        if not enable_smoothing:
            out_path = os.path.join(out_dir, output_filename)
            np.savez(out_path, **{
                schema.FREQS: freqs,
                schema.COMPLEX_DATA: results_raw,
                schema.META: meta
            })
            print(f"Saved RAW data to {out_path}")
            
        else:
            if keep_raw_and_smoothed:
                out_path_raw = os.path.join(out_dir, output_filename)
                np.savez(out_path_raw, **{
                    schema.FREQS: freqs,
                    schema.COMPLEX_DATA: results_raw,
                    schema.META: meta
                })
                print(f"Saved RAW data to {out_path_raw}")
                
                base, ext = os.path.splitext(output_filename)
                out_path_smooth = os.path.join(out_dir, f"{base}_smoothed{ext}")
                np.savez(out_path_smooth, **{
                    schema.FREQS: freqs,
                    schema.COMPLEX_DATA: results_smooth,
                    schema.META: meta
                })
                print(f"Saved SMOOTHED data to {out_path_smooth}")
                
            else:
                out_path = os.path.join(out_dir, output_filename)
                np.savez(out_path, **{
                    schema.FREQS: freqs,
                    schema.COMPLEX_DATA: results_smooth,
                    schema.META: meta
                })
                print(f"Saved SMOOTHED data to {out_path} (Raw discarded)")
    
    elapsed = time.time() - start_time
    print(f"\nStage 1 processing completed in {elapsed:.2f} seconds.")

    # Optional plotting of the resulting Frequency Domain representation
    if show_plot: 
        print("Opening Viewer...")
        FDWViewer(freqs, plot_data, meta, fs_common, ir_dir, crop_samples, data_dict_smooth=plot_smooth, fdw_f_min=fdw_f_min, fdw_rft_ms=fdw_rft_ms)

    return freqs, results_raw, results_smooth, meta

if __name__ == "__main__":
    # --- Import User Configuration ---
    from config_process import (
        INPUT_DIR_FDW, OUTPUT_DIR_FDW, OUTPUT_FILENAME_FDW,
        FDW_RFT_MS, FDW_OCT_RES, FDW_MAX_CAP_MS,
        FDW_ALPHA_HF, FDW_ALPHA_LF, FDW_F_MIN,
        FDW_WINDOWS_PER_OCT,
        ENABLE_AUTO_GAIN, TARGET_PEAK_DB,
        PLOT_OUTPUT, PEAK_DETECT_THRESHOLD_DB,
        ENABLE_SMOOTHING, SMOOTHING_OCT_RES, KEEP_RAW_AND_SMOOTHED
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, INPUT_DIR_FDW)
    out_dir = os.path.join(script_dir, OUTPUT_DIR_FDW)
    
    try: 
        fdwsmooth(
            input_dir=input_dir,
            out_dir=out_dir,
            output_filename=OUTPUT_FILENAME_FDW,
            fdw_rft_ms=FDW_RFT_MS,
            fdw_oct_res=FDW_OCT_RES,
            fdw_max_cap_ms=FDW_MAX_CAP_MS,
            enable_smoothing=ENABLE_SMOOTHING,
            smoothing_oct_res=SMOOTHING_OCT_RES,
            show_plot=PLOT_OUTPUT,
            save_to_disk=True,
            fdw_alpha_hf=FDW_ALPHA_HF,
            fdw_alpha_lf=FDW_ALPHA_LF,
            fdw_f_min=FDW_F_MIN,
            fdw_windows_per_oct=FDW_WINDOWS_PER_OCT,
            peak_detect_threshold_db=PEAK_DETECT_THRESHOLD_DB,
            enable_auto_gain=ENABLE_AUTO_GAIN,
            target_peak_db=TARGET_PEAK_DB,
            keep_raw_and_smoothed=KEEP_RAW_AND_SMOOTHED
        )
    except KeyboardInterrupt: print("\nInterrupted.")