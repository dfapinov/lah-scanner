#!/usr/bin/env python3

"""
Stage 3 – Spherical Harmonic Expansion (SHE) Solver with Adaptive Stop Criteria
===============================================================================


Description:
------------

Performs a least-squares solve to compute spherical harmonic coefficients
from two sets of impulse responses:

  • LF (long-gated)  – captures low-frequency content  
  • HF (short-gated) – captures high-frequency content

The process is parallelized for efficient CPU utilisation.

This solver supports aspherical input coordinates, allowing it to handle
measurement grids that are non-spherical or multi-layered — reducing
numerical artefacts that arise with uniform spherical sampling.

The harmonic order N is increased adaptively until one or more stopping
criteria are met, balancing numerical stability against spatial resolution.

Adaptive stop criteria are configured in config.py and may include:

  • k·r limit
  • Condition-number spike
  • Grid-density (guardrail)
  • User-defined maximum fallback order
  • Residual improvement plateau
  
Optional ridge regularisation (λ·I) stabilises the inversion when the
least-squares matrix becomes ill-conditioned.


Usage:
------

Configured via:  config_process.py

Run directly at command line:
    python stage2_she_config.py

Inputs:
    OUTPUT_DIR_REL/lf_irs/*.wav    – low-frequency gated IRs
    OUTPUT_DIR_REL/hf_irs/*.wav    – high-frequency gated IRs

Filename convention:
    (order ID, r_cyl_mm, phi_deg, z_mm)_LF.wav
    (order ID, r_cyl_mm, phi_deg, z_mm)_HF.wav
    
      • order ID  = order of measurment following path planner optimisation
      • r_cyl_mm  = radial distance in XY-plane (mm)
      • phi_deg   = azimuth angle (degrees, 0° = +X)
      • z_mm      = height (+Z up, mm)
      • p is used in place of decimal point
      
    Example: id87_r254_ph34p7_z334_LF.wav


Outputs:
  OUTPUT_DIR_REL/coefficients/
      asph_coeffs_LF.h5  – low-frequency coefficients
      asph_coeffs_HF.h5  – high-frequency coefficients

Each HDF5 file contains:
  • freqs           – frequency bins (Hz)
  • coeffs          – complex coefficient array
  • residual        – residual error per frequency
  • condition number– matrix conditioning metric
  • N_used          – adaptive harmonic order used
  

Code pipeline overview
----------------------

1) Load settings from config_process.py  
   Resolve paths (lf_irs, hf_irs, coefficients/) and frequency bands.

2) Parse input IR filenames  
   Extract (r_mm, φ_deg, z_mm) convert to spherical coordinates for math (r, θ, φ).

3) Build FFT grid and pressure matrix 
    Snap octave-spaced frequencies to FFT bins, compute RFFT for each IR.

4) Solve spherical harmonic coefficients (parallel cpu)  
   Incrementally increase order N until stop criteria trigger.
   Solve via least-squares or ridge regularisation.

5) Save results to HDF5  
   asph_coeffs_LF.h5 and asph_coeffs_HF.h5  
    
"""


from __future__ import annotations  # Enables forward type annotations for older Python versions

# --- Standard library imports ---
import argparse                     # For parsing command-line options (e.g., -j for jobs)
import logging                      # For formatted runtime logging
import math                         # For mathematical functions (pi, sqrt, floor, etc.)
import os                           # For filesystem paths and directory operations
import sys                          # For writing errors and exiting on fatal issues
import time                         # For timing how long the whole process takes
import importlib                    # To locate config.py regardless of working directory
from pathlib import Path            # (Not used heavily here) convenient path handling
from typing import Optional, Tuple  # Type hints for readability
import re                           # For parsing coordinates out of filenames

# --- Third-party imports ---
import h5py                         # To write results to HDF5 files
import numpy as np                  # Numerical arrays and linear algebra
import pandas as pd                 # Reading metadata CSVs
import soundfile as sf              # Loading WAV files (impulse responses)
from scipy.fft import rfft, rfftfreq                 # Fast Fourier Transform and FFT frequency bins
from scipy.special import spherical_jn, spherical_yn, sph_harm_y  # Spherical Bessel/Hankel + harmonics
from multiprocessing import Pool, cpu_count          # Parallel processing across CPU cores

# --- Project configuration: import user-tunable parameters from config_process.py ---
from config_process import (
    CROSSOVER_FREQ_HZ,          # Frequency that separates LF/HF processing bands
    OCT_RES,                    # Octave resolution (e.g., 24 for 1/24-oct spacing)
    RIDGE_LAMBDA,               # Ridge regularisation amount (0 = none, None = auto)
    RIDGE_COND_TARGET,          # Target conditioning for auto ridge selection
    SPEED_OF_SOUND,             # Speed of sound used to compute k = 2*pi*f/c
    USE_COND_SPIKE,             # Enable stopping on sudden condition-number spikes
    COND_SPIKE_FACTOR,          # Factor by which cond must jump to count as a spike
    USE_RESID_THRESHOLD,        # Enable residual-based plateau stopping rule
    RESID_IMPROVE_PCT,          # Minimum % improvement to keep increasing N
    RESID_ERROR_THRESHOLD_PCT,  # Only consider plateau after residual <= this %
    MAX_FALLBACK_N,             # Hard upper cap on harmonic order N
    USE_GRID_LIMIT,             # Limit N based on measurement count (sqrt(M)-1)
    USE_KR_LIMIT,               # Limit N based on max(kr)+2 rule
    N_MIN,                      # Do not apply stopping rules until N > N_MIN
    INPUT_IR_DIR_REL,           # Relative path to raw IRs folder (from project root)
    OUTPUT_DIR_REL,             # Relative path to outputs folder (from project root)
    LF_FMIN,                    # Absolute LF lower limit (Hz) from config
    HF_FMAX,                    # Absolute HF upper limit (Hz) from config
)

# --- Resolve input/output roots from config ---
project_root = os.path.dirname(                     # Take the directory that contains config.py ...
    importlib.import_module("config_process").__file__      # ... by importing it and reading its __file__ path
)
ir_folder = os.path.join(project_root, INPUT_IR_DIR_REL)   # Absolute path to raw IRs (not used here directly)
out_root  = os.path.join(project_root, OUTPUT_DIR_REL)     # Absolute path to outputs root
lf_dir    = os.path.join(out_root, 'lf_irs')               # Folder containing LF (long-gated) IR WAVs
hf_dir    = os.path.join(out_root, 'hf_irs')               # Folder containing HF (short-gated) IR WAVs
coeff_dir = os.path.join(out_root, 'coefficients')         # Folder where HDF5 coefficient files will be saved
os.makedirs(coeff_dir, exist_ok=True)                      # Ensure the coefficients folder exists (create if not)

# --- Utility: build the 1/N-octave centre frequencies over a range ---
def build_center_freqs(f_start=20.0, f_end=20000.0, oct_res=OCT_RES) -> np.ndarray:
    f = f_start                          # Start at the minimum frequency
    centres = []                         # Collect selected centre frequencies
    mul = 2 ** (1 / oct_res)             # Multiplicative step for 1/oct_res-octave spacing
    while f <= f_end * (1 + 1e-9):       # Loop until we pass the max (with small tolerance)
        centres.append(f)                # Add current centre frequency
        f *= mul                         # Step up by a constant ratio
    return np.array(centres)             # Return as a NumPy array

# --- Precompute band edges from crossover on the octave grid ---
CENTER_FREQS = build_center_freqs()                                              # Full grid of centre frequencies
LF_FMAX = CENTER_FREQS[np.argmin(np.abs(CENTER_FREQS - CROSSOVER_FREQ_HZ * 2))]  # LF band upper edge ≈ 2 × crossover
HF_FMIN = CENTER_FREQS[np.argmin(np.abs(CENTER_FREQS - CROSSOVER_FREQ_HZ * 0.5))]# HF band lower edge ≈ 0.5 × crossover
LF_FMIN, HF_FMAX = LF_FMIN, HF_FMAX                                             # Clamp absolute LF/HF limits

# --- Configure logging format and level (INFO is a good balance) ---
logging.basicConfig(
    level=logging.INFO,                                   # Show info, warnings, and errors
    format="%(message)s",     # message
)

# --- Math helpers used by the solver ---

def hankel1(n: int, z: np.ndarray) -> np.ndarray:
    """Compute spherical Hankel function of the first kind h_n^(1)(z) = j_n(z) + i y_n(z)."""
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)   # Combine spherical Bessel jn and yn with i

def snap_fft_grid(fmin, fmax, oct_res, fft_freqs):
    """
    Map octave-spaced target freqs to the nearest available FFT bins.
    Returns a sorted unique array of chosen FFT frequencies.
    """
    sel = []                                 # Selected FFT freqs
    f = fmin                                 # Start from lower band edge
    mul = 2 ** (1 / oct_res)                 # 1/oct_res-octave multiplier
    while f <= fmax * (1 + 1e-9):            # Step until upper band edge (with tolerance)
        # Find index of closest FFT frequency to current target f
        sel.append(float(fft_freqs[np.argmin(np.abs(fft_freqs - f))]))
        f *= mul                             # Move to next centre frequency
    return np.unique(sel)                    # Remove any duplicates and sort

def auto_lambda(s_max: float, s_min: float) -> float:
    """
    Choose a ridge regularisation value to hit a target condition number.
    lam = s_max / target_cond - s_min, floored at 0.
    """
    lam = s_max / RIDGE_COND_TARGET - s_min  # Formula derived from (s_max + lam)/(s_min + lam) ≈ target
    return max(0.0, lam)                     # Do not allow negative regularisation

# --- New: helper to parse your filename style --------------------------------
_ph_pat = re.compile(
    r"""
    ^.*?                # anything up front (e.g., id2_)
    _r(?P<rmm>-?\d+)    # r in millimetres, integer
    _ph(?P<ph>[-np\d]+) # phi in degrees with 'p' as decimal, optional leading 'n' for negative
    _z(?P<zmm>-?\d+)    # z in millimetres, integer (can be negative)
    (?:_.+)?$           # rest of the name (e.g., _mic_conditioned)
    """,
    re.VERBOSE | re.IGNORECASE,
)

def _parse_coords_from_filename(fname: str) -> tuple[float, float, float, str]:
    """
    Parse r_xy_mm, phi_deg(with 'p' as decimal), z_mm from a filename like:
      id2_r244_ph4p7_z83_mic_conditioned_HF.wav

    Returns (r_sph_m, theta_rad, phi_rad, base_wav_name)
    where base_wav_name has the band suffix replaced with '.wav' for internal use.
    """
    # Base name used throughout the solver: swap band suffix with ".wav"
    if fname.endswith("_LF.wav"):
        base_name = fname[:-len("_LF.wav")] + ".wav"
    elif fname.endswith("_HF.wav"):
        base_name = fname[:-len("_HF.wav")] + ".wav"
    else:
        base_name = fname[:-4] + ".wav"  # generic fallback

    m = _ph_pat.match(fname)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: '{fname}'")

    r_mm = float(m.group("rmm"))
    ph_str = m.group("ph")
    z_mm = float(m.group("zmm"))

    # Decode phi: '4p7' -> 4.7 ; optional leading 'n' → negative
    def _decode_phi(s: str) -> float:
        neg = s.startswith("n")
        s = s[1:] if neg else s
        s = s.replace("p", ".")
        val = float(s)
        return -val if neg else val

    phi_deg = _decode_phi(ph_str)

    # Convert cylindrical to spherical (SciPy convention)
    r_cyl_m = r_mm / 1000.0
    z_m = z_mm / 1000.0
    r_sph = math.hypot(r_cyl_m, z_m)
    # theta = arctan(r_xy / z); handle z=0 separately; clamp to [0, pi]
    if z_m == 0.0:
        theta = math.pi / 2
    else:
        theta = math.atan2(r_cyl_m, z_m)  # robust to signs
        if theta < 0:
            theta += math.pi
    phi = math.radians(phi_deg) % (2 * math.pi)

    return r_sph, theta, phi, base_name

# --- Core worker: solve for one frequency independently (runs in parallel) ---
def _solve_one_frequency(args: Tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) \
        -> Tuple[int, np.ndarray, float, float, int]:
    """
    Solve for a single frequency index.
    Returns (k, best_coeffs, best_resid, best_cond, best_N).
    """
    k, f, Pk, r, th, ph = args               # Unpack arguments: index, frequency, pressures, and spherical coords
    recon = np.zeros_like(Pk, complex)       # Running reconstruction of Pk as we add orders

    # Baseline norm for absolute residual percentages
    base_norm = np.linalg.norm(Pk)

    best_resid = base_norm                   # Start with worst-case residual (no reconstruction)
    best_cond: Optional[float] = None        # Best condition number seen so far
    best_N = -1                              # Best order chosen so far
    resid_prev = best_resid                  # Track previous residual to compute improvement
    cond_prev: Optional[float] = None        # Track previous condition number for spike detection
    reason_desc: Optional[str] = None        # Textual reason for stopping (for logging)
    stop_reason = "User Hard Cap Reached"    # Default reason (overwritten by rules)

    # --- Precompute wavenumber and kr for this frequency ---
    kw = 2 * math.pi * f / SPEED_OF_SOUND    # k = 2πf / c   (radians per metre)
    kr = r * kw                              # Element-wise product → kr at each mic position

    # --- Limits on maximum N based on rules from config.py ---
    N_kr   = int(math.floor(kr.max() + 2)) if USE_KR_LIMIT   else MAX_FALLBACK_N   # max(kr)+2 rule
    N_grid = int(math.floor(math.sqrt(len(r)) - 1)) if USE_GRID_LIMIT else MAX_FALLBACK_N  # sqrt(M)-1 rule
    N_max  = min(N_kr, N_grid, MAX_FALLBACK_N)             # Hard cap by the smallest limit

    A_cols: list[np.ndarray] = []          # Will store columns of the design matrix A incrementally
    coeff_list: list[complex] = []         # Accumulate solved coefficients as N increases

    # --- Increment N from 0 up to N_max, stopping when criteria fire ---
    for N in range(N_max + 1):
        # Build new columns for this order N only (append to the growing matrix)
        hn, jn = hankel1(N, kr), spherical_jn(N, kr)        # Outgoing (hn) and regular (jn) radial functions
        for m_ord in range(-N, N + 1):                      # For each azimuthal order m in [-N..N]
            Y = sph_harm_y(N, m_ord, th, ph)                # Spherical harmonic basis Y_N^m(theta, phi)
            A_cols.extend([hn * Y, jn * Y])                 # Two columns per (N,m): outgoing and regular parts
        A_d = np.column_stack(A_cols[-2*(2*N + 1):])        # Stack only the new columns for this N

        # --- Solve the incremental least-squares / ridge step for the new columns ---
        if RIDGE_LAMBDA == 0:
            # Plain least-squares solve of A_d * x ≈ (Pk - recon)
            x, *_ = np.linalg.lstsq(A_d, Pk - recon, rcond=1e-6)
        else:
            # Ridge-regularised solve: (A^H A + lam I) x = A^H b
            if RIDGE_LAMBDA is None:
                s = np.linalg.svd(A_d, compute_uv=False)    # Singular values for auto-lambda
                lam = auto_lambda(s[0], s[-1])              # Choose lam to target a desired conditioning
            else:
                lam = RIDGE_LAMBDA                          # Use fixed lam from config
            G = A_d.conj().T @ A_d + lam * np.eye(A_d.shape[1])  # Regularised normal matrix
            x = np.linalg.solve(G, A_d.conj().T @ (Pk - recon))  # Solve for new coefficients

        coeff_list.extend(x)                # Append newly solved coefficients to the full list
        recon += A_d @ x                    # Update reconstruction with current order N contribution

        resid_norm = np.linalg.norm(Pk - recon)  # Compute residual magnitude after adding this order

        # --- Estimate condition number for monitoring (plain or ridge) ---
        if RIDGE_LAMBDA == 0:
            s = np.linalg.svd(np.column_stack(A_cols), compute_uv=False)  # SVD of full matrix so far
            cond_now = s[0] / s[-1] if s[-1] > 0 else np.inf              # Ratio of largest to smallest singular value
        else:
            s = np.linalg.svd(G, compute_uv=False)                        # SVD of regularised system matrix
            cond_now = s[0] / s[-1] if s[-1] > 0 else np.inf

        # --- Apply stopping rules only after we pass a minimal order N_MIN ---
        plateau = False
        spike = False
        if N > N_MIN:
            # Residual plateau rule: gate by absolute residual vs baseline ||Pk||
            if USE_RESID_THRESHOLD:
                abs_resid_pct = (resid_norm / max(base_norm, 1e-20)) * 100.0
                if abs_resid_pct <= RESID_ERROR_THRESHOLD_PCT:
                    improve_pct = (resid_prev - resid_norm) / max(resid_prev, 1e-20) * 100.0  # % improvement vs previous step
                    plateau = improve_pct < RESID_IMPROVE_PCT                                # Stop if improvement too small
            # Condition spike rule: stop if condition number jumps sharply
            if USE_COND_SPIKE and cond_prev is not None and cond_now / cond_prev > COND_SPIKE_FACTOR:
                spike = True

        # --- If any rule fired, roll back to the previous accepted state and stop ---
        if plateau or spike:
            if spike:
                reason_desc = f"Matrix Condition Grew >x{COND_SPIKE_FACTOR:.1f} (x{cond_now/cond_prev:.1f})"  # ASCII 'x' for safety
            stop_reason = "Residual% Improvement Below Threshold" if plateau else (reason_desc or stop_reason)
            coeff_list = best_coeffs.copy()                                 # Revert coefficients to last best
            resid_norm = best_resid                                         # Revert residual
            cond_now = best_cond                                            # Revert condition number
            N = best_N                                                      # Revert order
            break                                                           # Exit the N loop

        # --- Accept this order as the new best and continue ---
        best_coeffs = coeff_list.copy()     # Save a snapshot of coefficients up to this order
        best_resid, best_cond, best_N = resid_norm, cond_now, N   # Save metrics
        resid_prev, cond_prev = resid_norm, cond_now              # Update previous-step trackers

    # --- If we exited because we hit the limit, record which limit applied ---
    if best_N == N_max:
        if USE_KR_LIMIT and N_max == N_kr and N_kr < MAX_FALLBACK_N:
            stop_reason = "KR-Limit Reached"                             # Reached the kr-based bound
        elif USE_GRID_LIMIT and N_max == N_grid and N_grid < MAX_FALLBACK_N:
            stop_reason = "Grid-Limit Reached"                           # Reached the grid-based bound
        else:
            stop_reason = "User Hard Cap Reached"

    # --- Log a compact summary line for this frequency ---
    logging.info("%6.2f Hz Harmonic Order=%d  Residual=%.2f%%  Matrix Condition=%.2e Finished!   %s",
                 f, best_N, best_resid / max(base_norm, 1e-20) * 100.0, best_cond or 0.0, stop_reason)

    # --- Return results for aggregation in the parent process ---
    return k, np.array(best_coeffs, dtype=complex), best_resid, best_cond or 0.0, best_N


# --- High-level routine: prepare inputs, run frequency-wise solves in parallel, save outputs ---
def run_solver(*, label, meta_path, ir_dir, ir_suffix, fmin, fmax,
               output_h5, jobs: int) -> None:
    """
    label     : "LF" or "HF" (just for logs/progress)
    meta_path : path to metadata.csv (if filename parsing is off or fails)
    ir_dir    : directory containing the gated IR WAVs (_LF.wav or _HF.wav)
    ir_suffix : file suffix to look for (e.g., "_LF.wav")
    fmin/fmax : frequency bounds for this band
    output_h5 : path to write the resulting HDF5 file
    jobs      : number of parallel worker processes
    """
    logging.info("── %s ── [%g–%g Hz]", label, fmin, fmax)   # Announce which band we’re solving

    # --- Option 1: parse coordinates directly from filenames like "(r_mm, phi_deg, z_mm).wav" ---
    # (Always-on filename parsing for new pattern: id2_r244_ph4p7_z83_mic_conditioned_{LF,HF}.wav)
    if not os.path.isdir(ir_dir):                        # Validate that the IR directory exists
        sys.stderr.write(f"ERROR: IR directory '{ir_dir}' does not exist or is not a directory.\n")
        sys.exit(1)

    # Gather all files that end with the given suffix (e.g., "_LF.wav")
    files_on_disk = sorted([
        fn for fn in os.listdir(ir_dir)
        if os.path.isfile(os.path.join(ir_dir, fn)) and fn.endswith(ir_suffix)
    ])
    if len(files_on_disk) == 0:                          # Fail fast if no files found
        sys.stderr.write(f"ERROR: No files ending with '{ir_suffix}' found in '{ir_dir}'.\n")
        sys.exit(1)

    # Prepare containers for parsed coordinates and filenames
    filenames: list[str] = []
    r_list: list[float] = []
    th_list: list[float] = []
    ph_list: list[float] = []

    # Parse each filename according to the new pattern
    for suff_name in files_on_disk:
        try:
            r_sph, theta, phi, base_name = _parse_coords_from_filename(suff_name)
        except Exception as e:
            sys.stderr.write(f"ERROR: {e}\n")
            sys.exit(1)

        filenames.append(base_name)
        r_list.append(r_sph)
        th_list.append(theta)
        ph_list.append(phi)

    # Build arrays for solver
    r  = np.array(r_list)                            # Radii (metres)
    th = np.array(th_list)                           # Polar angles (radians)
    ph = np.array(ph_list)                           # Azimuth angles (radians)
    files = np.array(filenames, dtype=str)           # Corresponding .wav basenames
    M = len(files)                                   # Number of measurement positions
    logging.info("Parsed %d coordinates directly from filenames.", M)

    # --- Determine FFT size/frequencies from the first IR (to set up P and f grid) ---
    example_file = files[0]                                  # Use the first file name as example
    first_ir_path = os.path.join(ir_dir, example_file.replace(".wav", ir_suffix))  # Replace .wav with desired suffix
    if not os.path.isfile(first_ir_path):                    # Ensure the file exists before reading
        sys.stderr.write(f"ERROR: Expected IR file '{first_ir_path}' not found. Exiting.\n")
        sys.exit(1)

    first_ir, fs = sf.read(first_ir_path)                    # Load the audio file → samples and sample rate
    Nfft = 1 << math.ceil(math.log2(len(first_ir)))          # Next power-of-two FFT size (fast and >= len)
    f_fft = rfftfreq(Nfft, 1 / fs)                           # Real-FFT frequency axis for this FFT size

    # --- Select the FFT bin indices closest to our octave-spaced target frequencies ---
    f_sel = snap_fft_grid(fmin, fmax, OCT_RES, f_fft)        # Map target centres to actual FFT frequencies
    idx = [int(np.argmin(np.abs(f_fft - f))) for f in f_sel] # Index of nearest FFT bin per target freq
    K = len(f_sel)                                           # Number of frequency points we will solve

    # --- Compute FFTs of all IRs at the selected bins to form P (K×M complex matrix) ---
    P = np.empty((K, M), complex)                            # Allocate complex pressure matrix
    bar = 40                                                 # Progress bar width
    sys.stdout.write(f"[{label}] FFTs: 0/{M}")               # Initial progress message

    for m, fname in enumerate(files, start=1):               # Iterate over all measurement positions
        ir_path = os.path.join(ir_dir, fname.replace(".wav", ir_suffix))  # Path to the staged IR file
        if not os.path.isfile(ir_path):                      # Validate the file exists
            sys.stderr.write(f"ERROR: IR file '{ir_path}' not found. Exiting.\n")
            sys.exit(1)

        buf = np.zeros(Nfft)                                 # Zero buffer for padding to Nfft
        data, _ = sf.read(ir_path)                           # Load the IR samples (mono expected)
        buf[:len(data)] = data                               # Copy IR into the start; remainder stays zero
        P[:, m-1] = rfft(buf)[idx]                           # Take RFFT and sample only the needed bins
        # Update the simple text progress bar
        done = "#" * int(bar * m / M) + "-" * (bar - int(bar * m / M))
        sys.stdout.write(f"\r[{label}] FFTs: [{done}] {m}/{M}")
    print()                                                  # Newline after progress bar is complete

    # --- Prepare output arrays to store best results per frequency ---
    J_max = 2 * (MAX_FALLBACK_N + 1) ** 2                    # Maximum number of coeffs (both outgoing/regular)
    coeffs = np.zeros((K, J_max), complex)                   # Coefficient matrix (pad with zeros past best length)
    residual = np.zeros(K)                                   # Residual magnitudes per frequency
    cond = np.zeros(K)                                       # Condition numbers per frequency
    N_used = np.zeros(K, int)                                # Best order N chosen per frequency

    # --- Parallel solve: dispatch one frequency per task across the worker pool ---
    with Pool(processes=jobs) as pool:                       # Create a pool with 'jobs' worker processes
        tasks = [(k, f_sel[k], P[k], r, th, ph) for k in range(K)]   # Build task tuples for all K freqs
        # imap_unordered yields results as soon as each task finishes (order not guaranteed)
        for k, best_coeffs, best_resid, best_cond, best_N in pool.imap_unordered(_solve_one_frequency, tasks):
            coeffs[k, :len(best_coeffs)] = best_coeffs       # Store the variable-length coeffs into fixed row
            residual[k] = best_resid                         # Store residual for this frequency
            cond[k] = best_cond                              # Store condition number
            N_used[k] = best_N                               # Store chosen order

    # --- Save all results to an HDF5 file for later stages/visualisation ---
    output_h5_parent = os.path.dirname(output_h5)            # Parent folder of the output file
    os.makedirs(output_h5_parent, exist_ok=True)             # Ensure it exists
    with h5py.File(output_h5, "w") as h:                     # Open/create the HDF5 file for writing
        h["freqs"] = f_fft[idx]                              # Frequencies we actually solved
        h["coeffs"] = coeffs                                 # Complex coefficients (rows align with freqs)
        h["N_used"] = N_used                                 # Best harmonic order per frequency
        h["residual"] = residual                             # Residual magnitudes
        h["pct_error"] = residual / np.linalg.norm(P, axis=1) * 100  # Residual as % of input norm
        h["cond"] = cond                                     # Condition numbers
    logging.info("[%s] results → %s", label, output_h5)      # Log where the file was written

# --- Command-line argument parsing (e.g., --meta and -j/--jobs) ---
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 – SHE with adaptive N.")  # Create a parser with a short description
    # Default metadata path is <project_root>/data/metadata.csv (matches Stage 1 style)
    default_meta = os.path.join(project_root, "data", "metadata.csv")          # Build default path to metadata.csv
    p.add_argument("--meta", default=default_meta, type=str,                   # Allow override via --meta <path>
                   help="metadata CSV")
    p.add_argument(
        "-j", "--jobs", type=int,                                             # Number of parallel worker processes
        default=max(1, cpu_count() // 2),                                     # Default to half your CPU cores (>=1)
        help="number of frequencies to solve in parallel (default: physical cores)"
    )
    return p.parse_args()                                                     # Parse and return the args object

# --- Program entry point: run LF and HF passes and report timing ---
def main() -> None:
    args = parse_args()                                 # Read command-line options (meta path, jobs)
    start_all = time.time()                             # Start a stopwatch for the whole run

    # --- Low-frequency pass (long-gated IRs) ---
    run_solver(label="LF",
               meta_path=args.meta,                     # Path to metadata.csv (or filename parsing if enabled)
               ir_dir=lf_dir,                           # Input directory containing *_LF.wav files
               ir_suffix="_LF.wav",                     # Suffix that identifies LF WAVs
               fmin=LF_FMIN,                            # Lower frequency bound for LF
               fmax=LF_FMAX,                            # Upper frequency bound for LF (≈ 2 × crossover)
               output_h5=os.path.join(coeff_dir, "asph_coeffs_LF.h5"),  # Output file path
               jobs=args.jobs)                          # Number of worker processes

    # --- High-frequency pass (short-gated IRs) ---
    run_solver(label="HF",
               meta_path=args.meta,                     # Same metadata or filename parsing strategy
               ir_dir=hf_dir,                           # Input directory containing *_HF.wav files
               ir_suffix="_HF.wav",                     # Suffix that identifies HF WAVs
               fmin=HF_FMIN,                            # Lower frequency bound for HF (≈ 0.5 × crossover)
               fmax=HF_FMAX,                            # Upper frequency bound for HF
               output_h5=os.path.join(coeff_dir, "asph_coeffs_HF.h5"),  # Output file path
               jobs=args.jobs)                          # Number of worker processes

    total = time.time() - start_all                     # Compute total elapsed time
    logging.info("Total solve time: %.2f s", total)     # Log the time nicely
   # print(f"Total solve time: {total:.2f} s")           # Also print to stdout for quick visibility

# --- Run main() only when executed as a script (not when imported) ---
if __name__ == "__main__":
    main()                                              # Kick off the pipeline

r"""
       ____  __  __ ___ _____ ______   __   
      |  _ \|  \/  |_ _|_   _|  _ \ \ / /   
      | | | | |\/| || |  | | | |_) \ V /    
      | |_| | |  | || |  | | |  _ < | |     
     _|____/|_| _|_|___| |_| |_|_\_\|_|   __
    |  ___/ \  |  _ \_ _| \ | |/ _ \ \   / /
    | |_ / _ \ | |_) | ||  \| | | | \ \ / / 
    |  _/ ___ \|  __/| || |\  | |_| |\ V /  
    |_|/_/   \_\_|  |___|_| \_|\___/  \_/   
                       
                     ███                 
                   █████         ███     
                 ███████         ████    
               █████████    ██    ████   
     ███████████████████    ████   ████  
    ████████████████████     ███   ████  
    ████████████████████      ██    ████ 
    ████████████████████      ███    ████ 
    ████████████████████      ███    ████ 
    ████████████████████     ███    ████  
     ███████████████████    ████   ████  
              ██████████    ███   ████   
                ████████    █    ████    
                  ██████         ███     
                     ███    
"""