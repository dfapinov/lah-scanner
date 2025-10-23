#!/usr/bin/env python3
"""
Stage 4 – Generate band-limited IR WAVs from merged complex spectra
==================================================================

Description:
------------

This script takes merged complex frequency responses (from the SHE pipeline)
and reconstructs clean, time-aligned impulse responses (IRs). It reads the
measurement radius r_m to compute the physical time-of-flight (TOF), builds a
linear frequency grid sized by your IR gate, interpolates magnitude and phase from
log input data, applies LF/HF tapers, optionally enforces causal minimum-phase,
then IFFT-gates, normalizes, and saves 32-bit float WAV files.

Usage:
------

Configured via:  config_process.py

Run directly at the command line:
    python stage4_make_ir_from_complex.py

Input:
    • SRC_NPZ  – Path to a single *.npz or a directory of *.npz files containing:
                 freqs (Hz), P (complex spectrum), r_m (meters)

Output:
    • outputs/response_files/<stem>_ir.wav   – gated, normalized IR (32-bit float)
    • data/H_interp_44k_<stem>.npy           – positive-frequency complex spectrum (for diagnostics)
    • (optional) on-screen plots when ENABLE_PLOTS = True

Code Pipeline Overview
----------------------
 1. Load NPZ:
       • Read freqs (Hz), complex P(f), and r_m → compute TOF = r_m / SPEED_OF_SOUND.
       • Ensure a DC bin exists at f = 0.
 2. Frequency grid:
       • Set gate T from IR_GEN_GATE_MS → df = 1/T; build linear f grid up to input Nyquist.
       • Round bin count to a power of two for clean IFFT sizing.
 3. Magnitude:
       • Interpolate |H| on log-log scale via PCHIP.
       • Apply LF cosine-rise to MAG_LF_TAPER_START_HZ.
       • Apply HF cosine-fall from MAG_HF_TAPER_START_HZ to NYQ_OUT; zero ≥ NYQ_OUT.
 4. Phase:
       • Remove TOF (linear phase), unwrap, PCHIP-interpolate vs log-f, then restore TOF.
 5. Assemble spectrum:
       • H_pos = |H| * exp(j·phase).
       • (Optional) Rebuild H_pos as minimum-phase from |H| (cepstrum/Hilbert trick), then re-apply TOF.
 6. Trim & symmetry:
       • Keep bins ≤ FS_OUT/2.
       • Set DC/Nyquist real to satisfy real-IFFT; mirror to full Hermitian spectrum.
 7. IFFT & gating:
       • ir = IFFT(H_full); apply half-Hanning gate after the peak (FADE_RATIO).
       • (Optional) hard-trim to gate end; normalize peak to −6 dBFS.
 8. Save & visualize:
       • Write 32-bit float WAV per input; save H_interp_*.npy.
       • (Optional) show IR, group delay, phase-slope, and symmetry diagnostics.
"""

from __future__ import annotations  # Allow postponed evaluation of type annotations

import math  # Standard math functions (log2, ceil, etc.)
from pathlib import Path  # Filesystem path manipulation
import matplotlib.pyplot as plt  # Plotting library for graphs
import numpy as np  # Numerical array and matrix operations
from scipy.interpolate import PchipInterpolator  # Shape‐preserving spline interpolator
from scipy.io import wavfile  # Reading and writing WAV audio files (write only in this script)
from scipy.signal import group_delay  # Compute group delay of frequency response


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def half_hanning_fade(n: int, ratio: float = 1.0) -> np.ndarray:
    """Generate a half‐Hanning window: flat then cosine fade to zero."""
    if n < 2 or ratio <= 0:  # If too few samples or no fade requested
        return np.ones(n, float)  # Return all ones (no fade)
    p = max(1, min(int(round(ratio * n)), n))  # Fade length in samples
    idx = np.arange(p)  # Array [0, 1, …, p−1] for computing cosine
    fade = 0.5 * (1 + np.cos(np.pi * idx / (p - 1)))  # Cosine from 1→0
    return np.concatenate((np.ones(n - p, float), fade))  # Flat + fade portions


def apply_gate(sig: np.ndarray,
               fs: int,
               gate_ms: float,
               fade_ratio: float = 1.0) -> np.ndarray:
    """
    Apply a half‐Hanning gate after the signal peak.

    Keeps gate_ms milliseconds of data after peak and fades to zero.
    """
    gate_samps = int(round(gate_ms / 1000.0 * fs))  # Gate length in samples
    if gate_samps < 2:  # If too short to gate
        return sig  # Return original signal unchanged
    peak_idx = int(np.argmax(np.abs(sig)))  # Index of maximum absolute amplitude
    start = peak_idx  # Start gating at the peak
    end = min(len(sig), start + gate_samps)  # Ensure we don't exceed signal length
    win_len = end - start  # Number of samples within gate window
    window = half_hanning_fade(win_len, fade_ratio)  # Create half‐Hanning window
    out = sig.copy()  # Copy original signal to modify
    out[start:end] *= window  # Apply fade window to the gated region
    out[end:] = 0.0  # Zero everything after the gate window
    return out  # Return the gated signal


def write_frd(freq: np.ndarray,
              H: np.ndarray,
              fn: str,
              desc: str,
              *,
              phase_override: np.ndarray | None = None) -> None:
    """Save frequency response data as an FRD file."""
    eps = np.finfo(float).eps  # Tiny number to avoid log of zero
    mag_db = 20 * np.log10(np.abs(H) + eps)  # Convert magnitude to dB
    if phase_override is None:  # If no override provided
        ph_deg = np.angle(H, deg=True)  # Wrapped phase in degrees
    else:
        # Normalize overridden unwrapped phase into ±180° range
        ph_deg = (np.rad2deg(phase_override) + 180) % 360 - 180

    hdr = (  # Header lines for the FRD file
        f"# FRD generated ({desc})\n"
        "# freq(Hz)    magnitude(dB)    phase(deg)"
    )
    with open(fn, "w", encoding="utf-8") as fh:  # Open file to write
        np.savetxt(
            fh,
            np.column_stack([freq, mag_db, ph_deg]),  # Stack columns
            header=hdr,  # Include header at top
            fmt=("%.2f", "%.5f", "%.2f"),  # Format for each column
        )


# ─────────────────────────────────────────────────────────────────────────────
# User parameters
# ─────────────────────────────────────────────────────────────────────────────
# NEW: choose which merged complex NPZ to use as the source measurement
# It must contain keys: "freqs" (Hz) and "P" (complex spectrum).
# NEW: This script now reads r_m from the NPZ (key "r_m") and computes the time-of-flight.
from config_process import (
SRC_NPZ,                      # File or directory of .npz inputs (merged complex spectra)
FS_OUT,                       # Output sample rate for generated IR WAV
MAG_LF_TAPER_START_HZ,        # Start frequency of LF magnitude cosine-rise
MAG_HF_TAPER_START_HZ,        # Start frequency of HF magnitude cosine-fall
FADE_RATIO,                   # Fraction of gate used for the half-Hanning tail
IR_GEN_GATE_MS,               # Gate length (ms) used to set Δf = 1/T and time-gate IR
SPEED_OF_SOUND,               # Speed of sound (m/s) for TOF calculation
ENFORCE_CAUSAL_MINPHASE,      # If True, rebuild spectrum as minimum-phase from |H|
TRIM_TO_GATE,                 # If True, hard-trim WAV at gate end (peak + gate_ms)
ENABLE_PLOTS,                 # If True, show diagnostic plots
NORMALIZE_TO_DBFS,            # If True, scale output IR to target dBFS (default −6 dB)
TARGET_PEAK_DBFS,             # Desired output peak level in dBFS
)



# ── Directory or single-file support ─────────────────────────────────────────
src_path = Path(SRC_NPZ)
if src_path.is_dir():
    NPZ_PATHS = sorted(p for p in src_path.iterdir() if p.suffix.lower() == ".npz")
    if not NPZ_PATHS:
        raise FileNotFoundError(f"No .npz files found in directory: {src_path}")
else:
    NPZ_PATHS = [src_path]

for NPZ_FILE in NPZ_PATHS:
    print(f"\n=== Processing: {NPZ_FILE.name} ===")
    stem = NPZ_FILE.stem  # used to make per-file outputs

    # New minimal replacement: read the already-generated merged complex spectrum
    npz = np.load(NPZ_FILE)
    f_meas = np.array(npz["freqs"], dtype=float)          # Hz
    H_meas = np.array(npz["P"], dtype=np.complex128)      # complex spectrum

    # NEW: read spherical radial distance in metres from the NPZ to compute TOF.
    #      Expect key "r_m". TOF (seconds) = r_m / SPEED_OF_SOUND.
    if "r_m" not in npz:
        raise KeyError(f"{NPZ_FILE} does not contain 'r_m'. Add r_m to the NPZ metadata.")
    r_m = float(npz["r_m"])
    T0_S = r_m / SPEED_OF_SOUND  # Time-of-flight in seconds derived from r_m

    # Ensure a DC bin exists at index 0 (the rest of the pipeline expects it)
    if f_meas.size == 0 or f_meas[0] > 0.0:
        f_meas = np.concatenate(([0.0], f_meas))
        H_meas = np.concatenate(([0.0 + 0.0j], H_meas))


    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Build uniform linear frequency grid
    # ─────────────────────────────────────────────────────────────────────────
    # Use the NPZ's highest frequency as the "input Nyquist".
    NYQ_IN = float(np.max(f_meas))
    NYQ_OUT = FS_OUT * 0.5

    # One-knob control: use gate length to set FFT window (Δf = 1/T)
    T_des = IR_GEN_GATE_MS / 1000.0
    df_target = 1.0 / T_des

    # Number of positive-frequency bins 0..NYQ_IN inclusive
    N_pos = int(np.ceil(NYQ_IN / df_target)) + 1

    # (Optional) round up to next power of two for cleaner interpolation / FFT
    def next_pow2(n: int) -> int:
        return 1 << (max(1, int(n)) - 1).bit_length()

    N_pos = max(3, next_pow2(N_pos))  # ensure at least 3 points

    # Construct the uniform linear grid [0 .. NYQ_IN] with N_pos points.
    f_lin = np.linspace(0.0, NYQ_IN, N_pos)

    # Find indices for frequency breakpoints
    idx_lf_taper_start = np.searchsorted(f_lin, MAG_LF_TAPER_START_HZ)  # LF taper start
    idx_hf_start = np.searchsorted(f_lin, MAG_HF_TAPER_START_HZ)        # HF taper start
    idx_nyq_out = np.searchsorted(f_lin, NYQ_OUT)                       # Output Nyquist index



    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Magnitude interpolation and tapering
    # ─────────────────────────────────────────────────────────────────────────

    mag_meas = np.abs(H_meas)  # Extract magnitudes from complex measurement
    # Build PCHIP spline in log–log domain for smooth interpolation
    mag_spl = PchipInterpolator(np.log(f_meas[1:]), np.log(mag_meas[1:]))
    mag_i = np.empty_like(f_lin)  # Allocate array for interpolated magnitude
    mag_i[1:] = np.exp(mag_spl(np.log(f_lin[1:])))  # Fill from bin 1 onward
    mag_i[0] = 0.0  # Ensure DC magnitude is zero

    # Taper up to one bin before Nyquist, then hard-zero Nyquist and above
    k_nyq = np.searchsorted(f_lin, NYQ_OUT)
    k_last = max(idx_hf_start, k_nyq - 1)

    ten_hf = mag_i[idx_hf_start: k_last + 1]
    x_h = np.linspace(0, 1, len(ten_hf))
    mask_h = 0.5 * (1 + np.cos(np.pi * x_h))
    mag_i[idx_hf_start: k_last + 1] = ten_hf * mask_h

    # Explicitly zero Nyquist and anything above (safety)
    mag_i[k_nyq:] = 0.0


    # Low‐frequency taper: cosine fade from 0→1 up to taper start
    ten_lf = mag_i[: idx_lf_taper_start + 1]  # LF slice
    x_l = np.linspace(0, 1, idx_lf_taper_start + 1)  # Normalize 0→1
    mask_l = 0.5 * (1 - np.cos(np.pi * x_l))  # Cosine rising 0→1
    mag_i[: idx_lf_taper_start + 1] = ten_lf * mask_l  # Apply LF taper

    mag_i[0] = 0.0  # Ensure DC bin stays at zero


    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Phase interpolation
    # ─────────────────────────────────────────────────────────────────────────

    # Remove time‐of‐flight delay before unwrapping phase
    H_no_delay = H_meas * np.exp(-1j * 2 * np.pi * f_meas * T0_S)
    phase_unw = np.unwrap(np.angle(H_no_delay))  # Unwrapped phase in radians

    # Build spline for unwrapped phase vs log-frequency
    phase_spl = PchipInterpolator(np.log(f_meas[1:]), phase_unw[1:])
    phase_i = np.empty_like(f_lin)  # Allocate for interpolated phase
    phase_i[1:] = phase_spl(np.log(f_lin[1:]))  # Fill phase bins ≥1
    phase_i[0] = phase_spl(np.log(f_lin[1]))  # Extrapolate flat for DC bin


    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Construct full‐band complex response
    # ─────────────────────────────────────────────────────────────────────────

    H_pos = mag_i * np.exp(1j * phase_i)  # Magnitude & phase combine
    # Restore time‐of‐flight delay
    H_pos *= np.exp(-1j * 2 * np.pi * f_lin * T0_S)


    # ─────────────────────────────────────────────────────────────────────────
    # Step 5b (NEW): Optional causal (minimum-phase) enforcement via Hilbert/cepstrum
    # ─────────────────────────────────────────────────────────────────────────
    def minimum_phase_spectrum_from_mag(mag_pos: np.ndarray) -> np.ndarray:
        """
        Build a minimum-phase spectrum from a positive-frequency magnitude array.
        Method: real-cepstrum trick (equivalent to Hilbert transform of log|H|):
          • mirror magnitude to full Hermitian spectrum (even)
          • log() → IFFT → real cepstrum
          • zero negative-time quefrencies, double positive-time (keep DC; if even N, keep Nyquist)
          • FFT → exp() to get the complex minimum-phase spectrum
          • return positive-frequency part
        """
        eps = np.finfo(float).eps  # avoid log(0)
        # Construct full even magnitude (pos 0..Nyq, then mirror Nyq-1..1)
        mag_full = np.concatenate([mag_pos, mag_pos[-2:0:-1]])
        # Log magnitude spectrum
        log_mag = np.log(np.maximum(mag_full, eps))
        # Real cepstrum
        cep = np.fft.ifft(log_mag).real
        N = len(cep)
        cep_min = np.zeros_like(cep)
        cep_min[0] = cep[0]                # keep DC quefrency
        # Double positive quefrencies (1..N/2-1 if even; 1..(N-1)/2 if odd)
        half = N // 2
        if N % 2 == 0:
            cep_min[1:half] = 2.0 * cep[1:half]
            cep_min[half] = cep[half]     # keep Nyquist quefrency (even length case)
        else:
            cep_min[1:half+1] = 2.0 * cep[1:half+1]
        # Back to log spectrum and exponentiate
        log_H_min = np.fft.fft(cep_min)
        H_min_full = np.exp(log_H_min)
        # Return positive frequencies 0..Nyq
        return H_min_full[: len(mag_pos)]


    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Trim to output band and (optionally) replace with minimum-phase
    # ─────────────────────────────────────────────────────────────────────────

    trim_mask = f_lin <= NYQ_OUT  # Boolean mask ≤ output Nyquist
    f_lin_out = f_lin[trim_mask]  # Frequency axis for output
    H_pos_out = H_pos[trim_mask]  # Complex spectrum for output

    # If enforcing causal IR, rebuild the positive-frequency spectrum as minimum-phase
    # using the already-tapered magnitude, then re-apply the TOF delay so timing is correct.
    if ENFORCE_CAUSAL_MINPHASE:
        mag_pos_out = mag_i[trim_mask]
        H_min_pos = minimum_phase_spectrum_from_mag(mag_pos_out)
        # Re-apply physical TOF (pure linear phase) so final IR is causal and arrives at t = T0_S
        H_min_pos *= np.exp(-1j * 2 * np.pi * f_lin_out * T0_S)
        H_pos_out = H_min_pos  # override the interpolated-phase version

    # Enforce real-valued DC & Nyquist bins (real IFFT requirement)
    H_pos_out[0] = 0.0 + 0.0j          # DC exactly zero
    H_pos_out[-1] = 0.0 + 0.0j         # Nyquist exactly zero (or .real + 0j if you prefer not to zero)

    # Compute IFFT length consistent with the frequency grid
    N_FFT_OUT = (len(f_lin_out) - 1) * 2  # Real-valued IFFT (pos + neg freqs)
    T_actual = N_FFT_OUT / FS_OUT         # actual time window (s)
    print(f"IR window length (FFT window) = {T_actual*1000:.1f} ms")

    # Build symmetric spectrum for real IFFT
    H_full_out = np.concatenate([
        H_pos_out,
        np.conj(H_pos_out[-2:0:-1])  # Mirror excluding endpoints
    ])


    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: IFFT to time‐domain, apply gate, and (NEW) trim file length to gate
    # ─────────────────────────────────────────────────────────────────────────

    ir_out = np.fft.ifft(H_full_out, n=N_FFT_OUT).real  # Compute real‐valued IR

    # Apply half‐Hanning gate to remove alias‐prone tail
    ir_out = apply_gate(ir_out, FS_OUT, IR_GEN_GATE_MS, FADE_RATIO)

    # ── NEW: If requested, hard-trim the file to end exactly at the gate end ──
    if TRIM_TO_GATE:
        gate_samps = int(round(IR_GEN_GATE_MS / 1000.0 * FS_OUT))  # gate length in samples
        peak_idx = int(np.argmax(np.abs(ir_out)))                  # where the gate starts
        end = min(len(ir_out), peak_idx + gate_samps)              # end of gate window
        ir_out = ir_out[:end]                                      # trim to peak→gate end
        print(f"WAV duration (trimmed to gate) = {len(ir_out) / FS_OUT * 1000:.1f} ms")
    else:
        print(f"WAV duration (full FFT window) = {len(ir_out) / FS_OUT * 1000:.1f} ms")

    # Normalize so maximum absolute amplitude is -6 dBFS
    target_peak = 10 ** (-6.0 / 20.0)  # ≈ 0.501187
    peak = np.max(np.abs(ir_out)) + np.finfo(float).eps
    ir_out *= (target_peak / peak)

    # Save generated IR as 32‐bit float WAV (per source file)
    out_wav_path = Path(f"outputs/response_files/response_irs/{stem}_ir.wav")
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)  # Create new subfolder if needed
    wavfile.write(
        str(out_wav_path),
        FS_OUT,
        ir_out.astype(np.float32)
    )
    print(f"Saved IR to: {out_wav_path}")


    # ─────────────────────────────────────────────────────────────────────────
    # Step 8: Save diagnostics and (optionally) visualize results
    # ─────────────────────────────────────────────────────────────────────────

    # Export final FRD for IFFT input (per source file)
   # write_frd(
   #     f_lin_out,
   #     H_pos_out,
   #     f"response_for_ifft_{stem}.frd",
   #     "final response for IFFT"
   # )
    # Save complex spectrum to NumPy file (per source file)
   # np.save(f"data/H_interp_44k_{stem}.npy", H_pos_out)

    if ENABLE_PLOTS:
        # Plot the impulse response
        plt.figure()
        plt.plot(ir_out)
        plt.title(f"Impulse response – {stem}")
        plt.grid()

        # Compute and plot group delay
        w, gd = group_delay((np.abs(H_pos_out),
                             np.unwrap(np.angle(H_pos_out))),
                            fs=FS_OUT)
        plt.figure()
        plt.plot(w, gd)
        plt.title(f"Group delay – {stem}")
        plt.grid()

        # Plot phase‐slope (delta phase) vs frequency
        plt.figure()
        plt.plot(f_lin_out[1:], np.diff(phase_i[trim_mask]))
        plt.title(f"Phase‐slope Δφ – {stem}")
        plt.grid()

        # Plot Hermitian‐symmetry deviation
        plt.figure()
        plt.plot(np.abs(H_full_out - np.conj(H_full_out[::-1])))
        plt.title(f"Hermitian‐symmetry deviation – {stem}")
        plt.grid()

        # Show and immediately free figure memory for this file
        plt.show()
        plt.close('all')

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