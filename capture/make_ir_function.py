#!/usr/bin/env python3 

"""
Impulse Response Builder – Regularized Sweep Deconvolution
==========================================================

Description:
============
This script reconstructs an impulse response (IR) by processing a recorded
microphone sweep using the original digital excitation signal for deconvolution,
in accordance with the Farina method for harmonic distortion analysis.
This approach ensures that only the true linear system response is recovered,
while harmonic and intermodulation distortion components appear at predictable
time offsets in the resulting IR.

Usage:
======

Configured via:  config_capture.py

Run directly from a terminal:
    python make_ir_function.py

→ Searches for "excitation_signal.wav" and all matching
  "<base>_mic_conditioned.wav" files in OUTDIR.
  For each valid measurement pair, the script outputs a
  gated impulse response "<base>_ir.wav".

----

Import and call the processing function directly:

    from make_ir_function import make_ir_from_pair
    h = make_ir_from_pair(mic_path, excitation_path)

This returns the computed IR as a NumPy float32 array (TOF preserved, gated).

Code Pipeline Overview
----------------------
Scans OUTDIR for matched sweep recordings:
  "*_mic_conditioned.wav" – microphone response
  "excitation_signal.wav" – original excitation sweep used for deconvolution

For each pair, the script:
  1. Crops the sweep window.
  2. Performs regularized frequency-domain deconvolution
     (Kirkeby-style epsilon) using the *digital excitation* as reference.
  3. Optionally applies LF/HF tapers, preserves time-of-flight,
     and gates the tail with a half-Hanning fade
     (duration = 4 / GATE_FREQ_HZ seconds).
  4. Writes "<base>_ir.wav" – the measured system impulse response.
  
The resulting impulse response represents the system transfer function
from the digital excitation to the measured acoustic response,
consistent with Farina’s swept-sine analysis framework.
"""


import re  # Regular expressions used to match and strip suffixes from filenames
from pathlib import Path  # Filesystem paths (OS-agnostic handling)
import numpy as np  # Numerical arrays and FFT operations
import soundfile as sf  # Reading/writing WAV files
import matplotlib.pyplot as plt  # optional plotting  # Plotting library (only used when PLOT is True)
from typing import Tuple, Optional  # Type hints for readability

from config_capture import(  # Import measurement and processing parameters from a single config source
# From Sweep_function.py settings:
FS,  # Sampling rate in Hz for all files and processing
F1_HZ,  # Sweep start frequency in Hz (low end)
F2_HZ,  # Sweep end frequency in Hz (high end); can be None → uses 0.48*FS
PRE_SIL_MS,  # Pre-roll silence before the sweep (ms)
SWEEP_DUR_S,  # Sweep duration (seconds)

# From ir_deconvolve_function.py settings:
LF_TAPER_START_HZ,  # Frequency where LF magnitude taper starts (for spectrum shaping)
HF_TAPER_START_HZ,  # Frequency where HF magnitude taper starts (cosine fade to Nyquist)
REGU_INSIDE,  # Regularization scale inside [f_lo, f_hi] band
REGU_OUTSIDE,  # Regularization scale outside that band
REGU_XFADE_FRAC,  # Fractional width of raised-cosine transitions at band edges
ENABLE_REGULARIZATION,  # Toggle to enable the epsilon regularization
ENABLE_MANUAL_TAPERS,  # Toggle LF/HF manual tapers on the complex spectrum
PLOT,  # If True, show diagnostic plots
FADE_RATIO,  # Portion of gate window used for the half-Hanning fade
GATE_FREQ_HZ  # Gate duration is set to 3 cycles at this frequency → 3 / GATE_FREQ_HZ seconds
)

# --- Optional toggle to disable gating entirely (default True if missing in config_capture) ---
try:
    from config_capture import ENABLE_GATE  # True = apply half-Hanning gate; False = leave IR un-gated
except Exception:
    ENABLE_GATE = True  # Backward-compatible default

OUTDIR = Path(r"E:\Spkr_Scanner\Good Code\Capture\demo_outputs")  # Root directory containing input WAVs and where outputs are written

def _load_mono(path: Path):  # Helper: read WAV and ensure mono float32, return (data, sample_rate)
    y, fs = sf.read(str(path), always_2d=False)  # Read audio as ndarray; returns (data, fs)
    if y.ndim > 1:  # If stereo/multichannel…
        y = y[:, 0]  # …take the first channel to make it mono
    return y.astype(np.float32, copy=False), int(fs)  # Return float32 audio and int sample rate

def _next_pow2(n: int) -> int:  # Compute next power-of-two ≥ n (good for FFT sizing)
    return 1 << int(np.ceil(np.log2(max(1, n))))  # Bit-shift power-of-two using ceil(log2(n))

def _raised_cosine_blend(a: float, b: float, x: np.ndarray) -> np.ndarray:  # Smoothly blend between a and b over x∈[0,1]
    """Blend from a (x=0) to b (x=1) with a raised cosine."""  # Docstring: explains the blend profile
    x = np.clip(x, 0.0, 1.0)  # Clamp x into [0, 1] so math behaves
    return a + 0.5 * (1 - np.cos(np.pi * x)) * (b - a)  # Classic raised-cosine crossfade

def _build_kirkeby_epsilon(freqs: np.ndarray,  # Build frequency-dependent epsilon for regularized deconvolution
                           denom_abs: np.ndarray,  # |X|^2 term from the loop spectrum
                           fs: int,  # Sample rate
                           f_lo: float,  # Lower bound of the "inside" band
                           f_hi: float,  # Upper bound of the "inside" band
                           regu_inside: float,  # Regularization scale inside
                           regu_outside: float,  # Regularization scale outside
                           xfade_frac: float) -> np.ndarray:  # Fractional crossfade width at edges
    """Build eps(f) with two-zone model and raised-cosine transitions."""  # Docstring for epsilon design
    fny = 0.5 * fs  # Nyquist frequency
    f_lo = float(max(0.0, f_lo))  # Ensure f_lo is not negative
    f_hi = float(min(f_hi, fny))  # Do not exceed Nyquist
    if f_hi < f_lo:  # If reversed, swap bounds
        f_lo, f_hi = f_hi, f_lo

    max_den = float(np.max(denom_abs) + 1e-20)  # Reference scale to tie epsilon to the data
    eps_in  = regu_inside  * max_den  # Inside-band epsilon
    eps_out = regu_outside * max_den  # Outside-band epsilon

    eps = np.full_like(freqs, eps_out, dtype=np.float64)  # Start with outside value everywhere
    inside = (freqs >= f_lo) & (freqs <= f_hi)  # Boolean mask of inside band
    eps[inside] = eps_in  # Use inside value within the passband

    band_w = max(f_hi - f_lo, 1.0)  # Bandwidth; ensure ≥1 Hz to avoid divide-by-zero
    w = xfade_frac * band_w  # Crossfade width in Hz

    # Lower transition
    if w > 0 and f_lo > 0:  # Only if we have a positive width and not at DC
        lo_a = max(f_lo - w, 0.0)  # Start of lower transition
        lo_b = f_lo  # End of lower transition
        idx = (freqs >= lo_a) & (freqs < lo_b)  # Transition region mask
        if np.any(idx):  # If there are bins in the transition
            x = (freqs[idx] - lo_a) / max(lo_b - lo_a, 1e-12)  # Normalize to 0..1
            eps[idx] = _raised_cosine_blend(eps_out, eps_in, x)  # Blend outside→inside

    # Upper transition
    if w > 0 and f_hi < fny:  # Only if we have room before Nyquist
        hi_a = f_hi  # Start of upper transition
        hi_b = min(f_hi + w, fny)  # End (clamped to Nyquist)
        idx = (freqs > hi_a) & (freqs <= hi_b)  # Transition region mask
        if np.any(idx):  # If bins exist here
            x = (freqs[idx] - hi_a) / max(hi_b - hi_a, 1e-12)  # Normalize 0..1
            eps[idx] = _raised_cosine_blend(eps_in, eps_out, x)  # Blend inside→outside

    eps[0]  = max(eps[0],  eps_out)  # Ensure DC is not under-regularized
    eps[-1] = max(eps[-1], eps_out)  # Ensure Nyquist bin has at least outside epsilon
    return eps.astype(np.float64)  # Return epsilon as float64 array

def _apply_manual_tapers_complex(H: np.ndarray, fs: int,  # Optional LF/HF tapering of the complex spectrum magnitude
                                 f_lf_start: float,
                                 f_hf_start: float) -> np.ndarray:
    """Apply cosine LF/HF tapers to magnitude of one-sided complex spectrum."""  # Docstring for tapers
    if H.size == 0:  # Empty input guard
        return H  # Nothing to do
    freqs = np.fft.rfftfreq((H.size - 1) * 2, d=1.0 / fs)  # Recover the frequency axis for one-sided spectrum
    mag = np.abs(H).astype(np.float64, copy=True)  # Work with magnitude in float64
    ph  = np.angle(H)  # Preserve phase for reconstruction

    idx_lf_taper_start = int(np.searchsorted(freqs, max(0.0, float(f_lf_start))))  # LF taper boundary index
    idx_hf_start       = int(np.searchsorted(freqs, max(0.0, float(f_hf_start))))  # HF taper start index
    idx_nyq_out        = H.size - 1  # Index of Nyquist bin

    # High-frequency taper
    if idx_hf_start <= idx_nyq_out:  # Only if HF start is within bounds
        ten_hf = mag[idx_hf_start: idx_nyq_out + 1]  # Slice from HF start to Nyquist
        if ten_hf.size > 0:  # Non-empty guard
            x_h = np.linspace(0, 1, len(ten_hf))  # Normalized 0..1 axis for fade
            mask_h = 0.5 * (1 + np.cos(np.pi * x_h))  # Cosine fade from 1→0
            mag[idx_hf_start: idx_nyq_out + 1] = ten_hf * mask_h  # Apply fade

    # Low-frequency taper
    n_lf = idx_lf_taper_start + 1  # Number of bins affected at LF side
    if n_lf > 0:  # Only if at least one bin
        ten_lf = mag[: n_lf]  # LF slice including DC
        x_l = np.linspace(0, 1, n_lf)  # Normalized 0..1 axis
        mask_l = 0.5 * (1 - np.cos(np.pi * x_l))  # Cosine rise from 0→1
        mag[: n_lf] = ten_lf * mask_l  # Apply LF taper
        mag[0] = 0.0  # Force DC magnitude to zero (avoid offset)

    return (mag * np.exp(1j * ph)).astype(np.complex128, copy=False)  # Recombine with original phase

def _regularized_deconvolution(x, y, fs, f1, f2):  # Perform deconvolution with optional regularization and tapers
    """Pyfar-like: H = (Y * X*) / (|X|^2 + eps(f))."""  # Docstring: formula for stabilized deconvolution
    n = min(len(x), len(y))  # Match lengths (safety if unequal)
    x = x[:n]; y = y[:n]  # Truncate to the same length

    Nfft = _next_pow2(n)  # Use power-of-two FFT size for speed
    X = np.fft.rfft(x, n=Nfft)  # One-sided FFT of excitation
    Y = np.fft.rfft(y, n=Nfft)  # One-sided FFT of response
    freqs = np.fft.rfftfreq(Nfft, d=1.0 / fs)  # Frequency axis for RFFT bins
    denom_abs = (X.real * X.real + X.imag * X.imag)  # |X|^2 computed without abs() for speed

    f_lo = float(LF_TAPER_START_HZ)  # Lower band edge for "inside" region
    f_hi = float(HF_TAPER_START_HZ)  # Upper band edge for "inside" region

    if ENABLE_REGULARIZATION:  # If enabled, build frequency-shaped epsilon
        eps_f = _build_kirkeby_epsilon(freqs, denom_abs, fs,  # Shape epsilon based on band and scales
                                       f_lo=f_lo, f_hi=f_hi,
                                       regu_inside=REGU_INSIDE,
                                       regu_outside=REGU_OUTSIDE,
                                       xfade_frac=REGU_XFADE_FRAC)
    else:
        eps_f = np.full_like(freqs, 1e-20, dtype=np.float64)  # Minimal epsilon (almost pure deconvolution)

    H = (Y * np.conj(X)) / (denom_abs + eps_f)  # Regularized deconvolution spectrum
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)  # Clean up any numeric issues

    if ENABLE_MANUAL_TAPERS:  # Optional magnitude tapers to shape LF/HF roll-offs
        H = _apply_manual_tapers_complex(H, fs, LF_TAPER_START_HZ, HF_TAPER_START_HZ)

    if PLOT:  # Optional diagnostic plot of |H(f)| before IFFT
        mask = freqs > 0  # Skip DC for dB conversion
        mag_db = 20.0 * np.log10(np.maximum(np.abs(H[mask]), 1e-20))  # Convert magnitude to dB safely
        plt.figure()  # New figure
        plt.semilogx(freqs[mask], mag_db)  # Log-frequency plot
        plt.xlabel("Frequency (Hz)")  # Axis label
        plt.ylabel("Magnitude (dB)")  # Axis label
        plt.title("Deconvolved |H(f)| before IFFT")  # Title
        plt.grid(True, which="both", ls=":")  # Light grid

    h = np.fft.irfft(H, n=Nfft).astype(np.float32)  # Back to time domain as float32 IR
    return h  # Return deconvolved impulse response

def _peak(x):  # Compute peak absolute value of an array (for quick reporting)
    return float(np.nanmax(np.abs(x))) if x.size else 0.0  # Handle empty arrays safely

# --- Gate helpers ---
def half_hanning_fade(length: int, fade_ratio: float) -> np.ndarray:  # Build a window that is flat then cosines down to zero
    """Flat top, then falling half-Hanning over the final fade_ratio portion."""  # Window description
    if length < 2 or fade_ratio <= 0:  # If too short or no fade requested
        return np.ones(length, dtype=float)  # Return all-ones (no fade)
    fade_samps = max(1, min(int(round(fade_ratio * length)), length))  # Number of samples in the fade region
    flat_samps = length - fade_samps  # Samples kept at unity before the fade
    flat = np.ones(flat_samps, dtype=float)  # Flat portion
    n = np.arange(fade_samps)  # Sample index for the cosine
    fade = 0.5 * (1 + np.cos(np.pi * n / (fade_samps - 1)))  # Cosine from 1 down to 0
    return np.concatenate((flat, fade))  # Concatenate flat then fade

def apply_gate(data: np.ndarray, fs: int, gate_dur_s: float, fade_ratio: float) -> tuple[np.ndarray, int]:  # Gate from peak for a given duration
    """Find IR peak, keep gate_dur_s seconds after it with half-Hanning tail, zero rest."""  # Gate behaviour description
    peak_idx = int(np.argmax(np.abs(data)))  # Find index of maximum absolute amplitude
    gate_samps = int(round(gate_dur_s * fs))  # Convert duration to samples
    start = peak_idx  # Start gating at the peak to keep direct sound + early reflections
    end = min(len(data), start + gate_samps)  # Do not exceed signal length
    length = end - start  # Window length in samples
    window = half_hanning_fade(length, fade_ratio)  # Build the fade window
    out = data.copy()  # Work on a copy
    out[start:end] = data[start:end] * window  # Apply window in-place over gate region
    out[end:] = 0.0  # Zero everything after the gate to suppress late content
    return out, end  # Return gated signal and the index of the last kept sample

# ─────────────────────────────────────────────────────────────────────────────
# Importable function (returns array, does not write)
# ─────────────────────────────────────────────────────────────────────────────
def make_ir_from_pair(mic_path: Path, excitation_path: Path) -> np.ndarray:  # Core routine to build a single IR from a loop/mic pair
    """
    Compute the final impulse response (preserving TOF, cropped to the sweep,
    regularized deconvolution, optional HF/LF tapers, gated tail) from one
    loop/mic pair. Returns the IR; does not write anything.

    Parameters
    ----------
    loop_path : Path
        Path to '<base>_loop_aligned.wav'
    mic_path : Path
        Path to '<base>_mic_conditioned.wav'

    Returns
    -------
    np.ndarray (float32)
        Final gated IR (TOF preserved) at FS.
    """  # Detailed docstring for import usage
    x_full, fs1 = _load_mono(excitation_path)  # Load excitation WAV and sample rate
    y_full, fs2 = _load_mono(mic_path)  # Load mic (response) WAV and sample rate
    if fs1 != FS or fs2 != FS:  # Sanity check: must match configured sample rate
        raise ValueError(f"SR mismatch ({fs1}/{fs2} vs {FS})")  # Fail fast if mismatched

    # --- CROP EXACTLY THE SWEEP WINDOW ---
    pre   = int(round(PRE_SIL_MS / 1000.0 * FS))  # Convert pre-roll ms to samples
    slen  = int(round(SWEEP_DUR_S * FS))  # Sweep length in samples
    start = pre  # Start sample index of the sweep
    end   = start + slen  # End sample index of the sweep
    if end > len(x_full) or end > len(y_full):  # Check files are long enough
        raise RuntimeError("Files shorter than expected sweep window")  # Abort if not

    x = x_full[start:end]  # Crop excitation to the sweep window
    y = y_full[start:end]  # Crop response to the same window

    # Deconvolution
    h_core = _regularized_deconvolution(  # Compute deconvolved IR (no absolute timing yet)
        x, y, FS, float(F1_HZ),
        float(F2_HZ) if F2_HZ is not None else float(0.48 * FS)  # Use explicit F2_HZ or 0.48*FS as upper bound
    )

    # --- PRESERVE ABSOLUTE TIMING ---
    h = np.concatenate([np.zeros(pre, dtype=np.float32), h_core], dtype=np.float32)  # Prepend pre-roll zeros to keep TOF

    # --- Optionally apply half-Hanning gate from IR peak, then trim to gate end ---
    #     If ENABLE_GATE is False, skip gating entirely and keep full IR.
    if ENABLE_GATE:
        gate_dur_s = 4.0 / float(GATE_FREQ_HZ)  # Gate length in seconds = four cycles at GATE_FREQ_HZ
        h_gated, end_idx = apply_gate(h, FS, gate_dur_s, FADE_RATIO)  # Gate the IR tail
        h = h_gated[:end_idx].astype(np.float32, copy=False)  # Trim trailing zeros beyond gate end
        gated_state = "gated"
    else:
        h = h.astype(np.float32, copy=False)
        gated_state = "ungated"

    if PLOT:  # Optional visualization of final IR
        plt.figure()  # New figure
        t = np.arange(len(h)) / FS  # Time axis in seconds
        plt.plot(t, h)  # Plot IR vs time
        plt.title(f"Final IR ({gated_state})")  # Title
        plt.xlabel("Time (s)")  # Label
        plt.grid(True)  # Grid

    return h  # Return the final IR as float32

# ─────────────────────────────────────────────────────────────────────────────
# Original CLI flow (updated to use shared excitation signal)
# ─────────────────────────────────────────────────────────────────────────────
def main():  # Entry point for command-line usage
    print(f"Looking for WAVs in: {OUTDIR.resolve()}")  # Inform user which directory is being scanned

    # Find all microphone files
    mic_files = sorted(OUTDIR.glob("*_mic_conditioned.wav"))  # Collect mic-conditioned files
    bases = [re.sub(r"_mic_conditioned\.wav$", "", mic.name) for mic in mic_files]  # Strip suffix to get base names

    if not bases:  # If no mic files found…
        print("No mic-conditioned files found.")  # …inform and exit
        return  # Nothing to process

    # Shared excitation file saved once per run by the capture script
    excitation_path = OUTDIR / "excitation_signal.wav"  # Global excitation reference

    for base in bases:  # Process each mic file with the shared excitation
        mic_path = OUTDIR / f"{base}_mic_conditioned.wav"  # Full path to mic file
        ir_path  = OUTDIR / f"{base}_ir.wav"               # Output path for IR file

        try:
            h = make_ir_from_pair(mic_path, excitation_path)  # Build the IR (core processing)
        except Exception as e:  # Catch errors to continue with other bases
            print(f"[{base}] ERROR: {e}")  # Report the problem
            continue  # Move on to the next base

        sf.write(str(ir_path), h.astype(np.float32), FS, format="WAV", subtype="FLOAT")  # Write 32-bit float WAV
        print(f"[{base}] → wrote {ir_path.name} ({'UNGATED' if not ENABLE_GATE else 'gated'}, len={len(h)} @ {FS} Hz, peak={_peak(h):.4g})")  # Quick summary line

    if PLOT:  # If plotting is enabled…
        plt.show()  # …display all figures at the end

    print("Done.")  # Final message


if __name__ == "__main__":  # Only run main() when executed as a script (not when imported)
    main()  # Call the CLI entry point

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
