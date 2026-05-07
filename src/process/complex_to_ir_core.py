#!/usr/bin/env python3
"""
complex_to_ir_core.py
=====================
Core mathematical engine for converting complex frequency-domain pressures 
into a time-domain impulse response.
"""

from __future__ import annotations
import numpy as np
from scipy.fft import next_fast_len

def complex_to_ir(
    p_complex: np.ndarray,
    freqs: np.ndarray,
    target_fs: int = None
) -> np.ndarray:
    """
    Converts complex pressure data into a time-domain impulse response.

    Args:
        p_complex: 1D array of complex pressure values.
        freqs: 1D array of corresponding frequencies.
        target_fs: Optional target sample rate. If None, auto-detects.

    Returns:
        A 1D NumPy array containing the real-valued impulse response.
    """
    if target_fs is None:
        target_fs = 44100 if freqs[-1] < 23000.0 else 48000

    # 1. Determine original and target FFT lengths using frequency resolution
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    n_fft_original = int(np.round(target_fs / df))
    n_fft_target = next_fast_len(n_fft_original)

    # 2. Create the full one-sided spectrum for the target FFT size
    n_freq_target = n_fft_target // 2 + 1
    full_p_complex = np.zeros(n_freq_target, dtype=np.complex128)
    full_freqs = np.fft.rfftfreq(n_fft_target, 1/target_fs)

    # Map the input frequencies to the closest bins in the target frequency grid
    df_target = target_fs / n_fft_target
    target_indices = np.round(freqs / df_target).astype(int)
    valid_mask = target_indices < n_freq_target
    full_p_complex[target_indices[valid_mask]] = p_complex[valid_mask]

    # 3. Apply magnitude tapers
    mag_taper = np.ones(n_freq_target)
    
    # HF taper (Rolls off over the top ~9% of the available measured band. Starts at ~20kHz for 44.1kHz SR and 21.8KHz for 48KHz SR)
    hf_max = min(freqs[-1], target_fs / 2.0)
    hf_taper_start = hf_max * 0.91
    hf_fade_mask = (full_freqs >= hf_taper_start) & (full_freqs <= hf_max)
    mag_taper[hf_fade_mask] = 0.5 * (1 + np.cos(np.pi * (full_freqs[hf_fade_mask] - hf_taper_start) / (hf_max - hf_taper_start)))
    
    # Ensure all bins above the available measured data are strictly zeroed
    mag_taper[full_freqs > hf_max] = 0.0

    # LF taper (DC to 15 Hz)
    lf_taper_end = 15.0
    lf_indices = (full_freqs <= lf_taper_end)
    mag_taper[lf_indices] = 0.5 * (1 - np.cos(np.pi * full_freqs[lf_indices] / lf_taper_end))
    full_p_complex *= mag_taper

    # 4. Build the real-valued impulse response using irfft
    ir_out = np.fft.irfft(full_p_complex, n=n_fft_target)

    # 5. Apply a causal half-Hann window to fade out the last 5% of the IR
    alpha = 0.05
    n_taper = int(len(ir_out) * alpha)
    if n_taper > 0:
        taper_curve = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, n_taper)))
        ir_out[-n_taper:] *= taper_curve
        
    return ir_out