#!/usr/bin/env python3
"""
stage1_preprocess_irs.py
========================

Description:
------------
Preprocesses measured impulse responses (IRs) by applying gating, filtering, and optional auto-gain normalization.
Each input WAV is split into a low-frequency (LF) and high-frequency (HF) version based on gate durations
derived from the configuration file. The LF/HF IRs are written as 32-bit float WAVs.

The purpose of this stage is to prepare measured IRs for spherical harmonic expansion (SHE) in stage-2 and later mergin
of LF and HF response in stage 3.

Usage:
------
Configured via:  config_process.py

Run directly ar command line:
    python stage1_preprocess_irs.py

Input:
    • Directory defined in config_process.py INPUT_IR_DIR_REL containing .wav files.

Output:
    • "lf_irs" and "hf_irs" folders inside OUTPUT_DIR_REL.
    • Each file written as 32-bit float WAV (_LF.wav and _HF.wav suffixes).
    • Optional log written to /logs/pipeline.log

Auto-gain - finds the file with greatest peak level and normalises all other files with the same gain level to
maintain relative levels between files.

Pipeline overview:
------------------
1. Load all .wav impulse responses (IRs) from INPUT_IR_DIR_REL.
2. (Optional) Compute auto-gain normalization to TARGET_PEAK_DB based on the
   highest peak across all IRs.
3. For each IR:
     a) (Optional) Zero-pad the start by ZERO_PAD_MS milliseconds.
     b) Apply two time-domain gates using half-Hanning fade windows:
           • LF gate length = 3 cycles at LF_GATE_FREQ_HZ
           • HF gate length = 3 cycles at CROSSOVER_FREQ_HZ
     c) Apply a 1st-order Butterworth high-pass filter (if enabled).
     d) Save results as 32-bit float WAVs:
           <name>_LF.wav  (long gate, low-frequency)
           <name>_HF.wav  (short gate, high-frequency)
4. Write all processed files into:
       OUTPUT_DIR_REL/lf_irs/
       OUTPUT_DIR_REL/hf_irs/

"""

from __future__ import annotations  # enables postponed type annotation evaluation

import os           # for path and directory handling
import sys          # for system exit and printing errors
import logging      # for writing log messages to file
import numpy as np  # numerical operations on arrays
import soundfile as sf  # read/write WAV audio files
from scipy.signal import butter, filtfilt  # for 1st-order Butterworth HP filter

# ── Import parameters from config_process.py ─────────────────────────────
from config_process import (
    LF_GATE_FREQ_HZ,     # frequency defining LF gate duration
    CROSSOVER_FREQ_HZ,   # frequency defining HF gate duration
    HP_CUTOFF_HZ,        # cutoff frequency for high-pass filter
    ZERO_PAD_MS,         # milliseconds to zero-pad start of IR
    FADE_RATIO,          # fraction of gate length used for fade-out
    TARGET_PEAK_DB,      # target level (dBFS) for auto-gain
    ENABLE_AUTO_GAIN,    # enable/disable auto-gain normalization
    INPUT_IR_DIR_REL,    # relative path to input IR directory
    OUTPUT_DIR_REL,      # relative path to output root directory
)

# ── Calculate gate durations in seconds (3 cycles of each defining frequency) ─────────────
GATE_DUR_LF = 4.0 / LF_GATE_FREQ_HZ      # 4 cycles of low-frequency limit
GATE_DUR_HF = 4.0 / CROSSOVER_FREQ_HZ    # 4 cycles of crossover frequency

# ─────────────────────────────────────────────────────────────────────────────
def half_hanning_fade(length: int, fade_ratio: float) -> np.ndarray:
    """Create a window that stays flat, then fades smoothly to zero (cosine taper)."""
    if length < 2 or fade_ratio <= 0:  # if too short or fade disabled
        return np.ones(length, dtype=float)  # no fade, all ones
    fade_samps = max(1, min(int(round(fade_ratio * length)), length))  # fade length in samples
    flat_samps = length - fade_samps   # number of constant samples before fade
    flat = np.ones(flat_samps, dtype=float)  # flat section = unity gain
    n = np.arange(fade_samps)  # array of sample indices [0, 1, 2, ...]
    fade = 0.5 * (1 + np.cos(np.pi * n / (fade_samps - 1)))  # cosine fade from 1 → 0
    return np.concatenate((flat, fade))  # combine flat and fading sections


def apply_gate(data: np.ndarray, fs: int, gate_dur: float, fade_ratio: float) -> np.ndarray:
    """Locate IR peak, keep gate_dur seconds after it, fade tail, zero the rest."""
    peak_idx = int(np.argmax(np.abs(data)))  # find index of largest absolute amplitude
    gate_samps = int(gate_dur * fs)          # convert gate duration to samples
    start = peak_idx                         # start gate at the IR peak
    end = min(len(data), start + gate_samps) # end gate without exceeding array length
    length = end - start                     # actual gated length
    window = half_hanning_fade(length, fade_ratio)  # make fade window
    out = data.copy()                        # copy data to avoid modifying original
    out[start:end] = data[start:end] * window  # apply fade to gated region
    out[end:] = 0                            # zero everything after gate
    return out                               # return processed signal


def apply_hp(data: np.ndarray, fs: int, cutoff_hz: float | None) -> np.ndarray:
    """Apply 1st-order Butterworth HP filter using zero-phase filtfilt."""
    if cutoff_hz and cutoff_hz > 0:                   # only if enabled
        b, a = butter(1, cutoff_hz / (fs / 2), btype='highpass')  # design filter
        return filtfilt(b, a, data)                    # apply zero-phase filtering
    return data                                        # return unchanged if disabled


def render_bar(idx: int, total: int, width: int = 40) -> str:
    """Return ASCII progress bar like ####------."""
    pct = (idx + 1) / total               # fraction complete
    filled = int(width * pct)             # number of '#' to display
    return '#' * filled + '-' * (width - filled)  # bar string


def _norm(p: str) -> str:
    """Return normalized absolute path for display."""
    return os.path.normpath(os.path.abspath(p))


def _fail(msg: str, code: int = 2) -> None:
    """Print error and exit program."""
    print(f"ERROR: {msg}")
    sys.exit(code)


def _append_suffix(filename: str, suffix: str) -> str:
    """Append suffix before .wav extension, preserving case."""
    root, ext = os.path.splitext(filename)
    return f"{root}{suffix}{ext}"


def preprocess_and_gate_irs() -> None:
    """Main routine: read IRs, apply gating, filter, auto-gain, and save."""
    # --- Setup logging ---
    script_dir = os.path.dirname(os.path.abspath(__file__))  # current script folder
    logs_dir = os.path.join(script_dir, 'logs')               # path for logs
    os.makedirs(logs_dir, exist_ok=True)                      # create if needed
    logging.basicConfig(
        level=logging.INFO,                      # log level
        format='%(asctime)s %(levelname)s: %(message)s'  # output format
    )  

    # --- Resolve input/output directories ---
    ir_folder = os.path.join(script_dir, INPUT_IR_DIR_REL)  # where IR WAVs are stored
    out_root = os.path.join(script_dir, OUTPUT_DIR_REL)     # root of output directory
    lf_dir = os.path.join(out_root, 'lf_irs')               # subfolder for LF IRs
    hf_dir = os.path.join(out_root, 'hf_irs')               # subfolder for HF IRs
    os.makedirs(lf_dir, exist_ok=True)                      # create LF folder
    os.makedirs(hf_dir, exist_ok=True)                      # create HF folder

    # --- Check input directory ---
    if not os.path.isdir(ir_folder):
        _fail(f"Input directory not found: {_norm(ir_folder)}\nCheck INPUT_IR_DIR_REL in config.py.")

    wav_files = [f for f in os.listdir(ir_folder) if f.lower().endswith('.wav')]
    if len(wav_files) == 0:
        _fail(f"No .wav files found in: {_norm(ir_folder)}")

    total = len(wav_files)  # number of files to process

    # --- Auto-gain normalization (optional) ---
    if ENABLE_AUTO_GAIN:
        max_peak = 0.0
        max_peak_fname = None
        print("Searching for max-peak reference…")
        for idx, fname in enumerate(wav_files):
            path = os.path.join(ir_folder, fname)
            data, _ = sf.read(path)
            peak = float(np.max(np.abs(data)))      # find max absolute value
            if peak > max_peak:                     # update reference if higher
                max_peak = peak
                max_peak_fname = fname
            sys.stdout.write(f"\r{render_bar(idx, total)} File {idx+1}/{total}")
            sys.stdout.flush()
        print("\n")
        msg = f"Auto-gain reference: {max_peak_fname} (peak {max_peak:.6f})"
        print(msg)
        logging.info(msg)
        gain = ((10 ** (TARGET_PEAK_DB / 20.0)) / max_peak) if max_peak > 0 else 1.0  # compute gain factor
    else:
        msg = "Auto-gain disabled; using unity gain."
        print(msg)
        logging.info(msg)
        gain = 1.0

    # --- Main processing loop ---
    print("\nGating → 1st-order HP filter → writing 32-bit float outputs\n")
    for idx, fname in enumerate(wav_files):
        path = os.path.join(ir_folder, fname)
        data, fs = sf.read(path)               # load WAV and sample rate

        # Optional zero-padding at start
        if ZERO_PAD_MS and ZERO_PAD_MS > 0:
            pad = int(fs * (ZERO_PAD_MS / 1000))     # convert ms to samples
            data = np.concatenate((np.zeros(pad, dtype=data.dtype), data))  # prepend zeros

        # Apply gating for LF and HF
        lf = apply_gate(data, fs, GATE_DUR_LF, FADE_RATIO)
        hf = apply_gate(data, fs, GATE_DUR_HF, FADE_RATIO)

        # Apply high-pass filter (DC removal)
        lf = apply_hp(lf, fs, HP_CUTOFF_HZ)
        hf = apply_hp(hf, fs, HP_CUTOFF_HZ)

        # Save processed LF and HF files as float WAVs
        out_lf = os.path.join(lf_dir, _append_suffix(fname, '_LF'))
        out_hf = os.path.join(hf_dir, _append_suffix(fname, '_HF'))
        sf.write(out_lf, lf * gain, fs, subtype='FLOAT')
        sf.write(out_hf, hf * gain, fs, subtype='FLOAT')

        # Update console progress bar
        sys.stdout.write(f"\r{render_bar(idx, total)} File {idx+1}/{total}")
        sys.stdout.flush()

    print("\n\nPreprocessing & gating complete.\n")


# ── Run main if executed directly ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        preprocess_and_gate_irs()
    except KeyboardInterrupt:
        _fail("Interrupted by user.", code=130)

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