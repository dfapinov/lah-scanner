#!/usr/bin/env python3
# stage3_run_capture.py

"""
Description:
===========

This script automates acoustic sweep measurements across a set of predefined
3D coordinates. It reads a coordinate list from the input CSV file (COORD_FILE)
and, for each position, runs a logarithmic sweep using the function defined in
`sweep_function.py`.

The sweep function saves the digital excitation signal together with the sweep
captured by the microphone, performing time alignment internally to remove
system latency.

After each measurement, the script calls `make_ir_function.py` to convert
the sweep and excitation pair into a time-domain impulse response (IR),
which is later processed by the Spherical Harmonic Expansion (SHE) pipeline.

Although this version only simulates motion, it is designed to eventually
control a robotic loudspeaker measurement system that moves the microphone
automatically between coordinates. A practical implementation can use a
Marlin-based 3D printer control board (e.g. BTT SKR 1.4 Turbo), which accepts
G-code commands over USB serial. This allows homing, precise movement, and
tuning of parameters such as acceleration or steps per millimetre, all of which
can be controlled and saved over USB.

Usage:
======

Run this script at the command line:

    python stage3_run_capture.py

Input:
    - Coordinate file CSV containing measurement points. Set in config_capture.py
      Each row specifies a position (r_xy_mm, phi_deg, z_mm, order_idx).

Output:
    - For each coordinate, the script:
        1. Runs a measurement sweep.
        2. Optionally saves conditioned mic, loopback and transmitted sweep WAV files.
        3. Generates and saves an impulse response (IR) WAV.
    - Files are written to OUTDIR defined in config_capture.py.

"""

# ─────────────────────────────────────────────
# Imports and Configuration
# ─────────────────────────────────────────────

import os                     # For file and directory handling
import csv                    # To read coordinate CSV files
import time                   # For simulating movement delay
from pathlib import Path       # Path manipulation across platforms
import numpy as np             # Numerical array operations
import soundfile as sf         # Read/write audio files (WAVs)
from sweep_function import run_measurement  # External module to run sweep capture
from make_ir_function import make_ir_from_pair  # External module to derive IR from loop/mic pair

# Import user settings from configuration file
from config_capture import (
    COORD_FILE,       # CSV file containing coordinate list
    OUTDIR,           # Output directory for all recordings
    MAX_POINTS,       # Limit number of points processed
    DEBUG_SAVES # Toggle to retain or delete intermediate files
)

# Ensure output directory exists (create if not)
OUTDIR = Path(OUTDIR)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def _fmt_num_for_name(val):
    """Convert numeric/str values to filename-safe strings (replace '.' with 'p')."""
    if val is None:
        return "NA"
    s = str(val)
    return s.replace(".", "p")


def _write_wav(path: Path, data: np.ndarray, fs: int, *, label: str | None = None, print_status: bool = True) -> None:
    """Small helper to write 32-bit float WAVs and (optionally) print a single confirmation line."""
    sf.write(str(path), np.asarray(data, dtype=np.float32), fs, format="WAV", subtype="FLOAT")
    if print_status:
        print(f"Saved {label + ': ' if label else ''}{path.name}")


# ─────────────────────────────────────────────
# Core Processing
# ─────────────────────────────────────────────

def _process_point(idx: int, row: dict) -> None:
    """Run one measurement, save optional artifacts, and build IR."""
    print(f"\n=== Point {idx} / {MAX_POINTS} ===")

    # Extract coordinate data (as text from CSV)
    r_xy_mm = row.get("r_xy_mm")     # Radial distance in XY plane (mm)
    phi_deg = row.get("phi_deg")     # Azimuth angle (degrees)
    z_mm    = row.get("z_mm")        # Height position (mm)
    order_i = row.get("order_idx")   # Optional order/index column

    # Print coordinate info for user feedback
    print(f"Target coordinate: r_xy_mm={r_xy_mm}, phi_deg={phi_deg}, z_mm={z_mm}, order_idx={order_i}")

    # Simulate robotic movement (in a real system this would move the mic/arm)
    print("Simulating robotic movement...")
    time.sleep(2.0)  # Wait for 2 seconds to mimic motion time
    print("Robot in position. Starting measurement...")

    # Call sweep measurement function to acquire signals
    result = run_measurement()  # Returns dict with audio data

    # Extract sample rate and convert arrays to 32-bit float
    fs = int(result["fs"])  # Sampling rate in Hz
    mic_conditioned = result["rx_mic_conditioned"].astype(np.float32, copy=False)  # Conditioned mic recording
   
    # Also pull optional signals for debug saves
    mic_aligned  = np.asarray(result.get("rx_mic_aligned"), dtype=np.float32) if "rx_mic_aligned" in result else None
    mic_raw      = np.asarray(result.get("rx_mic_raw"), dtype=np.float32) if "rx_mic_raw" in result else None
    loop_raw     = np.asarray(result.get("rx_loop_raw"), dtype=np.float32) if "rx_loop_raw" in result else None
    loop_aligned = np.asarray(result.get("rx_loop_aligned"), dtype=np.float32) if "rx_loop_aligned" in result else None
    
    # Construct base name for all output files (coordinate-based)
    base = (
        f"id{_fmt_num_for_name(order_i)}"
        f"_r{_fmt_num_for_name(r_xy_mm)}"
        f"_ph{_fmt_num_for_name(phi_deg)}"
        f"_z{_fmt_num_for_name(z_mm)}"
    )

    # --- Always save mic (needed by IR builder which takes a file path)
    mic_path = OUTDIR / f"{base}_mic_conditioned.wav"
    _write_wav(mic_path, mic_conditioned, fs, label="mic", print_status=DEBUG_SAVES)

    # Save the transmitted sweep to OUTDIR (overwrite each loop)
    tx_signal = np.asarray(result["tx_signal"], dtype=np.float32)  # Mandatory field
    excitation_file = OUTDIR / "excitation_signal.wav"
    _write_wav(excitation_file, tx_signal, fs, label="excitation_signal", print_status=DEBUG_SAVES)

    # Save the reference-channel timeline (marker with pre/post) if present (overwrite each loop)
    loopback_file = OUTDIR / "loopback_signal.wav"
    if "tx_ref_signal" in result:
        tx_ref_signal = np.asarray(result["tx_ref_signal"], dtype=np.float32)
        _write_wav(loopback_file, tx_ref_signal, fs, label="loopback_signal", print_status=DEBUG_SAVES)

    # --- Optional diagnostics: loop-aligned, raw/aux files (when DEBUG_SAVES is True)
    if DEBUG_SAVES:
        loop_aligned_path = OUTDIR / f"{base}_loop_aligned.wav"
        _write_wav(loop_aligned_path, loop_aligned, fs, label="loop_aligned")

        if mic_raw is not None:
            _write_wav(OUTDIR / f"{base}_mic_raw.wav", mic_raw, fs, label="mic_raw")

        if mic_aligned is not None:
            _write_wav(OUTDIR / f"{base}_mic_aligned.wav", mic_aligned, fs, label="mic_aligned")

        if loop_raw is not None:
            _write_wav(OUTDIR / f"{base}_loop_raw.wav", loop_raw, fs, label="loop_raw")

    # Generate impulse response (IR) from mic + original excitation sweep
    ir_array = make_ir_from_pair(mic_path, excitation_file)
    ir_path = OUTDIR / f"{base}_ir.wav"
    _write_wav(ir_path, ir_array.astype(np.float32), fs, label="IR", print_status=True)  # Always report the IR

    # If user chose not to keep raw signals, delete them after IR creation (only those we created here)
    if not DEBUG_SAVES:
        # Delete mic_conditioned (already your behavior)
        try:
            mic_path.unlink(missing_ok=True)
        except Exception:
            pass

        # Delete excitation and loopback after they have served make_ir_from_pair
        try:
            excitation_file.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            loopback_file.unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────
# Main Loop and Entry Point
# ─────────────────────────────────────────────

def main():
    print(f"Reading coordinates from {COORD_FILE}...")  # Inform user
    with open(COORD_FILE, newline="") as f:             # Open CSV file
        reader = csv.DictReader(f)                      # Parse header-based rows
        coords = list(reader)                           # Convert to list for indexing

    # Loop over coordinates, up to MAX_POINTS
    for idx, row in enumerate(coords[:MAX_POINTS], start=1):
        _process_point(idx, row)

    # After all coordinates processed, print completion message
    print("\nAll demo points complete.")


if __name__ == "__main__":
    main()  # Run the main measurement loop


r"""
       ____  __  __ ___ _____ ______   __
      |  _ \|  \/  |_ _|_   _|  _ \ \ / /
      | | | | |\/| || |  | | | |_) \ V /
      | |_| | |  | || |  | | |  _ < | |
     _|____/|_| _|_|___| |_| |_|_\_\|_|   __
    |  ___/ \  |  _ \_ _| \ | |/ _ \ \   / /
    | |_ / _ \ | |)_ | ||  \| | | | \ \ / /
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
