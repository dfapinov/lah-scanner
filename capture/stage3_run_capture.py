#!/usr/bin/env python3
# stage3_run_capture.py

"""
Description:
===========

This script automates acoustic sweep measurements across a set of predefined
3D coordinates. It reads a coordinate list from the input CSV file (COORD_FILE)
and, for each position, runs a logarithmic sweep using the function defined in
`sweep_function.py`. The sweep captures both the microphone signal and the
electrical loopback reference. After each measurement, the script calls
`make_ir_function.py` to convert the sweep pair into a time-domain impulse
response (IR), which is later processed by the Spherical Harmonic Expansion
(SHE) pipeline.

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
        2. Optionally saves conditioned mic,loopback and transmitted sweep WAV files.
        3. Generates and saves an impulse response (IR) WAV.
    - Files are written to OUTDIR defined in config_capture.py.

"""

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
    SAVE_MIC_LOOP_SIG # Toggle to retain or delete intermediate files
)

# Ensure output directory exists (create if not)
os.makedirs(OUTDIR, exist_ok=True)


def _fmt_num_for_name(val):
    """Convert numeric values to filename-safe strings (replace '.' with 'p')."""
    s = str(val)
    return s.replace(".", "p")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    print(f"Reading coordinates from {COORD_FILE}...")  # Inform user
    with open(COORD_FILE, newline="") as f:             # Open CSV file
        reader = csv.DictReader(f)                      # Parse header-based rows
        coords = list(reader)                           # Convert to list for indexing

    # Loop over coordinates, up to MAX_POINTS
    for idx, row in enumerate(coords[:MAX_POINTS], start=1):
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
        result = run_measurement()  # Returns dict with mic, loop, and metadata

        # Extract sample rate and convert arrays to 32-bit float
        fs = int(result["fs"])  # Sampling rate in Hz
        mic = result["rx_mic_conditioned"].astype(np.float32, copy=False)  # Conditioned mic recording
        loop = result["rx_loop_aligned"].astype(np.float32, copy=False)    # Loopback reference aligned

        # Construct base name for all output files (coordinate-based)
        base = (
            f"id{_fmt_num_for_name(order_i)}"
            f"_r{_fmt_num_for_name(r_xy_mm)}"
            f"_ph{_fmt_num_for_name(phi_deg)}"
            f"_z{_fmt_num_for_name(z_mm)}"
        )

        # Define full paths for mic and loop WAV files
        mic_path  = os.path.join(OUTDIR, f"{base}_mic_conditioned.wav")
        loop_path = os.path.join(OUTDIR, f"{base}_loop_aligned.wav")

        # Write mic and loop signals to disk (needed for IR generation)
        sf.write(mic_path,  mic, fs, format="WAV", subtype="FLOAT")
        sf.write(loop_path, loop, fs, format="WAV", subtype="FLOAT")

        # Only print confirmation messages if user wants to keep these files
        if SAVE_MIC_LOOP_SIG:
            print(f"Saved {mic_path}")
            print(f"Saved {loop_path}")
            print(f"Saved {mic_path}")
                    # If SAVE_MIC_LOOP_SIG=True, also save raw mic signal
        if SAVE_MIC_LOOP_SIG:
            rx_mic_raw = result.get("rx_mic_raw")
            if rx_mic_raw is not None:
                rx_mic_raw = np.asarray(rx_mic_raw, dtype=np.float32)
                raw_path = os.path.join(OUTDIR, f"{base}_mic_raw.wav")
                sf.write(raw_path, rx_mic_raw, fs, format="WAV", subtype="FLOAT")
                print(f"Saved {raw_path}")


        # If SAVE_MIC_LOOP_SIG=True, also store transmitted sweep signal
        if SAVE_MIC_LOOP_SIG:
            tx_signal = result.get("tx_signal")  # Retrieve transmitted signal
            if tx_signal is not None:
                tx_signal = np.asarray(tx_signal, dtype=np.float32)
                sf.write(os.path.join(OUTDIR, f"{base}_tx_signal.wav"), tx_signal, fs, format="WAV", subtype="FLOAT")
                print(f"Saved {os.path.join(OUTDIR, f'{base}_tx_signal.wav')}")

        # Generate impulse response (IR) from mic/loop pair using external function
        ir_array = make_ir_from_pair(Path(loop_path), Path(mic_path))
        ir_path = os.path.join(OUTDIR, f"{base}_ir.wav")

        # Save IR as 32-bit float WAV file
        sf.write(ir_path, ir_array.astype(np.float32), fs, format="WAV", subtype="FLOAT")
        print(f"Saved {ir_path}")

        # If user chose not to keep raw signals, delete them after IR creation
        if not SAVE_MIC_LOOP_SIG:
            try:
                os.remove(mic_path)
            except OSError:
                pass  # Ignore if already deleted or missing
            try:
                os.remove(loop_path)
            except OSError:
                pass

    # After all coordinates processed, print completion message
    print("\nAll demo points complete.")


# ─────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()  # Run the main measurement loop

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