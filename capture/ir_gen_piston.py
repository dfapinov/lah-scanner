#!/usr/bin/env python3  # Use the system’s Python 3 interpreter

"""
───────────────────────────────────────────────────────────────────────────────
Generate Piston Impulse Responses at Multiple Coordinates
───────────────────────────────────────────────────────────────────────────────

This script simulates the sound field of a circular piston 
in a rigid baffle, producing impulse responses (IRs) at specified
measurement coordinates.

Each coordinate represents a microphone position in 3D space (given in a CSV file),
and for each position, the script generates a time-domain IR that models how a
plane piston would be heard there in an ideal free field.

The resulting WAV files can be used for testing loudspeaker acoustic holography
system.

───────────────────────────────────────────────────────────────────────────────
Usage
───────────────────────────────────────────────────────────────────────────────
At command line type:

python piston_ir_generator.py

Default input coordinates file is ir_gen_piston_coords.csv in the same directory
as the script.

Generated IRs go to outputs/ directory by default. A log CSV is written as outputs/audio_file_log.csv.

Move IRs to process/input_irs/ to use in the processing pipeline.

───────────────────────────────────────────────────────────────────────────────
Pipeline Overview
───────────────────────────────────────────────────────────────────────────────
1. Load coordinate list (r_xy_mm, phi_deg, z_mm) from CSV.
2. Convert cylindrical coordinates → spherical (r, θ, φ).
3. Compute piston directivity D(θ, f) using the Bessel function J1.
4. Compute free-field propagation term (Green’s function) G(f, r).
5. Multiply D(f) × G(f) to form the frequency-domain response.
6. Apply a high-frequency cosine taper to avoid aliasing.
7. Perform an inverse FFT to obtain the time-domain IR.
8. Scale to the chosen output level and write as 32-bit float WAV.
9. Log results (filename, delay, amplitude, peak time) to CSV.

Each output file is named using its spatial coordinates, e.g.:
   id5_r150_ph90p0_z100.wav
which encodes r=150 mm, φ=90°, z=100 mm.
───────────────────────────────────────────────────────────────────────────────
"""

# ───────────────────────────────────────────────
# Import required Python modules
# ───────────────────────────────────────────────
import csv                      # To read and write CSV files
import math                     # Provides mathematical functions like sin, cos, pi, etc.
from pathlib import Path         # For OS-independent file path handling
import numpy as np               # Numerical operations on arrays
import sys                      # For simple console progress bar updates
from scipy.special import j1 as scipy_j1  # Bessel function of the first kind (J1)
from scipy.io.wavfile import write as wavwrite  # For saving audio files in WAV format

# ───────────────────────────────────────────────
# Resolve base directory so relative paths work anywhere
# ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent  # Folder where this script is located

# ───────────────────────────────────────────────
# User configuration parameters (edit these as needed)
# ───────────────────────────────────────────────
DEFAULT_COORDS_CSV = BASE_DIR / "ir_gen_piston_coords.csv"  # CSV file of microphone positions
DEFAULT_OUT_DIR = BASE_DIR / "outputs"                      # Folder to save generated files
DEFAULT_PISTON_RADIUS_M = 0.03                              # Piston radius in meters (e.g. 30 mm)
PISTON_VOLUME_LEVEL = 0.5                                   # Overall volume scaling (0–1)

# Advanced Settings:
DEFAULT_FS = 96_000           # Sampling rate of generated WAVs (Hz)
DEFAULT_DURATION_S = 0.5      # Duration of each generated IR (seconds)
DEFAULT_C = 343.0             # Speed of sound in air (m/s)
DEFAULT_PREFIX = "ir"         # Prefix for generated filenames
HF_TAPER_START_HZ = 20_000.0  # Frequency (Hz) where HF cosine taper starts

# ───────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────
def ensure_dir(p: Path):
    """Ensure a directory exists; create it if necessary."""
    p.mkdir(parents=True, exist_ok=True)


def bessel_j1(x: np.ndarray) -> np.ndarray:
    """Return Bessel function J1(x). Uses SciPy’s implementation."""
    return scipy_j1(x)


def piston_directivity(theta_rad, freqs_hz, a_m, c):
    """Compute piston directivity function D(θ,f)."""
    k = 2.0 * np.pi * freqs_hz / c              # Wavenumber (radians per meter)
    arg = k * a_m * np.sin(theta_rad)           # Argument for Bessel function
    D = np.ones_like(arg)                       # Initialize unity array
    nz = np.abs(arg) > 1e-12                    # Avoid divide-by-zero at 0
    D[nz] = 2.0 * bessel_j1(arg[nz]) / arg[nz]  # Classic piston formula
    return D                                    # Return directivity response


def greens_function(freqs_hz, r_m, c):
    """Return free-field Green's function e^(-jkr)/r (propagation delay and decay)."""
    return np.exp(-1j * 2.0 * np.pi * freqs_hz * (r_m / c)) / max(r_m, 1e-12)


def cosine_taper_from(freqs_hz, f_start, f_end):
    """Create a cosine-shaped taper window starting at f_start and ending at f_end."""
    w = np.ones_like(freqs_hz)
    if f_end <= f_start:
        return w  # No taper if limits are invalid
    m = (freqs_hz >= f_start) & (freqs_hz <= f_end)
    x = (freqs_hz[m] - f_start) / (f_end - f_start)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * x))  # Cosine fade-down shape
    w[freqs_hz > f_end] = 0.0               # Zero out above taper end
    return w


def build_spectrum_for_point(r_m, theta_deg, fs, n_fft, c, a_m):
    """Build the frequency-domain pressure spectrum for a given measurement point."""
    n_pos = n_fft // 2 + 1                                  # Number of positive frequencies
    freqs = np.linspace(0.0, fs/2.0, n_pos)                 # Frequency axis (Hz)
    theta_rad = np.deg2rad(theta_deg)                       # Convert degrees → radians
    D = piston_directivity(theta_rad, freqs, a_m, c)        # Get piston directivity
    G = greens_function(freqs, r_m, c)                      # Get propagation term
    P = D * G                                               # Multiply to form complex spectrum
    if n_pos > 2:
        f_end = freqs[-2]                                   # End of valid frequency range
        taper = cosine_taper_from(freqs, HF_TAPER_START_HZ, f_end)
        P *= taper                                          # Apply high-frequency taper
    P[0] = np.real(P[0]) + 0j                               # DC component must be real
    if n_fft % 2 == 0:
        P[-1] = 0.0 + 0j                                    # Ensure Nyquist bin symmetry
    return P


def spectrum_to_ir(P_pos, n_fft):
    """Convert one-sided complex spectrum to a real time-domain impulse response."""
    return np.fft.irfft(P_pos, n=n_fft).astype(np.float64)


def write_wav_float32(path: Path, fs: int, x: np.ndarray):
    """Write a NumPy array as a 32-bit float WAV file."""
    x32 = x.astype(np.float32, copy=False)
    wavwrite(str(path), fs, x32)


def _format_decimal_for_name(val: float, digits: int = 1) -> str:
    """Convert float to text for filenames (replace '.' with 'p')."""
    return f"{val:.{digits}f}".replace(".", "p")


def _nearest_int_str(val: float) -> str:
    """Convert a float to its nearest integer string (for naming)."""
    return str(int(round(val)))


def _cylindrical_to_spherical(r_xy_mm: float, phi_deg: float, z_mm: float):
    """Convert cylindrical coordinates (r_xy, φ, z) to spherical (r, θ, φ)."""
    r_xy_m = r_xy_mm / 1000.0                       # Convert from mm → m
    z_m = z_mm / 1000.0
    r_m = math.hypot(r_xy_m, z_m)                   # Full radius in meters
    theta_deg = math.degrees(math.atan2(r_xy_m, z_m)) if z_m != 0 else 90.0
    return r_m, theta_deg, float(phi_deg)


# ───────────────────────────────────────────────
# Main execution routine
# ───────────────────────────────────────────────
def main():
    coords_csv = Path(DEFAULT_COORDS_CSV)  # Input coordinate CSV
    out_dir = Path(DEFAULT_OUT_DIR)        # Output directory for generated files

    # Check that the coordinate file exists
    if not coords_csv.exists():
        raise FileNotFoundError(f"Coordinate file '{coords_csv}' not found.")

    ensure_dir(out_dir)  # Make sure output directory exists

    # Load key configuration parameters
    fs = DEFAULT_FS
    dur = DEFAULT_DURATION_S
    c = DEFAULT_C
    a_m = DEFAULT_PISTON_RADIUS_M
    prefix = DEFAULT_PREFIX

    # Compute FFT length as next power of two (faster FFT computation)
    n_fft = int(2 ** math.ceil(math.log2(max(8, int(dur * fs)))))

    # Prepare CSV log file to record generated IR info
    meta_path = out_dir / "audio_file_log.csv"

    # Open input coordinates and output metadata CSV simultaneously
    with coords_csv.open("r", newline="") as f_in, meta_path.open("w", newline="") as f_out:
        rdr = csv.DictReader(f_in)
        input_fields = rdr.fieldnames or []
        fieldnames_lower = {name.lower() for name in input_fields}
        required = {"r_xy_mm", "phi_deg", "z_mm"}
        if not required.issubset(fieldnames_lower):
            raise ValueError("CSV must contain: r_xy_mm, phi_deg, z_mm")

        # Add extra columns to output CSV for generated data
        append_fields = ["filename", "delay_s", "max_abs_ir", "peak_sample", "peak_time_s"]
        out_fields = list(input_fields) + [c for c in append_fields if c not in input_fields]
        w = csv.DictWriter(f_out, fieldnames=out_fields)
        w.writeheader()

        # Read all coordinate rows at once
        rows = list(rdr)

        # --- simple console progress bar (no external deps) ---
        total = len(rows)
        bar_width = 40
        print("Processing coordinates")
        # ------------------------------------------------------

        for i, row in enumerate(rows):
            # Extract coordinates from CSV
            r_xy_mm = float(row["r_xy_mm"])
            phi_deg_in = float(row["phi_deg"])
            z_mm = float(row["z_mm"])

            # Convert to spherical coordinates
            r_m, theta_deg, phi_deg = _cylindrical_to_spherical(r_xy_mm, phi_deg_in, z_mm)

            # Build unique filename for this coordinate
            order_idx = row.get("order_idx")
            order_idx_txt = (order_idx.strip() if order_idx not in (None, "") else str(i))
            r_txt = _nearest_int_str(r_xy_mm)
            ph_txt = _format_decimal_for_name(phi_deg, 1)
            z_txt = _nearest_int_str(z_mm)
            fname = f"id{order_idx_txt}_r{r_txt}_ph{ph_txt}_z{z_txt}.wav"

            # Rotate piston 90° into XY-plane so it faces +X direction
            # Compute the angle between the measurement vector and +X
            alpha_deg = math.degrees(math.acos(np.clip(
                math.sin(math.radians(theta_deg)) * math.cos(math.radians(phi_deg)),
                -1.0, 1.0
            )))

            # Generate complex pressure spectrum for this direction
            P = build_spectrum_for_point(r_m, alpha_deg, fs, n_fft, c, a_m)

            # Convert to time-domain impulse response
            ir = spectrum_to_ir(P, n_fft)
            ir *= PISTON_VOLUME_LEVEL  # Apply output level scaling

            # Write impulse response to WAV file
            wav_path = out_dir / fname
            write_wav_float32(wav_path, fs, ir)

            # Collect and record metadata for this IR
            delay_s = r_m / c                           # Time-of-flight delay
            peak_idx = int(np.argmax(np.abs(ir)))       # Sample index of max amplitude
            peak_time_s = peak_idx / fs                 # Peak position in seconds
            max_abs = float(np.max(np.abs(ir)))         # Max absolute amplitude

            out_row = dict(row)                         # Copy input row
            out_row.update({
                "filename": fname,
                "delay_s": f"{delay_s:.9f}",
                "max_abs_ir": f"{max_abs:.3e}",
                "peak_sample": str(peak_idx),
                "peak_time_s": f"{peak_time_s:.9f}",
            })
            w.writerow(out_row)  # Write metadata row to log file

            # --- progress bar update ---
            filled = int(bar_width * (i + 1) / max(1, total))
            bar = "#" * filled + "-" * (bar_width - filled)
            sys.stdout.write(f"\r[{bar}] {i + 1}/{total}")
            sys.stdout.flush()
            # ---------------------------

    # end of loop – newline after bar
    print()

    # Print summary of generation results
    print(f"\nGenerated IRs for {meta_path}")
    print(f"Sample rate: {fs} Hz | Duration: {dur} s | FFT: {n_fft}")
    print(f"Piston radius: {a_m*1000:.1f} mm | Speed of sound: {c} m/s")
    print(f"Piston volume level: {PISTON_VOLUME_LEVEL:.3f}×")
    print(f"Input: {coords_csv}")
    print(f"Output: {out_dir}\nLog: {meta_path}\n")


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
    
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