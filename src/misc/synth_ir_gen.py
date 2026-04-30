#!/usr/bin/env python3
"""
───────────────────────────────────────────────────────────────────────────────
Generate Impulse Responses for any measurement grid CSV.
───────────────────────────────────────────────────────────────────────────────
Simulates a circular piston source inside your grid by modeling it as a dense cluster 
of monopoles (Discrete Rayleigh Integral). It crops the output to the
user-defined duration.

Acoustic Properties:
- You can shift the source position or change its size, which directly affects its directivity.
- It has no baffle; its directivity is purely related to its physical radius.
- It acts essentially as an omnidirectional monopole at low frequencies (LF) and becomes highly directional at high frequencies (HF).
- Note: Unlike a real unbaffled dipole driver, the front and rear radiation are in-phase because the surface is simulated as a cluster of monopoles.

Usage Example:
    Just run the script directly from the terminal. 
    Make sure 'cylindrical_grid_1000pts.csv' is in the same directory.
    $ python synth_ir_gen.py
    
    The script will output .wav files to the 'input_irs_synth' directory
    along with a 'piston_dist_log.csv' containing generation metadata.
"""

import csv  # Used for reading and writing comma-separated values files (like our grid)
import math  # Provides standard mathematical functions (like cos, sin, radians)
from pathlib import Path  # Object-oriented filesystem paths (better than standard os.path)
import numpy as np  # The core library for numerical computation and arrays in Python
import sys  # System-specific parameters and functions, used here for console output
from scipy.io.wavfile import write as wavwrite  # Function to save numpy arrays as WAV audio files

# ───────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────
# Define the base directory as the folder where this script is located
BASE_DIR = Path(__file__).resolve().parent
# Define the path to the input CSV file containing the microphone coordinates
DEFAULT_COORDS_CSV = BASE_DIR /   "cylindrical_grid_1000pts.csv"
# Define the directory where the generated WAV files will be saved
DEFAULT_OUT_DIR = BASE_DIR / "input_irs_synth"

# --- PHYSICS CONFIGURATION ---
# The 3D coordinate (X, Y, Z) for the center of our simulated speaker piston
SOURCE_CENTER_M = (0.0, 0.0, 0.0)  # Offset center
# The physical radius of the simulated piston in meters (0.015m = 15mm radius = 30mm diameter tweeter)
PISTON_RADIUS_M = 0.015              # 30mm radius
# How many discrete point sources (monopoles) we will use to approximate the flat piston surface
# More points = more accurate high frequencies, but slower calculation
PISTON_POINT_COUNT = 200            # Surface discretization
# The direction the piston is facing as a 3D vector. (1, 0, 0) means it points straight along the positive X axis
PISTON_FACING_AXIS = (1.0, 0.0, 0.0)# Facing +X

# A simple multiplier to make the final audio files louder or quieter
VOLUME_GAIN = 1.0

# --- AUDIO SETTINGS ---
# The sample rate for the audio files (48,000 samples per second is standard for high quality audio)
DEFAULT_FS = 48_000
# How long the final impulse response audio file should be, in seconds
DEFAULT_DURATION_S = 0.1      # Output will be exactly this length (e.g. 4800 samples)
# The speed of sound in air, in meters per second (approximate room temperature)
DEFAULT_C = 343.0
# The frequency at which we start gently rolling off the high frequencies to prevent digital aliasing
HF_TAPER_START_HZ = 20_000.0


# ───────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────
def ensure_dir(p: Path):
    """Creates a directory if it doesn't already exist. parents=True creates intermediate folders too."""
    p.mkdir(parents=True, exist_ok=True)

def _format_decimal_for_name(val: float, digits: int = 1) -> str:
    """Formats a floating point number for use in a file name by replacing the decimal point with a 'p'."""
    return f"{val:.{digits}f}".replace(".", "p")

def _nearest_int_str(val: float) -> str:
    """Rounds a floating point number to the nearest integer and returns it as a string."""
    return str(int(round(val)))

def cosine_taper_from(freqs_hz, f_start, f_end):
    """
    Creates a 'window' or 'filter' array that gently tapers (fades) from 1.0 down to 0.0 
    using the shape of half a cosine wave. This is used to smoothly roll off high frequencies.
    """
    w = np.ones_like(freqs_hz)  # Start with an array of all 1s (no attenuation)
    if f_end <= f_start: return w  # If the end is before the start, do nothing
    
    # Create a boolean mask for the frequencies that fall within our tapering range
    m = (freqs_hz >= f_start) & (freqs_hz <= f_end)
    
    # Calculate 'x' as a normalized value from 0.0 to 1.0 across the tapering range
    x = (freqs_hz[m] - f_start) / (f_end - f_start)
    
    # Apply the cosine formula to create the smooth curve from 1.0 to 0.0
    w[m] = 0.5 * (1.0 + np.cos(np.pi * x))
    
    # Any frequency above the end of the taper is completely silenced (multiplied by 0.0)
    w[freqs_hz > f_end] = 0.0
    return w

def _cylindrical_to_cartesian(r_xy_mm, phi_deg, z_mm):
    """
    Converts cylindrical coordinates (radius, angle, height) in millimeters and degrees
    into standard 3D Cartesian coordinates (X, Y, Z) in meters.
    """
    r_xy_m = r_xy_mm / 1000.0  # Convert radius from mm to meters
    z_m = z_mm / 1000.0        # Convert height from mm to meters
    phi_rad = math.radians(phi_deg)  # Convert angle from degrees to radians for math functions
    
    # Calculate X and Y using standard trigonometry, and return the 3D array
    return np.array([r_xy_m * math.cos(phi_rad), r_xy_m * math.sin(phi_rad), z_m])

def generate_piston_points(center, radius, count, normal):
    """
    Generates a cluster of 3D points arranged in a flat disk (representing a speaker cone).
    It uses a Fibonacci spiral (sunflower pattern) to ensure the points are evenly distributed.
    """
    # Create an array of indices from 0.5 to count - 0.5
    indices = np.arange(0, count, dtype=float) + 0.5
    
    # Calculate the radius of each point. The square root ensures equal area distribution
    r = np.sqrt(indices / count)
    
    # Calculate the angle (theta) for each point using the golden ratio (creates the spiral)
    theta = np.pi * (1 + 5**0.5) * indices
    
    # Combine the radius and angle to place the points on a flat disk in the X-Y plane (Z=0)
    points_local = np.column_stack((r * np.cos(theta) * radius, r * np.sin(theta) * radius, np.zeros(count)))
    
    # Normalize the 'normal' vector (the direction the piston is facing) so its length is exactly 1
    normal = np.array(normal) / np.linalg.norm(normal)
    
    # The default direction our disk is facing is straight up (+Z)
    z_axis = np.array([0, 0, 1])
    
    # We need a rotation matrix (R) to tilt our flat disk so it faces the requested 'normal' direction
    if np.allclose(normal, z_axis): R = np.eye(3)
    elif np.allclose(normal, -z_axis): R = np.diag([1, -1, -1])
    else:
        # Calculate the cross product to find the axis to rotate around
        v = np.cross(z_axis, normal); s = np.linalg.norm(v); c = np.dot(z_axis, normal)
        # Create a skew-symmetric cross-product matrix (math magic for 3D rotations)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        # Use Rodrigues' rotation formula to generate the final 3x3 rotation matrix
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))
        
    # Rotate all our points using the matrix R, and then move them to the requested 'center' position
    return (points_local @ R.T) + np.array(center)

def compute_distributed_piston_spectrum(mic_pos, piston_points, fs, n_fft, c):
    """
    This is the core DSP math. It calculates the frequency response (spectrum) 
    that a microphone would hear from our simulated speaker.
    It does this by adding up the sound from every single little point on the piston.
    """
    # Calculate how many frequency 'bins' we will have. It's half the FFT size plus one for the DC (0Hz) bin.
    n_pos = n_fft // 2 + 1
    
    # Create an array of all the exact frequencies we are calculating, from 0Hz up to the Nyquist limit (half sample rate)
    freqs = np.linspace(0.0, fs/2.0, n_pos)
    
    # Calculate the 'wavenumber' (k) for every frequency. k = 2 * pi * frequency / speed_of_sound.
    k = 2.0 * np.pi * freqs / c
    
    # Initialize an array of complex numbers (magnitude and phase) to hold our total summed sound
    P_total = np.zeros(n_pos, dtype=np.complex128)
    
    # Calculate the physical distance from the microphone to every single point on the speaker piston
    dists = np.linalg.norm(piston_points - mic_pos, axis=1)
    
    # Loop through each point on the piston
    for d in dists:
        # Green's function for a point source: ( e^(-j*k*d) ) / ( 4 * pi * d )
        # The numerator creates the phase shift (delay) based on distance.
        # The denominator reduces the volume (magnitude) based on distance (inverse square law).
        # We use max(d, 1e-6) to prevent dividing by zero if the mic is exactly touching the point.
        P_total += np.exp(-1j * k * max(d, 1e-6)) / (4.0 * np.pi * max(d, 1e-6))
        
    # Average the result by dividing by the number of points on the piston
    P_total /= len(piston_points)
    
    # If we have enough frequencies, apply our smooth high-frequency roll-off (anti-aliasing)
    if n_pos > 2:
        P_total *= cosine_taper_from(freqs, HF_TAPER_START_HZ, freqs[-2])
        
    # Force the DC (0Hz) bin to be purely real, and the Nyquist bin to be exactly 0
    P_total[0] = np.real(P_total[0]); P_total[-1] = 0.0
    
    # Return the complex frequency spectrum, and the average distance from the mic to the piston
    return P_total, np.mean(dists)

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
def main():
    # Make sure our output directory exists before we try to write to it
    ensure_dir(DEFAULT_OUT_DIR)
    
    # Generate the 3D coordinates for all the points that make up our virtual speaker cone
    piston_points = generate_piston_points(SOURCE_CENTER_M, PISTON_RADIUS_M, PISTON_POINT_COUNT, PISTON_FACING_AXIS)
    
    # Load our audio settings into local variables
    fs, dur, c = DEFAULT_FS, DEFAULT_DURATION_S, DEFAULT_C
    
    # Calculate the optimal FFT size (n_fft). FFTs are fastest when their size is a power of 2 (e.g., 1024, 2048).
    # We find the smallest power of 2 that is longer than our requested audio duration.
    n_fft = int(2 ** math.ceil(math.log2(max(8, int(dur * fs)))))
    
    # Calculate exactly how many audio samples the final file should contain based on duration and sample rate
    target_samples = int(fs * dur) # Exact samples requested

    print(f"Generating Distributed Piston (Target: {target_samples} samples)")
    
    # Open the input CSV file containing our grid coordinates, and a new output CSV for our log
    with open(DEFAULT_COORDS_CSV, "r") as f_in, open(DEFAULT_OUT_DIR / "piston_dist_log.csv", "w", newline="") as f_out:
        # Create a CSV reader that treats each row as a dictionary (using column headers as keys)
        rdr = csv.DictReader(f_in)
        
        # Create a CSV writer, copying the headers from the input file and adding three new columns
        w = csv.DictWriter(f_out, fieldnames=list(rdr.fieldnames) + ["filename", "avg_dist_m", "delay_s"])
        w.writeheader()

        # Read all the rows from the input CSV into a list
        rows = list(rdr)
        
        # Loop through every row (every microphone position)
        for i, row in enumerate(rows):
            # Convert the cylindrical coordinates in the CSV to 3D Cartesian (X,Y,Z) meters
            mic_pos = _cylindrical_to_cartesian(float(row["r_xy_mm"]), float(row["phi_deg"]), float(row["z_mm"]))
            
            # Run the core DSP math to get the complex frequency spectrum for this microphone position
            P, avg_dist = compute_distributed_piston_spectrum(mic_pos, piston_points, fs, n_fft, c)
            
            # Inverse FFT
            # This is where the magic happens! We convert the frequency spectrum (P) back into a time-domain audio wave.
            ir_full = np.fft.irfft(P, n=n_fft).astype(np.float64)
            
            # --- TRUNCATION STEP ---
            # Crop the power-of-two buffer back to the user's requested length
            # And apply our volume gain modifier
            ir_truncated = ir_full[:target_samples] * VOLUME_GAIN
            
            # Construct a descriptive file name using the coordinate values
            # Using our helper functions to ensure the filename doesn't contain decimals or messy numbers
            fname = f"id{row.get('order_idx', str(i))}_r{_nearest_int_str(float(row['r_xy_mm']))}_ph{_format_decimal_for_name(float(row['phi_deg']), 1)}_z{_nearest_int_str(float(row['z_mm']))}.wav"
            
            # Save the truncated audio wave as a 32-bit floating point WAV file
            wavwrite(str(DEFAULT_OUT_DIR / fname), fs, ir_truncated.astype(np.float32))
            
            # Write a row to our log CSV, copying the original data and adding our new info
            w.writerow({**row, "filename": fname, "avg_dist_m": f"{avg_dist:.4f}", "delay_s": f"{avg_dist/c:.9f}"})
            
            # Print a progress update to the console. \r overwrites the current line.
            sys.stdout.write(f"\rProcessing {i+1}/{len(rows)}"); sys.stdout.flush()

    print(f"\nCompleted. Files saved to {DEFAULT_OUT_DIR}")

# Standard Python boilerplate to ensure 'main()' only runs if this script is executed directly (not imported)
if __name__ == "__main__":
    main()