# config_capture.py
"""
User settings for the capture scripts that generate measruemtn grids, drive robotic microphone, sweep and IR capture.
"""

""" Grid Generator """
# ───────────────────────────────────────────────
# Cylinder and pattern settings:
cyl_radius              = 0.25    # Cylinder radius (m)
cyl_height              = 0.80     # Cylinder height (m)
num_points              = 500     # Points per spiral (total = forward + reverse)
cap_fraction            = 0.15     # Fraction on end‐caps (0..1) Adjust to optimise vs. cylider height.
jitter_enabled          = True    # Enable radial jitter
jitter_range_mm         = 30      # Jitter range in mm (± half)
generate_reverse_spiral = True    # Make second (reverse) spiral
z_rotation_deg          = 137.5   # Rotate second spiral around Z (deg)
flip_poles              = False   # Flip Z sign of second spiral
bottom_cutoff_mm        = 100      # Remove bottom‐cap points within this radius from center (mm) for support pole

# Optimisation settings:
k_neighbors   = 20      # how many points to group for nearest neighbour proximity check
threshold_mm  = 8       # radial distance to neighbour threshold (mm)
num_runs      = 100     # max runs, if not run_until_zero
run_until_zero = False  # if True, loop until zero flagged

""" Path Planner """
# ───────────────────────────────────────────────
INPUT_CSV          = "best_coordinates.csv" # Default "best_coordinates.csv"
OUTPUT_CSV         = "scan_path.csv" # Default "scan_path.csv"

#Advanced:
DELTA_THETA_DEG    = 7.5   # θ-bin width (deg). Smaller → more bins, smoother θ progression.
CAP_TOL_MM         = 3.0   # Points within ±CAP_TOL_MM of min/max z are treated as caps (top/bottom).
SIDE_SNAKE_START   = "up"  # Initial z traversal direction for sidewall bins: "up" or "down".
CAP_ORDER          = "side_then_caps"  # Handle all side bins first, then caps; or "interleave_caps".
CAP_RADIAL_ORDER   = "inner_to_outer"  # For caps: sweep radially "inner_to_outer" or "outer_to_inner".
PRINT_SUMMARY = True  # Whether to print counts and θ-bin info after writing the path


""" G-code driver """
# ───────────────────────────────────────────────

# No gcode driver written yet. Needs integration with hardware and run_capture.py script.

""" Audio Capture """
# ───────────────────────────────────────────────
# Audio capture is spread across 3 scripts below:

# ─────────── run_capture.py settings:

COORD_FILE = "scan_path.csv" # Input CSV file name (same directory as script)
OUTDIR = "demo_outputs" # output directory (relative to script)
MAX_POINTS = 5 # Maximum number of points to capture in demo mode
SAVE_MIC_LOOP_SIG = False # True = keep signal, loopback and raw mic sweep files. False = keep only IR.

# ─────────── sweep_function.py settings: 

""" OUT/IN device indices and channel numbers are host-dependent.
 Run "python sweep_function.py" at terminal to list devices by number.
 If you add or remove a device the numbers may change.
 ASIO is preferred, WASAPI secondary, MME finally.udio Capture """

 
# Device setup:

IN_DEV  = 20         # Input device index  (mic + loopback capture)
OUT_DEV = 20         # Output device index (speaker + loopback reference)

# Channel numbers (0-based within each device)
IN_CH_MIC   = 0      # Microphone input channel
IN_CH_LOOP  = 1      # Electrical loopback input channel
OUT_CH_SPKR = 0      # Speaker output channel
OUT_CH_REF  = 1      # Reference/loopback output channel

# Sweep parameters:
FS               = 96_000     # Sampling rate (Hz)
SWEEP_DUR_S      = 6.0        # Duration of exponential sine sweep (seconds)
F1_HZ            = 10.0       # Start frequency (Hz)
F2_HZ            = None       # End frequency (None → 0.48 × FS)
SWEEP_LEVEL_DBFS = -6.0      # Playback level (dBFS) adjust to avoid clipping at the DAC.
FADE_MS          = 15.0       # Fade-in/out at sweep ends (ms)

# Timing:
PRE_SIL_MS  = 100.0           # Silence before sweep (ms)
POST_SIL_MS = 2000.0           # Silence after sweep (ms) (Extend to account for system latency)

# Conditioning:
MIC_TAIL_TAPER_MS  = 20.0               # Fade-out after sweep to suppress noise (ms)

# Streaming:
BLOCKSIZE        = 2048         # I/O buffer size (For ASIO this should match the H/W buffer in device control panel).
WASAPI_EXCLUSIVE = True       # True = WASAPI Exclusive mode, False = Shared

# ─────────── make_ir_function.py settings: 

# Basic settings for IR generation from the recorded sweeps:
GATE_FREQ_HZ = 5.0        # Gate length to trim IR, set as 4*cycles at desired frequency extension. (e.g. 10 Hz → 4 cycles = 800 ms)
FADE_RATIO   = 0.12        # Fraction of gate length used for the half-Hanning fade-out

# Advanced settings:
ENABLE_REGULARIZATION = True
ENABLE_MANUAL_TAPERS  = True
PLOT                  = False # Plot frequency manitude before IRFFT for debugging.

LF_TAPER_START_HZ = 10.0       # HP filter to remove low frequency noise. Helps to ensure minimal energy near DC.
HF_TAPER_START_HZ = 24_000.0   # LP filter to remove high frequency noise. Helps to ensure minimal energy near nquist.

REGU_INSIDE    = 1e-10     # Weak reguarlisation epsilon inside usable band. (Inside LF_TAPER_START_HZ to HF_TAPER_START_HZ)
REGU_OUTSIDE   = 1.0       # Strong reguarlisation epsilon outside the band. (Outside LF_TAPER_START_HZ to HF_TAPER_START_HZ)
REGU_XFADE_FRAC = 0.10     # Reguarlisation crossfade width as fraction of band width (each side)


