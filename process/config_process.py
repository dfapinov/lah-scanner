"""
config_process.py

Configuration settings for the scripts that process IRs via spherical harmonic expansion to generate FRD response files and system IRs.
"""

""" Stage 1 Configuration – IR processing """
# ───────────────────────────────────────────────
INPUT_IR_DIR_REL  = "input_irs"     # Directory (relative to root) where your raw IR .wav files live.
OUTPUT_DIR_REL    = "outputs"      # Directory (relative to root) for all stage outputs (lf_irs, hf_irs, debug, response_files, etc).
# ───────────────────────
LF_GATE_FREQ_HZ   = 20             # LF extension frequency used to set LF gate length. e.g. 20 Hz = 200 ms gate (4× wavelength).
# Note: HF gate length is determined by the CROSSOVER_FREQ_HZ parameter in Stage 2, so is not listed here.
HP_CUTOFF_HZ      = None           # Hz – high-pass filter cut-off. None to disable. (For DC removal, applied to ALL outputs.)
ZERO_PAD_MS       = None           # ms – zero-pad at IR start, or None to disable.
FADE_RATIO        = 0.12           # Fraction of total gate length to apply fade-out.
TARGET_PEAK_DB    = -3.0           # dBFS target level for auto-gain. Calculated from loudest IR and applied equally.
ENABLE_AUTO_GAIN  = False          # True to normalize to TARGET_PEAK_DB. Relative level between files is preserved.
# ───────────────────────────────────────────────


""" Stage 2 Configuration - Spherical Harmonic Expansion Solver """
# ───────────────────────────────────────────────
"""
Note: CROSSOVER_FREQ_HZ = 1000hz is reasonable if the closest boundary to the measruemtn system is ~1.0m.
Use the script crossover_check.py to estimate an appropritate CROSSOVER_FREQ_HZ for your enviroment.
A good starting setup for stop condition is using USE_KR_LIMIT = True and MAX_FALLBACK_N = 20.
Other stop conditions can remain off, unless you want to play.
"""

# Basic settings:
CROSSOVER_FREQ_HZ  = 1_000.0       # LF/HF split (Hz). Determines HF gate length in Stage 1 (higher freq = shorter HF gate).
OCT_RES            = 24            # Octave resolution (recommend 1/12, 1/24, or 1/48).
#USE_FILENAME_PARSE = True          # True = parse coordinates from filename (e.g., (r_mm, φ_deg, z_mm)_LF.wav). False = use metadata.csv.
SPEED_OF_SOUND     = 343.0         # Speed of sound in m/s.

# Advanced settings:
LF_FMIN = 5.0                     # Lower bound for the LF band (best left at 5 Hz).
HF_FMAX = 24_000.0                # Upper bound for the HF band (typically 24 kHz). Grid spatial density must be fine enough to resolve wavelength.

# Ridge regularisation stabilises the solve by penalising large coefficients.
# (Adds λ·I to AᴴA before solving, suppressing unstable directions.)
RIDGE_LAMBDA       = 0             # None = auto, 0 = off, >0 = fixed.
RIDGE_COND_TARGET  = 5e1           # Target condition number after regularisation. Lower = stronger smoothing.

# ───── Stop condition for increasing harmonic order N ─────

# KR limit method (theoretical N-order bound):
USE_KR_LIMIT       = True          # ⌊max(kr)+2⌋ limit based on radius & wavelength.
N_MIN              = 1             # Minimum N to solve to, regardless of other stop conditions.
MAX_FALLBACK_N     = 15            # Hard upper cap on harmonic order N.

# Condition-number spike method:
USE_COND_SPIKE     = False         # Enable condition-number spike stop method.
COND_SPIKE_FACTOR  = 15            # Stop if condition number spikes by this factor.

# Grid density method:
USE_GRID_LIMIT     = True         # √M−1 limit based on input-grid density. Guardrail against too few measurment points.

# Residual-error method (kept for reference; not effective metric for stop condition):
USE_RESID_THRESHOLD       = False  # Enable residual-improvement check.
RESID_IMPROVE_PCT         = 20.0    # Stop if residual improvement < this %.
RESID_ERROR_THRESHOLD_PCT = 50.0   # Only check once residual ≤ this %.
# ───────────────────────────────────────────────


""" Stage 3 Configuration - Sound Field Seperation and Response File Generation """
# ───────────────────────────────────────────────

# LF and HF coefficient paths
COEFF_LF_PATH = "outputs/coefficients/asph_coeffs_LF.h5"  # Low-frequency coefficients
COEFF_HF_PATH = "outputs/coefficients/asph_coeffs_HF.h5"  # High-frequency coefficients
OUTPUT_DIR    = "outputs/response_files"                  # Output directory for FRD files and coordinates.txt

OBSERVATION_MODE = "Internal" # Wavefront observation mode (Internal = anechoic response. External = Room response, Full = both).
SUBTRACT_TOF       = True         # Subtract time-of-flight (TOF) phase to reduce FRD phase wrapping.

# When USE_COORD_LIST=True, the TOF corresponding to the smallest radius is subtracted
# from all coordinates. This preserves the relative phase between points at different distances.

# ────────── Mode 1: Measurement Arc Sweep ─────────────
RANGE_DEG     = 180        # Sweep ± range in degrees (e.g. 90 = ±90°)
DIST_MIC      = 2.0       # Radius of measurement arc (m)
INCREMENT_DEG = 5        # Increment of sweep (°)
DIRECTION     = "horizontal"  # "horizontal", "vertical", or "hor_vert"

# Cartesian offset (m) to shift mic arc (e.g. align with tweeter)
# This applies a positional shift to the observation points only; it does not transform
# the underlying SHE coefficients or the original measurement coordinates.
OFFSET_MIC_X = 0.0
OFFSET_MIC_Y = 0.0
OFFSET_MIC_Z = 0.0

# Zero (central) coordinate angles (°)
# Usually the speaker front; depends on measurement orientation.
ZERO_THETA_DEG = 90   # Polar/elevation (0° = north pole, 90° = equator)
ZERO_PHI_DEG   = 0  # Azimuth (0° = reference, increasing toward +φ)

# ────────── Mode 2: Explicit Coordinate List ─────────────
USE_COORD_LIST = False  # Override sweep mode and use explicit coordinates

# This list can help find the front axis direction for use as the zero coordinate above.
COORD_LIST = [
    # (theta_deg, phi_deg, radius_m)
    (0,   0,   2.0),    # +Z (top)
    (90,  0,   2.0),    # +X (front)
    (90, 90,  2.0),    # +Y (right)
    (90, 180, 2.0),    # -X (back)
    (90, 270, 2.0),    # -Y (left)
    (180, 0,  2.0),    # -Z (bottom)
]


""" Stage 4 Configuration - IR Generation from System Response """
# ───────────────────────────────────────────────
SRC_NPZ = "outputs/response_files/merged_complex/" # Source file directory containing complex pressures files.

FS_OUT = 44_100  # Output sampling rate (Hz) (Keep it low so source data has good phase full band)
MAG_LF_TAPER_START_HZ = 15.0  # Start frequency for LF magnitude taper (Hz)
MAG_HF_TAPER_START_HZ = 20_000.0  # Start frequency for HF magnitude taper (Hz)
FADE_RATIO = 0.2  # Portion of gate length used for fade
IR_GEN_GATE_MS = 300.0  # Gate length in milliseconds - Recommend 4* wavelength of lowest frequency in ms. e.g. 20Hz = 200ms
ENFORCE_CAUSAL_MINPHASE = False # Enforce causal minimum-phase IR via Hilbert transform.
TRIM_TO_GATE = False # chops off the extra tail left over from power-of-two FFT padding.
NORMALIZE_TO_DBFS = True   # If True, scale output IR to target dBFS (default −6 dB)
TARGET_PEAK_DBFS = -6.0    # Desired output peak level in dBFS
ENABLE_PLOTS = False # Debug plots

