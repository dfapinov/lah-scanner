"""
SHE processing pipeline configuration user settings.
"""

""" Stage 1 – FDW & Batch FFT processing """
# ────────────────────────────────────────────
INPUT_DIR_FDW  = "input_irs"     # Directory where raw .wav IRs are located.
OUTPUT_DIR_FDW    = "outputs"       # Root directory for all pipeline outputs.
OUTPUT_FILENAME_FDW   = "complex_data.npz" # Name of the consolidated complex data file. complex_data_centered.npz

# --- FDW (Frequency Dependent Window) Parameters ---
FDW_RFT_MS        = 5.0             # Reflection Free Time (ms): Defines the fixed window length at high frequencies.
FDW_OCT_RES       = 12              # Target Octave Resolution: Sets the fractional octave smoothing (e.g., 12 for 1/12th oct).
FDW_MAX_CAP_MS    = 200  #None      # Optional cap (ms) on the maximum window length. Will limit oct res at LF.

# --- Smoothing Options ---
ENABLE_SMOOTHING = True        # Enable or dissable complex smoothing
SMOOTHING_OCT_RES = FDW_OCT_RES*2       # Default = use FDW_OCT_RES*2 so that smoothing does not reduce resolution of initial FDW.
KEEP_RAW_AND_SMOOTHED = False   # Save both files if True

# --- Gain ---
ENABLE_AUTO_GAIN  = True            # Enable global normalization across all files in the batch.
TARGET_PEAK_DB    = -3.0            # Target peak level (dBFS) for the loudest file in the set.

# --- Advanced ---
FDW_ALPHA_HF      = 0.2             # Taper alpha for High Frequencies (0.0=Rectangular, 1.0=Hann).
FDW_ALPHA_LF      = 1.0             # Taper alpha for Low Frequencies(0.0=Rectangular, 1.0=Hann).
FDW_WINDOWS_PER_OCT = 3             # Windows per octave. Interpolation is performed in complex domain between windows.
PEAK_DETECT_THRESHOLD_DB = -12.0 # Peak detection finds loudest peak, then searches for earlier peaks above this threshold. A reflection may be louder than the true direct sound peak.

# --- Visualization ---
PLOT_OUTPUT       = True            # If True, launches the interactive data viewer after processing.
FDW_F_MIN         = 20.0            # Minimum frequency (Hz) for the X axis in the plot view of the FDW analysis. Visual only.


""" Stage 2 - Acoustic Origin Search """
# ───────────────────────────────────────────────
INPUT_DIR_ORIGINS      = OUTPUT_DIR_FDW       # Use variable from Stage 1 or point directly e.g."outputs"
INPUT_FILENAME_ORIGINS = OUTPUT_FILENAME_FDW  # Use variable from Stage 1 or point directly e.g. "complex_data.npz"
OUTPUT_FILENAME_ORIGINS = OUTPUT_FILENAME_FDW # Use variable from Stage 1 or point directly e.g. "complex_data_origins.npz"

# --- Search Strategy Settings ---

TWEETER_COORDS_MM = (30.0, 0.0, 50.0)    # Initial seed coordinate (X, Y, Z) in mm for the highest frequency
FREQ_START_HZ = 20.0 # Low frequency
FREQ_END_HZ = 20000.0 # High frequency
OCTAVE_RESOLUTION = 1/6


# --- Full 3D Volumetric Bounds for Full Grid Scan --
# For 'trouble' frequencies or 3D landscape visualization
X_BOUNDS = (-200.0, 150.0)
Y_BOUNDS = (-100.0, 100.0)
Z_BOUNDS = (-200.0, 200.0)
GRID_RES_MM = 5.0  

# --- Visualize Results --
PLOT_RESULTS_ORIGINS = True # Set to True to open the validation plots at the end.


# --- Advanced ---
INITIAL_SIMPLEX_STEP = 15.0  # Size of the simplex when dropped (it can adapt).
MAX_ITERATIONS = 50  # How many steps the simplex takes to find the best result.
TARGET_N_MAX_ORIGINS = 4 # A lower order_N acts like a low-pass filter for spatial detail.

# --- Generate 3D Landscape ---
ENABLE_FULL_GRID_SCAN = False # Full CPU-heavy volumetric rendering for every frequency. Mostly for diagnostic or learning.
USE_CACHE = False  # Load a full render file to view it again.
READ_CACHE_FILE = "origins_cache_3D.pkl" # Specify file to read if USE_CACHE is True


""" Stage 3 - Find Optimal Solve Settings """
# ───────────────────────────────────────────────
INPUT_DIR_OPTI      = OUTPUT_DIR_FDW
INPUT_FILENAME_OPTI = OUTPUT_FILENAME_ORIGINS

TEST_ORDER_RANGE    = (4, 15)           # (min_N, max_N)
TEST_START_DB_RANGE = (-20.0, -60.0)    # (highest_dB, lowest_dB) automated in 5dB steps
TEST_LAMBDA_RANGE   = (0.0000001, 0.01)      # (min_lambda, max_lambda) logarithmic test steps
TEST_DB_TRANSITION_SPAN  = 20.0              # The dB range between start of damping and maximum damping


""" Stage 4 - Spherical Harmonic Expansion Solver """
# ───────────────────────────────────────────────
INPUT_FILENAME_SHE = OUTPUT_FILENAME_ORIGINS
OUTPUT_FILENAME_SHE = "she_coeffs.h5"
INPUT_DIR_SHE      = "outputs"
OUTPUT_DIR_SHE     = "outputs/coefficients"

# --- Order N (Harmonic Degree) ---
TARGET_N_MAX     = 8           # Hard upper cap on harmonic order N.
KR_OFFSET = 2           # Default=2. User adjustable from 0 to 2. Shifts the growth of Order_N with frequency.

USE_MANUAL_TABLE = False # Set to True to define custom order_N using the table below. Enter any frequency and order N.
MANUAL_ORDER_TABLE = {
    350:   3,
    500:   4,
    625:   5,
    750:   6,
    875:   7,
    1500:  8,
    2100:  9,
    2800:  10,
    3500:  11,
    4100:  12,
    4600:  13,
    5000:  14,
    6000:  15  # Covers 6kHz up to 20kHz
}

# --- Reguarlization ---
NOISE_FLOOR_START_DB = -30.0 # The point (in dB relative to the peak mode) where damping starts.
NOISE_FLOOR_MAX_DB = -50.0 # The point (in dB relative to the peak mode) where damping hits MAX_LAMBDA.
MAX_LAMBDA = 0.00000010 # The maximum penalty applied to modes below NOISE_FLOOR_MAX_DB. Prevents coefficient explosion with ill-conditioning.


CONDITION_METRICS = True  # Show matrix condition number for analysis.
USE_OPTIMIZED_ORIGINS = True # Essential for best fit.

""" Stage 5 - Sound Field Seperation and Response File Generation """
# ───────────────────────────────────────────────

# coefficient paths
COEFF_PATH = "outputs/coefficients/she_coeffs.h5"  # High-frequency coefficients
OUTPUT_DIR    = "outputs/response_files"                  # Output directory for FRD files and coordinates.txt
FRD_PREFIX = "Response"   # base name for arc-sweep FRD file exports (e.g., "response", "speaker-1", etc.)
OBSERVATION_MODE = "Internal" # Wavefront observation mode (Internal = anechoic response. External = Room response, Full = both).
GENERATE_IR_FILES = False # Set to True to generate .wav impulse responses from complex pressures
SUBTRACT_TOF       = False         # Subtract time-of-flight (TOF) phase to reduce FRD phase wrapping.
USE_OPTIMIZED_ORIGINS = True

# When USE_COORD_LIST=True, the TOF corresponding to the smallest radius is subtracted
# from all coordinates. This preserves the relative phase between points at different distances.

# ────────── Microphone Calibration ─────────────
APPLY_MIC_CALIBRATION = True
MIC_CALIBRATION_FILE  = "MM1_Mic_Cal.txt"
MIC_CALIBRATION_MODE  = "subtract"  # 'add' or 'subtract'
MIC_CALIBRATION_FADE_OCTAVES = 1.0  # Octaves to fade out of band correction to zero



# ────────────── Global Settings ──────────────
# Cartesian offset (m) to shift acoustic evaluation origin relative to measurment grid origin (e.g. to align with tweeter)

OFFSET_MIC_X = 0.0
OFFSET_MIC_Y = 0.0
OFFSET_MIC_Z = 0.0 #1795


# ────────────── Spherical Orientation  ──────────────
# Acoustic Evaluation Zero Degree angle, relative to spherical harmonic expansion coordiante system.
# Usually Theta 0 degrees in SHE is UP (north pole) while in acoustics 0 degrees means FRONT FACING.
# Therfore, typically to match acoustics expectation Zero_Theta = 90 degrees, but depends on DUT orientation. 

ZERO_THETA_DEG = 90   # Polar/elevation (0° = north pole, 90° = equator)
ZERO_PHI_DEG   = 180  # Azimuth (0° = +X, 180° = -X / front facing)

# ───────Mode 1: CTA-2034 (Spinorama) ──────────────
CTA_MODE = False  # Set to True to generate full CTA-2034-A metrics. Overides Mode 2 and Mode 3.
                 # CTA-2034 honours Dist_MIC in the below settings. 2.0 m is standard.

# ────────── Mode 2: Measurement Arc Sweep ─────────────
RANGE_DEG     = 0        # Sweep ± range in degrees (e.g. 90 = ±90°)
DIST_MIC      = 0.30      # Radius of measurement arc (m) 0.305 for toms mid-layer
INCREMENT_DEG = 10        # Increment of sweep (°)
DIRECTION     = "horizontal"  # "horizontal", "vertical", or "hor_vert"

# ────────── Mode 3: Manual Coordinate List ─────────────
USE_COORD_LIST = False  # Override sweep mode and use explicit coordinates



# This list can help find the front axis direction for use as the zero coordinate above.
COORD_LIST = [
    # (theta_deg, phi_deg, radius_m)
    (0,   0,   2.0),    # +Z (top)
    (90,  0,   2.0),    # +X (back)
    (90, 90,  2.0),    # +Y (left)
    (90, 180, 2.0),    # -X (front)
    (90, 270, 2.0),    # -Y (right)
    (180, 0,  2.0),    # -Z (bottom)
]

# ────────── Global Setting ─────────────
SPEED_OF_SOUND     = 343.0         # Speed of sound in m/s.