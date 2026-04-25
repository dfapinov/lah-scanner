#!/usr/bin/env python3
"""
utils.py
================
Centralized spherical/Cartesian coordinate transformations and translation 
logic for Stage 2, Stage 4, and Stage 5.
"""

import math
import re
from typing import Tuple, Union, List, Dict
import soundfile as sf
import numpy as np
import h5py
from pathlib import Path
from scipy.special import spherical_jn, spherical_yn
import schema

# =============================================================================
# --- Data Loading & Parsing Helpers ---
# =============================================================================

def atoi(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text

def natural_keys(text: str) -> List[Union[int, str]]:
    """
    Sorts strings containing numbers in a natural way.
    Example: ['file1', 'file10', 'file2'] -> ['file1', 'file2', 'file10']
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# Regex for parsing coordinates from filenames (e.g., prefix_r1000_phn10p5_z200.wav)
_ph_pat = re.compile(
    r"""
    ^.*?                            # anything up front (e.g., id2_)
    _r(?P<rmm>-?\d+(?:p\d+)*)       # r: integer or any p-decimal chain
    _ph(?P<ph>[-np\d]+)             # phi: digits, 'p', maybe leading 'n'
    _z(?P<zmm>-?\d+(?:p\d+)*)       # z: integer or any p-decimal chain
    (?:_.*)?                        # optional suffix (like _HF), or nothing
    (?:\.[^.]+)?$                   # optional file extension (like .wav) at the end
    """,
    re.VERBOSE | re.IGNORECASE,
)

def parse_coords_from_filename(fname: str) -> Tuple[float, float, float]:
    """
    Parses spherical coordinates from a structured filename.
    Returns:
        r_sph (float): Spherical radius in meters
        theta (float): Polar angle in radians
        phi (float): Azimuthal angle in radians
    """
    m = _ph_pat.match(fname)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: '{fname}'")

    def _decode_coord(s: str) -> float:
        return float(s.replace("p", "."))

    def _decode_phi(s: str) -> float:
        neg = s.startswith("n")
        s = s[1:] if neg else s
        s = s.replace("p", ".")
        val = float(s)
        return -val if neg else val

    r_mm = _decode_coord(m.group("rmm"))
    ph_str = m.group("ph")
    z_mm = _decode_coord(m.group("zmm"))
    phi_deg = _decode_phi(ph_str)

    r_cyl_m = r_mm / 1000.0
    z_m = z_mm / 1000.0
    r_sph = math.hypot(r_cyl_m, z_m)

    theta = math.pi / 2 if z_m == 0.0 else math.atan2(r_cyl_m, z_m)
    if theta < 0: theta += math.pi
    phi = math.radians(phi_deg) % (2 * math.pi)
    
    return r_sph, theta, phi

def load_and_parse_npz(filepath: Union[str, Path]) -> Dict:
    """Loads an NPZ dataset, natural-sorts keys, and parses coordinates."""
    loaded = np.load(filepath, allow_pickle=True)
    f_all = loaded[schema.FREQS]
    data_dict = loaded[schema.COMPLEX_DATA].item()
    
    raw_filenames = sorted(data_dict.keys(), key=natural_keys)
    filenames, r_list, th_list, ph_list = [], [], [], []
    
    for fn in raw_filenames:
        try:
            rs, ts, ps = parse_coords_from_filename(fn)
            r_list.append(rs)
            th_list.append(ts)
            ph_list.append(ps)
            filenames.append(fn)
        except Exception:
            pass
            
    return {
        'freqs': f_all,
        'complex_data': data_dict,
        'filenames': filenames,
        'r_arr': np.array(r_list),
        'th_arr': np.array(th_list),
        'ph_arr': np.array(ph_list),
        'origins_mm': loaded[schema.ORIGINS_MM] if schema.ORIGINS_MM in loaded else None,
        'raw_data': loaded
    }

def load_she_h5(source: Union[str, Path, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Standardized loading of Spherical Harmonic Expansion coefficients from HDF5."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Coefficient file not found: {path}")
        with h5py.File(path, "r") as h:
            origins_mm = h.get(schema.ORIGINS_MM)
            return {
                schema.FREQS: h[schema.FREQS][()],
                schema.COEFFS: h[schema.COEFFS][()],
                schema.N_USED: h[schema.N_USED][()].astype(int),
                schema.ORIGINS_MM: origins_mm[()] if origins_mm is not None else None 
            }
    elif isinstance(source, dict):
        return source
    else:
        raise TypeError("Input must be a file path or a dictionary.")

# =============================================================================
# --- File Writers ---
# =============================================================================

def write_wav(audio_data: np.ndarray, fs: int, filepath: Union[str, Path], desc: str = ""):
    """Writes a NumPy array to a 32-bit float WAV file."""
    # Ensure data is float32 for writing
    audio_data = audio_data.astype(np.float32)
    # Normalize if peak is > 1.0 to prevent clipping
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data /= max_val
    sf.write(filepath, audio_data, fs, subtype='FLOAT')

# =============================================================================
# --- Coordinate Systems Conversion ---
# =============================================================================

def spherical_to_cartesian(
    r: Union[float, np.ndarray], 
    th: Union[float, np.ndarray], 
    ph: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Converts spherical coordinates (r, theta, phi) to Cartesian (x, y, z).
    Theta is the polar angle (0 to pi), Phi is the azimuthal angle.
    """
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    return x, y, z

def cartesian_to_spherical(
    x: Union[float, np.ndarray], 
    y: Union[float, np.ndarray], 
    z: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Converts Cartesian coordinates (x, y, z) to spherical (r, theta, phi).
    Includes safe fallbacks to prevent division by zero at the origin.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Safely handle potential division by zero
    if isinstance(r, np.ndarray):
        r_safe = np.maximum(r, 1e-12)
    else:
        r_safe = max(r, 1e-12)

    # Use arccos for the polar angle, clipped to handle floating point anomalies
    th = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    ph = np.arctan2(y, x)
    
    return r, th, ph

# =============================================================================
# --- Translate Coordinates X,Y,Z (Acoustic Origin)---
# =============================================================================

def translate_coordinates(
    r: Union[float, np.ndarray], 
    th: Union[float, np.ndarray], 
    ph: Union[float, np.ndarray], 
    origin_xyz: Union[list, tuple, np.ndarray],
    inverse: bool = False
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Translates spherical coordinates by shifting the Cartesian origin.
    
    Parameters:
        r, th, ph: Initial spherical coordinates.
        origin_xyz: The (x, y, z) offset to apply.
        inverse: 
            - If False (Default): Shifts the grid relative to the new origin 
              (xyz_new = xyz - origin). Used in Stages 2 & 4.
            - If True: Performs the mathematical inverse shift 
              (xyz_new = xyz + origin). Can be used in Stage 5 evaluations.
    """
    x, y, z = spherical_to_cartesian(r, th, ph)
    
    sign = 1 if inverse else -1
    x_new = x + sign * origin_xyz[0]
    y_new = y + sign * origin_xyz[1]
    z_new = z + sign * origin_xyz[2]
    
    return cartesian_to_spherical(x_new, y_new, z_new)

# =============================================================================
# --- Mathematical Helpers ---
# =============================================================================

def hankel1(n: Union[int, np.ndarray], z: np.ndarray) -> np.ndarray:
    """Return spherical Hankel function of the first kind (outgoing)."""
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)

def hankel2(n: Union[int, np.ndarray], z: np.ndarray) -> np.ndarray:
    """Return spherical Hankel function of the second kind (outgoing for e^{-jwt} convention)."""
    return spherical_jn(n, z) - 1j * spherical_yn(n, z)

# =============================================================================
# --- SHE Order_N Constraint Helpers ---
# =============================================================================

def get_grid_limit(th: np.ndarray, ph: np.ndarray) -> Tuple[int, int]:
    """
    Calculates the maximum resolvable Harmonic Order (N_grid) based on spatial Nyquist.
    Returns a tuple of (N_grid, M_unique_points).
    """
    coords_stacked = np.column_stack((np.round(th, 4), np.round(ph, 4)))
    M_unique = len(np.unique(coords_stacked, axis=0))
    N_grid = int(math.floor(math.sqrt(M_unique / 2) - 1))
    return max(0, N_grid), M_unique

def get_kr_limit(f_hz: float, r: Union[float, np.ndarray], c_sound: float = 343.0, kr_offset: float = 2.0) -> int:
    """
    Calculates the maximum physical Harmonic Order (N_kr) to prevent wave function instability.
    The kr_offset is the "+x" in the kr+x rule.
    """
    k = 2 * math.pi * f_hz / c_sound
    kr = r * k
    return int(math.floor(np.max(kr) + kr_offset))


# =============================================================================
# --- Mic Calibration Helper ---
# =============================================================================

def apply_mic_calibration(complex_data: np.ndarray, target_freqs: np.ndarray, cal_file_path: Union[str, Path], mode: str = 'subtract', fade_octaves: float = 1.0) -> np.ndarray:
    """
    Loads a mic calibration file, derives the minimum phase response, interpolates
    to match `target_freqs`, and applies the correction directly to the `complex_data`.
    Dynamically handles 1D (freqs,) and 2D (freqs, points) data structures.
    """
    path = Path(cal_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    cal_data = np.loadtxt(path, comments=['#', '*'])
    cal_freq = cal_data[:, 0]
    cal_amp = cal_data[:, 1]
    
    # 1. Create a dense linear frequency grid for minimum phase generation
    fs_sim = 192000
    n_fft = 262144
    f_lin = np.fft.rfftfreq(n_fft, 1 / fs_sim)
    
    # 2. Interpolate magnitude to linear grid
    mag_lin_db = np.interp(f_lin, cal_freq, cal_amp)
    
    # 3. Apply Cosine Fade out-of-band (OOB) to 0 dB
    f_min, f_max = cal_freq[0], cal_freq[-1]
    f_lower_fade = f_min / (2.0 ** fade_octaves)
    f_upper_fade = f_max * (2.0 ** fade_octaves)
    
    weights = np.ones_like(f_lin)
    
    lower_idx = (f_lin > f_lower_fade) & (f_lin < f_min)
    if np.any(lower_idx):
        norm_f = np.log2(f_lin[lower_idx] / f_lower_fade) / fade_octaves
        weights[lower_idx] = 0.5 * (1 - np.cos(np.pi * norm_f))
        
    upper_idx = (f_lin > f_max) & (f_lin < f_upper_fade)
    if np.any(upper_idx):
        norm_f = np.log2(f_lin[upper_idx] / f_max) / fade_octaves
        weights[upper_idx] = 0.5 * (1 + np.cos(np.pi * norm_f))
        
    weights[f_lin <= f_lower_fade] = 0.0
    weights[f_lin >= f_upper_fade] = 0.0
    
    mag_lin_db *= weights
    
    # 4. Generate Minimum Phase via Real Cepstrum
    ln_mag = mag_lin_db / 8.685889638  # Convert dB to natural log (Neper)
    ceps = np.fft.irfft(ln_mag, n=n_fft)
    
    w = np.zeros(n_fft)
    w[0] = 1.0
    w[1:n_fft//2] = 2.0
    w[n_fft//2] = 1.0
    
    complex_spec = np.fft.rfft(ceps * w)
    phase_rad_lin = np.imag(complex_spec)
    
    # 5. Interpolate back to target_freqs
    interp_cal_amp = np.interp(target_freqs, f_lin, mag_lin_db)
    interp_cal_phase = np.interp(target_freqs, f_lin, phase_rad_lin)

    if mode.lower() == 'subtract':
        cal_mult = (10 ** (-interp_cal_amp / 20.0)) * np.exp(1j * -interp_cal_phase)
    elif mode.lower() == 'add':
        cal_mult = (10 ** (interp_cal_amp / 20.0)) * np.exp(1j * interp_cal_phase)
    else:
        raise ValueError("Calibration mode must be 'add' or 'subtract'")
        
    # Broadcast the 1D complex multiplier array across the input data.
    # Using .T (transpose) forces the frequency axis to the end, aligning it perfectly
    # for NumPy's native broadcasting. This makes the math shape-agnostic, effortlessly
    # handling both 1D arrays (e.g., individual IRs) and 2D matrices (e.g., arc sweeps)
    # before returning the array to its original shape.
    return (complex_data.T * cal_mult).T

