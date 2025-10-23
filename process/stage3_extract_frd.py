#!/usr/bin/env python3
"""
Stage 3 – Extract FRD files from SHE coefficients and apply sound-field separation
=================================================================================

Description:
------------

This script evaluates the complex pressure at one or more observation points
based on precomputed Spherical Harmonic Expansion (SHE) coefficients.  
It can separate internal and external sound-field components and merge low- and
high-frequency bands into composite FRD files.

Usage:
------

Configured via:  config_process.py

Run directly ar command line:
    python stage3_extract_frd_config4.py
    
Input:
    • COEFF_LF_PATH  – Low-frequency coefficient HDF5 file  
    • COEFF_HF_PATH  – High-frequency coefficient HDF5 file  
    
Output:
    • OUTPUT_DIR/*_merged.frd          – merged LF/HF FRDs  
    • OUTPUT_DIR/merged_complex/*.npz  – complex spectra for later IFFT  
    • OUTPUT_DIR/lf_response/*.frd     – LF band FRDs  
    • OUTPUT_DIR/hf_response/*.frd     – HF band FRDs  
    • OUTPUT_DIR/coordinates.txt       – evaluated coordinates and labels
    

Code Pipeline Overview
-------------------------
 1. Load LF and HF spherical harmonic coefficients from .h5 files.
 2. Generate target observation points:
       • From COORD_LIST (explicit) or
       • By sweeping in θ/φ according to RANGE_DEG, INCREMENT_DEG, and DIRECTION.
 3. Evaluate complex pressure at each target using the SHE coefficients.
 4. Optionally subtract time-of-flight (TOF) phase for flattened FRD phase.
 5. Write output files:
       • *_LF.frd  → low-frequency band response
       • *_HF.frd  → high-frequency band response
       • <VituixCAD pattern>.frd → merged LF+HF response
       • *_merged_complex.npz → complex spectrum for IFFT
 6. Save coordinates.txt listing all evaluated directions and radii.
    
"""

from __future__ import annotations  # Allow postponed evaluation of type hints

import math                         # For trigonometric and geometry operations
from pathlib import Path             # Cross-platform file-path handling
from typing import List, Tuple       # Type hints for readability

import h5py                          # Read/write HDF5 coefficient files
import numpy as np                   # Numerical arrays
from scipy.special import spherical_jn, spherical_yn, sph_harm_y  # Spherical funcs

# Import configuration parameters from config_process.py
from config_process import (
    COEFF_LF_PATH,     # Path to low-frequency SHE coefficient file (.h5)
    COEFF_HF_PATH,     # Path to high-frequency SHE coefficient file (.h5)
    OUTPUT_DIR,        # Output directory for FRD files and coordinate list
    OBSERVATION_MODE,  # Field type to evaluate: "Internal", "External", or "Full"
    USE_COORD_LIST,    # True = use explicit coordinate list; False = perform angular sweep
    COORD_LIST,        # List of (theta, phi [, radius]) tuples if USE_COORD_LIST=True
    RANGE_DEG,         # ± angular range for sweep mode (degrees)
    INCREMENT_DEG,     # Step size between measurement angles (degrees)
    DIRECTION,         # Sweep direction: "horizontal", "vertical", or "hor_vert"
    ZERO_THETA_DEG,    # Reference polar angle (degrees) for sweep centre
    ZERO_PHI_DEG,      # Reference azimuth angle (degrees) for sweep centre
    DIST_MIC,          # Default radius (metres) for measurement arc
    OFFSET_MIC_X,      # Cartesian X offset (metres) to shift microphone position
    OFFSET_MIC_Y,      # Cartesian Y offset (metres) to shift microphone position
    OFFSET_MIC_Z,      # Cartesian Z offset (metres) to shift microphone position
    SUBTRACT_TOF,      # Whether to subtract time-of-flight phase (True/False)
    SPEED_OF_SOUND,    # Speed of sound (m/s) used for TOF and wave calculations
)


# -------------------------------------------------
# Constants and printing precision
# -------------------------------------------------
np.set_printoptions(precision=6, suppress=True)  # Neater console printing

# -------------------------------------------------
# Mathematical helpers
# -------------------------------------------------

def hankel1(n: int, z: np.ndarray | float) -> np.ndarray:
    """Return spherical Hankel function of the first kind."""
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)

def load_coeff_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load frequency axis, coefficients, and used orders from HDF5 file."""
    with h5py.File(path, "r") as h:
        freqs = h["freqs"][()]          # Frequency bins (Hz)
        coeffs = h["coeffs"][()]        # Complex coefficients per frequency
        n_used = h["N_used"][()].astype(int)  # Order N used per frequency
    return freqs, coeffs, n_used

# -------------------------------------------------
# Generate target coordinates from config
# -------------------------------------------------

def _format_int_angle(val: int) -> str:
    """Return integer angle as string."""
    return str(val)

def _format_signed_deg(val: int) -> str:
    """Return signed angle with explicit +/-, or 0."""
    if val == 0:
        return "0"
    return f"+{val}" if val > 0 else f"-{abs(val)}"

def _radius_to_mm_str(r: float) -> str:
    """Return distance in millimetres as '<mm>mm' (no leading 'r')."""
    r_mm = int(round(r * 1000))
    return f"{r_mm}mm"

def get_targets() -> List[Tuple[np.ndarray, str]]:
    """Generate a list of 3-D Cartesian target coordinates and filename prefixes."""
    default_r = float(DIST_MIC)     # Default mic radius from config
    off_x = float(OFFSET_MIC_X)     # Cartesian offsets (mic placement)
    off_y = float(OFFSET_MIC_Y)
    off_z = float(OFFSET_MIC_Z)
    targets: List[Tuple[np.ndarray, str]] = []  # List of (XYZ, prefix)

    if USE_COORD_LIST:  # If explicit coordinate list mode
        for entry in COORD_LIST:
            # Accept (theta, phi) or (theta, phi, radius)
            if len(entry) == 2:
                theta_deg, phi_deg = entry; r = default_r
            elif len(entry) == 3:
                theta_deg, phi_deg, r = entry
            else:
                raise ValueError("Each COORD_LIST entry must be (theta, phi) or (theta, phi, radius)")

            # Convert spherical to Cartesian and apply offset
            theta = math.radians(theta_deg)
            phi = math.radians(phi_deg)
            x = r * math.sin(theta) * math.cos(phi) + off_x
            y = r * math.sin(theta) * math.sin(phi) + off_y
            z = r * math.cos(theta) + off_z

            # NOTE: For coord-list we keep the descriptive style; VituixCAD hor/ver
            # naming is only well-defined for plane sweeps.
            prefix = f"th{_format_int_angle(theta_deg)}_ph{_format_int_angle(phi_deg)}_{_radius_to_mm_str(r)}"
            targets.append((np.array([x, y, z]), prefix))
    else:  # Generate sweeps horizontally and/or vertically
        rng = int(RANGE_DEG); inc = int(INCREMENT_DEG)
        zero_th = int(ZERO_THETA_DEG); zero_ph = int(ZERO_PHI_DEG)
        direction = DIRECTION.lower()
        offsets = list(range(-rng, rng + 1, inc))
        base = "response"  # VituixCAD base name

        # Horizontal sweep → vary phi (Vituix plane = 'hor')
        if direction in ("horizontal", "hor_vert"):
            for off in offsets:
                th, ph, r = zero_th, zero_ph + off, default_r
                theta = math.radians(th); phi = math.radians(ph)
                x = r * math.sin(theta) * math.cos(phi) + off_x
                y = r * math.sin(theta) * math.sin(phi) + off_y
                z = r * math.cos(theta) + off_z
                prefix = f"{base}_hor{_format_signed_deg(off)}_{_radius_to_mm_str(r)}"
                targets.append((np.array([x, y, z]), prefix))

        # Vertical sweep → vary theta (Vituix plane = 'ver')
        if direction in ("vertical", "hor_vert"):
            for off in offsets:
                th, ph, r = zero_th + off, zero_ph, default_r
                theta = math.radians(th); phi = math.radians(ph)
                x = r * math.sin(theta) * math.cos(phi) + off_x
                y = r * math.sin(theta) * math.sin(phi) + off_y
                z = r * math.cos(theta) + off_z
                prefix = f"{base}_ver{_format_signed_deg(off)}_{_radius_to_mm_str(r)}"
                targets.append((np.array([x, y, z]), prefix))
    return targets

# -------------------------------------------------
# Evaluate pressure field at each target point
# -------------------------------------------------

def evaluate_field_at_points(
    freqs: np.ndarray,
    coeffs: np.ndarray,
    n_used: np.ndarray,
    points_xyz: np.ndarray,
    obs_mode: str,
) -> np.ndarray:
    """
    Compute complex pressure at given XYZ points from SHE coefficients.
    obs_mode controls which coefficients (internal/external/full) are used.
    """
    x, y, z = points_xyz.T                   # Split coordinates
    r = np.sqrt(x**2 + y**2 + z**2)          # Radial distance
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # Polar angle
    phi = np.arctan2(y, x)                   # Azimuthal angle
    P, K = r.size, freqs.size                # P points × K freqs
    pressures = np.zeros((K, P), dtype=np.complex128)

    for k_idx, f in enumerate(freqs):        # Loop over frequencies
        k_wave = 2 * math.pi * f / SPEED_OF_SOUND  # Wave number
        N_max = int(n_used[k_idx])           # Max harmonic order used
        idx_in = 0                           # Linear coeff index
        kr = k_wave * r                      # Dimensionless radius
        radial_h = [hankel1(n, kr) for n in range(N_max + 1)]  # Outgoing
        radial_j = [spherical_jn(n, kr) for n in range(N_max + 1)]  # Regular

        for n in range(N_max + 1):           # Loop over orders
            h_n, j_n = radial_h[n], radial_j[n]
            for m in range(-n, n + 1):       # Loop over harmonics
                C_nm = coeffs[k_idx, idx_in]     # Internal coeff
                D_nm = coeffs[k_idx, idx_in + 1] # External coeff
                Y_nm = sph_harm_y(n, m, theta, phi)  # Angular term
                # Combine per observation mode
                if obs_mode.lower() == "internal":
                    term = C_nm * h_n * Y_nm
                elif obs_mode.lower() == "external":
                    term = D_nm * j_n * Y_nm
                elif obs_mode.lower() in ("full", "full field"):
                    term = C_nm * h_n * Y_nm + D_nm * j_n * Y_nm
                else:
                    raise ValueError(f"Unknown OBSERVATION_MODE '{obs_mode}'")
                pressures[k_idx] += term     # Sum across m,n
                idx_in += 2                   # Advance coefficient index
    return pressures

# -------------------------------------------------
# Write helpers: FRD and complex spectra
# -------------------------------------------------

def write_frd(freqs: np.ndarray, pressures: np.ndarray, filepath: Path, desc: str):
    """Write frequency, magnitude(dB), and phase(deg) to a .frd text file."""
    eps = np.finfo(float).eps
    mags_db = 20 * np.log10(np.abs(pressures) + eps)
    phases_deg = np.angle(pressures, deg=True)
    header = [f"# FRD generated ({desc})", "# freq(Hz)    magnitude(dB)    phase(deg)"]
    data = np.column_stack([freqs, mags_db, phases_deg])
    np.savetxt(filepath, data, header="\n".join(header), fmt=("%.2f","%.5f","%.2f"))

def write_complex_npz(freqs: np.ndarray,
                      pressures: np.ndarray,
                      filepath: Path,
                      theta_deg: float | None = None,
                      phi_deg: float | None = None,
                      r_m: float | None = None):
    """Save complex pressure spectrum and optional spherical coords to .npz."""
    save_dict = {
        "freqs": freqs.astype(np.float64),
        "P": pressures.astype(np.complex128),
    }
    if theta_deg is not None: save_dict["theta_deg"] = float(theta_deg)
    if phi_deg is not None:   save_dict["phi_deg"] = float(phi_deg)
    if r_m is not None:       save_dict["r_m"] = float(r_m)
    np.savez_compressed(filepath, **save_dict)

def merge_pressures(freqs_lf, p_lf, freqs_hf, p_hf) -> Tuple[np.ndarray, np.ndarray]:
    """Blend LF and HF complex pressures smoothly over overlapping region."""
    f_overlap = np.intersect1d(freqs_lf, freqs_hf)          # Shared freqs
    f_lf_only = freqs_lf[~np.isin(freqs_lf, f_overlap)]     # LF-only region
    f_hf_only = freqs_hf[~np.isin(freqs_hf, f_overlap)]     # HF-only region
    idx_lf = {f: i for i, f in enumerate(freqs_lf)}         # Lookup tables
    idx_hf = {f: i for i, f in enumerate(freqs_hf)}
    p_lf_only = np.array([p_lf[idx_lf[f]] for f in f_lf_only])
    p_hf_only = np.array([p_hf[idx_hf[f]] for f in f_hf_only])
    p_ov_lf = np.array([p_lf[idx_lf[f]] for f in f_overlap])
    p_ov_hf = np.array([p_hf[idx_hf[f]] for f in f_overlap])

    # Blend overlapping magnitudes and phases linearly
    if f_overlap.size > 0:
        mag_lf, mag_hf = np.abs(p_ov_lf), np.abs(p_ov_hf)
        ph_lf, ph_hf = np.unwrap(np.angle(p_ov_lf)), np.unwrap(np.angle(p_ov_hf))
        p_lf_u = mag_lf * np.exp(1j * ph_lf)
        p_hf_u = mag_hf * np.exp(1j * ph_hf)
        N = f_overlap.size; w = np.linspace(0, 1, N)
        real_blend = (1 - w) * p_lf_u.real + w * p_hf_u.real
        imag_blend = (1 - w) * p_lf_u.imag + w * p_hf_u.imag
        p_overlap = real_blend + 1j * imag_blend
    else:
        p_overlap = np.array([], dtype=np.complex128)
    freqs_m = np.concatenate([f_lf_only, f_overlap, f_hf_only])
    p_m = np.concatenate([p_lf_only, p_overlap, p_hf_only])
    return freqs_m, p_m

# -------------------------------------------------
# TOF (time-of-flight) phase subtraction
# -------------------------------------------------

def apply_tof_phase_fixed(freqs, pressures, r_ref_m, c=SPEED_OF_SOUND):
    """Advance phase to remove delay corresponding to fixed reference distance."""
    if r_ref_m is None or r_ref_m <= 0:
        return pressures
    t_ref = float(r_ref_m) / float(c)
    ph = np.exp(-1j * 2.0 * np.pi * freqs * t_ref)
    return pressures * ph

# (Retained legacy helper – not normally used)
def apply_tof_phase_subtraction(freqs, pressures, eval_radius_m, pct, c=SPEED_OF_SOUND):
    """Remove a percentage of TOF phase relative to evaluated radius."""
    if pct <= 0:
        return pressures
    r_eff = (pct / 100.0) * float(eval_radius_m)
    t_eff = r_eff / float(c)
    ph = np.exp(-1j * 2.0 * np.pi * freqs * t_eff)
    return pressures * ph

# -------------------------------------------------
# Main routine
# -------------------------------------------------

def main():
    """Main execution entry – evaluate pressures and write FRDs/NPZs."""
    coeff_lf = Path(COEFF_LF_PATH)
    coeff_hf = Path(COEFF_HF_PATH)
    output_dir = Path(OUTPUT_DIR)
    obs_mode = OBSERVATION_MODE

    # Prepare directories
    output_dir.mkdir(parents=True, exist_ok=True)
    lf_dir = output_dir / "lf_response"
    hf_dir = output_dir / "hf_response"
    lf_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)
    merged_cx_dir = output_dir / "merged_complex"
    merged_cx_dir.mkdir(parents=True, exist_ok=True)

    # Report working mode to console
    if USE_COORD_LIST:
        print(f"List Mode: {len(COORD_LIST)} coordinate points")
    else:
        print(f"Sweep Mode: Direction={DIRECTION.upper()}, ±{RANGE_DEG}°, step={INCREMENT_DEG}°")
    print(f"Observation mode: {obs_mode}")

    # Load coefficient data
    freqs_lf, coeffs_lf, N_lf = load_coeff_file(coeff_lf)
    freqs_hf, coeffs_hf, N_hf = load_coeff_file(coeff_hf)

    targets = get_targets()                 # Build list of target points
    pts = np.array([t[0] for t in targets]) # Extract coordinates

    # Determine reference for TOF phase subtraction
    tof_ref_m: float | None = None
    if SUBTRACT_TOF:
        if USE_COORD_LIST:
            radii_all = [float(np.linalg.norm(p)) for p, _ in targets]
            tof_ref_m = float(min(radii_all)) if len(radii_all) > 0 else float(DIST_MIC)
            print(f"TOF subtraction ON (ref {tof_ref_m:.6f} m from coord list)")
        else:
            tof_ref_m = float(DIST_MIC)
            print(f"TOF subtraction ON (ref {tof_ref_m:.6f} m from DIST_MIC)")
    else:
        print("TOF subtraction: OFF")
#----
    p_lf_all = evaluate_field_at_points(freqs_lf, coeffs_lf, N_lf, pts, obs_mode)  # LF pressures at all targets
    p_hf_all = evaluate_field_at_points(freqs_hf, coeffs_hf, N_hf, pts, obs_mode)  # HF pressures at all targets

    for idx, (point_xyz, prefix) in enumerate(targets):      # Iterate targets
        x, y, z = point_xyz                                  # XYZ for this target
        r_eval = float(math.sqrt(x*x + y*y + z*z))           # Radius (m)
        theta_eval = float(math.degrees(math.acos(z / r_eval))) if r_eval != 0 else 0.0  # θ (deg)
        phi_eval = float(math.degrees(math.atan2(y, x)))     # φ (deg)

        p_lf_raw = p_lf_all[:, idx]                          # LF (raw, full TOF)
        p_hf_raw = p_hf_all[:, idx]                          # HF (raw, full TOF)

        # FRD exports: apply fixed-reference TOF if enabled
        if SUBTRACT_TOF and tof_ref_m is not None:
            p_lf_for_frd = apply_tof_phase_fixed(freqs_lf, p_lf_raw, tof_ref_m, SPEED_OF_SOUND)  # LF w/ TOF removed
            p_hf_for_frd = apply_tof_phase_fixed(freqs_hf, p_hf_raw, tof_ref_m, SPEED_OF_SOUND)  # HF w/ TOF removed
        else:
            p_lf_for_frd = p_lf_raw                           # Use raw LF
            p_hf_for_frd = p_hf_raw                           # Use raw HF

        # LF/HF band files keep suffixes
        write_frd(freqs_lf, p_lf_for_frd, lf_dir/f"{prefix}_LF.frd","LF")    # Save LF FRD
        write_frd(freqs_hf, p_hf_for_frd, hf_dir/f"{prefix}_HF.frd","HF")    # Save HF FRD

        # Merge for the VituixCAD-pattern FRD (no extra '_merged' in name)
        freqs_m_frd, p_m_frd = merge_pressures(freqs_lf, p_lf_for_frd, freqs_hf, p_hf_for_frd)
        write_frd(freqs_m_frd, p_m_frd, output_dir/f"{prefix}.frd","merged")              # Save merged FRD

        # NPZ always stores raw (with TOF) for later IFFT
        freqs_m_raw, p_m_raw = merge_pressures(freqs_lf, p_lf_raw, freqs_hf, p_hf_raw)           # Merge raw spectra
        write_complex_npz(                                                                         # Save NPZ
            freqs_m_raw,
            p_m_raw,
            merged_cx_dir / f"{prefix}_merged_complex.npz",
            theta_deg=theta_eval,
            phi_deg=phi_eval,
            r_m=r_eval,
        )

        print(f"Generated {prefix} FRDs + merged complex spectrum")  # Progress

    coords_file = output_dir / "coordinates.txt"               # Path for coordinate log
    with open(coords_file,"w") as f:                           # Write summary file
        f.write("# prefix    theta(deg)    phi(deg)    r(m)\n")  # Header
        for point,prefix in targets:                           # One line per target
            x,y,z = point; r=math.sqrt(x**2+y**2+z**2)
            theta=math.degrees(math.acos(z/r)) if r!=0 else 0.0
            phi=math.degrees(math.atan2(y,x))
            f.write(f"{prefix}    {theta:.2f}    {phi:.2f}    {r:.6f}\n")
    print("All FRD files, merged complex spectra, and coordinates.txt written to",output_dir)

if __name__=="__main__":
    main()
