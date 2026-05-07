#!/usr/bin/env python3
"""
Stage 3 – Extract FRD files from SHE coefficients and apply sound-field separation
=================================================================================
Updated to support CTA-2034-A (Spinorama) Metrics Generation and 
Dynamic Per-Frequency Inverse Coordinate Translation.

CTA-2034-A Compliance Updates (Reviewer Feedback Implemented):
--------------------------------------------------------------
1. Geometry: Generates 70 unique points (Horizontal + Vertical orbits).
2. Sound Power: Uses 'Zonal Density Weighting'.
3. Averaging Domains: Energy (Sound Power) and SPL (LW & ER).
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path
import numpy as np
from utils import spherical_to_cartesian, cartesian_to_spherical, apply_mic_calibration, write_wav

from extract_pressures_core import evaluate_she_field
from complex_to_ir_core import complex_to_ir

# -------------------------------------------------
# Shared Helpers
# -------------------------------------------------

def get_tof_phasor(freqs, dist_m, c_sound):
    """Calculates the complex phase rotation array needed to subtract time-of-flight."""
    return np.exp(1j * 2.0 * np.pi * freqs * (dist_m / c_sound))

# -------------------------------------------------
# CTA-2034 Helpers
# -------------------------------------------------

def generate_cta2034_coords(dist_m, zero_th, zero_ph):
    """
    Generates the 70 unique spherical coordinates required for CTA-2034-A.
    """
    # Setup rotation matrix from the zero-axis angles
    th_rad_zero = np.radians(float(zero_th))
    ph_rad_zero = np.radians(float(zero_ph))
    F = np.array([np.sin(th_rad_zero) * np.cos(ph_rad_zero), np.sin(th_rad_zero) * np.sin(ph_rad_zero), np.cos(th_rad_zero)])
    R = np.array([-np.sin(ph_rad_zero), np.cos(ph_rad_zero), 0])
    U = np.cross(F, R)
    rot_matrix = np.array([F, R, U]).T

    angles = list(range(0, 360, 10))
    unique_coords = []
    coord_to_idx = {} 
    map_indices = {} 
    coord_deviations = {} 

    def get_phys_coords(orbit, angle_deg):
        if orbit == 'H':
            th_p = 90.0
            ph_p = float(angle_deg)
        elif orbit == 'V':
            if 0 <= angle_deg <= 180:
                th_p = 90.0 - angle_deg
                ph_p = 0.0
                if th_p < 0:
                    th_p = abs(th_p)
                    ph_p = 180.0
            else:
                rem = angle_deg - 180
                th_p = 90.0 + rem
                ph_p = 180.0
                if th_p > 180:
                    th_p = 360 - th_p
                    ph_p = 0.0
        return th_p, ph_p

    def add_point(key_name, th_phys, ph_phys):
        # Convert local spherical coordinates (standard acoustic orientation) to local Cartesian
        th_phys_rad = np.radians(th_phys)
        ph_phys_rad = np.radians(ph_phys)
        x_local = dist_m * np.sin(th_phys_rad) * np.cos(ph_phys_rad)
        y_local = dist_m * np.sin(th_phys_rad) * np.sin(ph_phys_rad)
        z_local = dist_m * np.cos(th_phys_rad)
        p_local = np.array([x_local, y_local, z_local])

        # Rotate the local Cartesian point to the global frame
        p_global = rot_matrix @ p_local
        r_out, th_out_rad, ph_out_rad = cartesian_to_spherical(p_global[0], p_global[1], p_global[2])
        k_t = round(np.degrees(th_out_rad), 2)
        k_p = round(np.degrees(ph_out_rad), 2)
        coord_key = (k_t, k_p)
        
        if coord_key not in coord_to_idx:
            coord_to_idx[coord_key] = len(unique_coords)
            unique_coords.append( (k_t, k_p, dist_m) )
            coord_deviations[len(unique_coords)-1] = th_phys
            
        map_indices[key_name] = coord_to_idx[coord_key]

    for a in angles:
        th, ph = get_phys_coords('H', a)
        add_point(f"H{a}", th, ph)

    for a in angles:
        th, ph = get_phys_coords('V', a)
        add_point(f"V{a}", th, ph)
        
    return unique_coords, map_indices, coord_deviations

def calculate_cta2034_metrics(freqs, p_matrix, map_idx, coord_deviations):
    w_table = {
        0: 0.0006, 180: 0.0006, 10: 0.0047, 170: 0.0047,
        20: 0.0093, 160: 0.0093, 30: 0.0136, 150: 0.0136,
        40: 0.0175, 140: 0.0175, 50: 0.0208, 130: 0.0208,
        60: 0.0233, 120: 0.0233, 70: 0.0251, 110: 0.0251,
        80: 0.0261, 100: 0.0261, 90: 0.0264
    }

    def p_to_db(p_complex):
        eps = np.finfo(float).eps
        return 20 * np.log10(np.abs(p_complex) + eps)

    def spl_avg(indices):
        subset = p_matrix[:, indices]
        dbs = p_to_db(subset)
        return np.mean(dbs, axis=1)

    def e_to_db(energy):
        eps = np.finfo(float).eps
        return 10 * np.log10(energy + eps)

    idx_on_axis = [map_idx['H0']]
    
    lw_keys = ['H0', 'H10', 'H350', 'H20', 'H340', 'H30', 'H330', 'V10', 'V350']
    idx_lw = [map_idx[k] for k in lw_keys]
    
    keys_floor = ['V340', 'V330', 'V320']
    keys_ceil = ['V40', 'V50', 'V60']    
    keys_front = ['H0', 'H10', 'H350', 'H20', 'H340', 'H30', 'H330']
    
    keys_side = []
    for a in range(40, 90, 10): 
        keys_side.append(f"H{a}")
        keys_side.append(f"H{360-a}")
        
    keys_rear = ['H180']
    for a in range(90, 180, 10): 
        keys_rear.append(f"H{a}")
        keys_rear.append(f"H{360-a}")
        
    idx_floor = [map_idx[k] for k in keys_floor]
    idx_ceil = [map_idx[k] for k in keys_ceil]
    idx_front = [map_idx[k] for k in keys_front]
    idx_side = [map_idx[k] for k in keys_side]
    idx_rear = [map_idx[k] for k in keys_rear]
    
    idx_er_total = list(set(idx_floor + idx_ceil + idx_front + idx_side + idx_rear))
    
    db_on_axis = p_to_db(p_matrix[:, idx_on_axis[0]])
    db_lw = spl_avg(idx_lw)         
    
    db_floor = spl_avg(idx_floor)
    db_ceil = spl_avg(idx_ceil)
    db_front = spl_avg(idx_front)
    db_side = spl_avg(idx_side)
    db_rear = spl_avg(idx_rear)
    
    db_er = spl_avg(idx_er_total)   
    
    band_counts = {}
    for i in range(len(coord_deviations)):
        dev = coord_deviations[i]
        dev_r = int(round(dev / 10.0)) * 10
        if dev_r > 180: dev_r = 180 
        if dev_r < 0: dev_r = 0
        band_counts[dev_r] = band_counts.get(dev_r, 0) + 1
        
    e_sp_sum = np.zeros_like(db_on_axis, dtype=float)
    
    for i in range(p_matrix.shape[1]):
        dev = coord_deviations[i]
        dev_r = int(round(dev / 10.0)) * 10
        
        if dev_r in w_table and dev_r in band_counts:
            w_point = w_table[dev_r] / band_counts[dev_r]
            energy = np.abs(p_matrix[:, i])**2
            e_sp_sum += energy * w_point
            
    db_sp = e_to_db(e_sp_sum) 

    e_final_lw = 10**(db_lw/10)
    e_final_er = 10**(db_er/10)
    e_final_sp = 10**(db_sp/10)
    
    e_pir = 0.44 * e_final_lw + 0.44 * e_final_er + 0.12 * e_final_sp
    db_pir = e_to_db(e_pir)
    
    db_spdi = db_lw - db_sp
    db_erdi = db_lw - db_er
    
    ph_ref = np.angle(p_matrix[:, idx_on_axis[0]], deg=True)
    
    results = {
        "Response_OnAxis": (db_on_axis, ph_ref),
        "Response_ListeningWindow": (db_lw, ph_ref),
        "Response_EarlyReflections": (db_er, ph_ref),
        "Response_SoundPower": (db_sp, ph_ref),
        "Response_PIR": (db_pir, ph_ref),
        "Response_ERDI": (db_erdi, np.zeros_like(ph_ref)),
        "Response_SPDI": (db_spdi, np.zeros_like(ph_ref)),
        
        "reflections_breakout/Response_ER_Floor": (db_floor, ph_ref),
        "reflections_breakout/Response_ER_Ceiling": (db_ceil, ph_ref),
        "reflections_breakout/Response_ER_FrontWall": (db_front, ph_ref),
        "reflections_breakout/Response_ER_SideWalls": (db_side, ph_ref),
        "reflections_breakout/Response_ER_RearWall": (db_rear, ph_ref),
    }
    return results

# -------------------------------------------------
# Writers
# -------------------------------------------------

def write_frd(freqs: np.ndarray, mags: np.ndarray, phases: np.ndarray, filepath: Path, desc: str):
    header = [f"# FRD generated ({desc})", "# freq(Hz)    magnitude(dB)    phase(deg)"]
    data = np.column_stack([freqs, mags, phases])
    np.savetxt(filepath, data, header="\n".join(header), fmt=("%.2f","%.5f","%.2f"))

def write_complex_npz(freqs: np.ndarray, pressures: np.ndarray, filepath: Path, **meta):
    save_dict = {
        "freqs": freqs.astype(np.float64),
        "P": pressures.astype(np.complex128),
    }
    save_dict.update(meta)
    np.savez_compressed(filepath, **save_dict)

# -------------------------------------------------
# CTA-2034 Driver Function
# -------------------------------------------------
def run_cta2034_extraction(
    coeff_path=None, output_dir=None, dist_mic=None,
    zero_theta=None, zero_phi=None, offset_xyz=None, c_sound=None, save_to_disk=True,
    apply_mic_cal=None, mic_cal_file=None, mic_cal_mode=None, 
    obs_mode=None, mic_cal_fade_octaves=None, use_optimized_origins=True,
    subtract_tof=None
):
    start_time = time.time()

    import config_process
    coeff_path = coeff_path if coeff_path is not None else config_process.COEFF_PATH
    output_dir = Path(output_dir if output_dir is not None else config_process.OUTPUT_DIR)
    dist_mic = dist_mic if dist_mic is not None else config_process.DIST_MIC
    zero_theta = zero_theta if zero_theta is not None else config_process.ZERO_THETA_DEG
    zero_phi = zero_phi if zero_phi is not None else config_process.ZERO_PHI_DEG
    offset_xyz = offset_xyz if offset_xyz is not None else (config_process.OFFSET_MIC_X, config_process.OFFSET_MIC_Y, config_process.OFFSET_MIC_Z)
    c_sound = c_sound if c_sound is not None else config_process.SPEED_OF_SOUND
    
    print("\n" + "="*50)
    print(" MODE: CTA-2034-A (SPINORAMA)")
    print("="*50)
    
    cta_dir = output_dir / "CTA2034"
    breakout_dir = cta_dir / "reflections_breakout"
    if save_to_disk:
        cta_dir.mkdir(parents=True, exist_ok=True)
        breakout_dir.mkdir(parents=True, exist_ok=True)
    
    eval_dist = 2.0 
    if dist_mic != 2.0:
        print(f"Note: Using configured distance {dist_mic}m (Standard is 2.0m)")
        eval_dist = float(dist_mic)
    else:
        print("Note: Using standard reference distance 2.0m")
        
    obs_mode_val = obs_mode if obs_mode else "Internal"
    print(f"Observation Mode: {obs_mode_val}")
    
    print("Generating 70 CTA-2034 measurement angles...")
    coords_spherical, map_indices, coord_deviations = generate_cta2034_coords(
        eval_dist, float(zero_theta), float(zero_phi)
    )
    
    # Apply static offsets here in stage5
    pts_sph = np.array(coords_spherical, dtype=float)
    r_in, th_in_rad, ph_in_rad = pts_sph[:, 2], np.radians(pts_sph[:, 0]), np.radians(pts_sph[:, 1])
    x_b, y_b, z_b = spherical_to_cartesian(r_in, th_in_rad, ph_in_rad)
    off_x, off_y, off_z = offset_xyz
    x_b += off_x; y_b += off_y; z_b += off_z
    r_final, th_final_rad, ph_final_rad = cartesian_to_spherical(x_b, y_b, z_b)
    coords_spherical_final = np.column_stack((np.degrees(th_final_rad), np.degrees(ph_final_rad), r_final)).tolist()

    result_raw = evaluate_she_field(
        coords_sph=coords_spherical_final,
        she_input=coeff_path,
        obs_mode=obs_mode_val,
        c_sound=c_sound,
        use_optimized_origins=use_optimized_origins
    )
    
    freqs = result_raw["freqs"]
    p_raw_all = result_raw["complex"] 
    
    apply_cal_val = apply_mic_cal if apply_mic_cal is not None else getattr(config_process, 'APPLY_MIC_CALIBRATION', False)
    if apply_cal_val:
        cal_file = mic_cal_file if mic_cal_file else getattr(config_process, 'MIC_CALIBRATION_FILE', 'MM1_Mic_Cal.txt')
        cal_mode = mic_cal_mode if mic_cal_mode is not None else getattr(config_process, 'MIC_CALIBRATION_MODE', 'subtract')
        fade_oct = float(mic_cal_fade_octaves) if mic_cal_fade_octaves is not None else getattr(config_process, 'MIC_CALIBRATION_FADE_OCTAVES', 1.0)
        print(f"Applying mic calibration: {cal_file} (Mode: {cal_mode}, Fade: {fade_oct} oct)")
        p_raw_all = apply_mic_calibration(p_raw_all, freqs, cal_file, cal_mode, fade_oct)

    subtract_tof_val = subtract_tof if subtract_tof is not None else getattr(config_process, 'SUBTRACT_TOF', False)
    if subtract_tof_val:
        # NOTE: Using `r_in` (the radius before cartesian mic offsets are applied) 
        # ensures the TOF reference point shifts *with* the mic offset. For example, 
        # if the mic offset aligns with a tweeter, the TOF is calculated back to that 
        # offset point. Previously, this used `r_final` which always calculated TOF 
        # back to the absolute 0,0,0 origin regardless of offset.
        tof_ref_dist = np.min(r_in)
        print(f"TOF subtraction ON (ref {tof_ref_dist:.6f} m)")
        p_raw_all *= get_tof_phasor(freqs, tof_ref_dist, c_sound)[:, np.newaxis]
    else:
        print("TOF subtraction: OFF")

    print("Computing Spinorama metrics...")
    metrics = calculate_cta2034_metrics(freqs, p_raw_all, map_indices, coord_deviations)
    
    if save_to_disk:
        print(f"Writing files to {cta_dir}...")
        for name, (mag, phase) in metrics.items():
            fpath = cta_dir / f"{name}.frd"
            if "/" in name:
                fpath = cta_dir / f"{name.split('/')[0]}" / f"{name.split('/')[1]}.frd"
            write_frd(freqs, mag, phase, fpath, "CTA-2034")
            
        print("Success. CTA-2034 generation complete.")
        
    elapsed = time.time() - start_time
    print(f"\nStage 5 processing completed in {elapsed:.2f} seconds.")

    return {"freqs": freqs, "metrics": metrics}

# -------------------------------------------------
# Standard Sweep Driver Function
# -------------------------------------------------
def run_sweep_extraction(
    coeff_path=None, output_dir=None, use_coord_list=None, coord_list=None,
    direction=None, range_deg=None, increment_deg=None, zero_theta=None,
    zero_phi=None, dist_mic=None, obs_mode=None, offset_xyz=None, subtract_tof=None,
    frd_prefix=None, c_sound=None, save_to_disk=True, generate_ir_files=None,
    apply_mic_cal=None, mic_cal_file=None, mic_cal_mode=None,
    mic_cal_fade_octaves=None, use_optimized_origins=True
):
    start_time = time.time()

    import config_process
    coeff_path = coeff_path if coeff_path is not None else config_process.COEFF_PATH
    output_dir = Path(output_dir if output_dir is not None else config_process.OUTPUT_DIR)
    use_coord_list = use_coord_list if use_coord_list is not None else config_process.USE_COORD_LIST
    coord_list = coord_list if coord_list is not None else config_process.COORD_LIST
    direction = direction if direction is not None else config_process.DIRECTION
    range_deg = range_deg if range_deg is not None else config_process.RANGE_DEG
    increment_deg = increment_deg if increment_deg is not None else config_process.INCREMENT_DEG
    zero_theta = zero_theta if zero_theta is not None else config_process.ZERO_THETA_DEG
    zero_phi = zero_phi if zero_phi is not None else config_process.ZERO_PHI_DEG
    dist_mic = dist_mic if dist_mic is not None else config_process.DIST_MIC
    obs_mode = obs_mode if obs_mode is not None else config_process.OBSERVATION_MODE
    offset_xyz = offset_xyz if offset_xyz is not None else (config_process.OFFSET_MIC_X, config_process.OFFSET_MIC_Y, config_process.OFFSET_MIC_Z)
    subtract_tof = subtract_tof if subtract_tof is not None else config_process.SUBTRACT_TOF
    frd_prefix = frd_prefix if frd_prefix is not None else config_process.FRD_PREFIX
    c_sound = c_sound if c_sound is not None else config_process.SPEED_OF_SOUND
    gen_ir = generate_ir_files if generate_ir_files is not None else getattr(config_process, 'GENERATE_IR_FILES', False)

    output_dir = output_dir / frd_prefix
    complex_dir = output_dir / "complex"
    ir_dir = output_dir / "ir"
    if save_to_disk:
        output_dir.mkdir(parents=True, exist_ok=True)
        complex_dir.mkdir(parents=True, exist_ok=True)
        if gen_ir:
            ir_dir.mkdir(parents=True, exist_ok=True)

    coords_spherical = []
    prefixes = []
    default_r = float(dist_mic)

    if not use_coord_list:
        print(f"Sweep Mode: Direction={direction.upper()}, ±{range_deg}°, step={increment_deg}°")
    else:
        print(f"List Mode: {len(coord_list)} coordinate points")
    print(f"Observation mode: {obs_mode}")

    if use_coord_list:
        for entry in coord_list:
            if len(entry) == 2:
                th, ph = entry; r = default_r
            else:
                th, ph, r = entry
            coords_spherical.append((th, ph, r))
            r_str = f"{int(round(r*1000))}mm"
            prefixes.append((f"{frd_prefix}_{r_str}_th{th}_ph{ph}", ""))
    else:
        rng = int(range_deg); inc = int(increment_deg)
        
        th_rad = np.radians(float(zero_theta))
        ph_rad = np.radians(float(zero_phi))
        # Define the local coordinate system basis vectors robustly to avoid gimbal lock
        # Forward vector (local X')
        F = np.array([np.sin(th_rad) * np.cos(ph_rad), np.sin(th_rad) * np.sin(ph_rad), np.cos(th_rad)])
        
        # Right vector (local Y')
        R = np.array([-np.sin(ph_rad), np.cos(ph_rad), 0])

        # Up vector (local Z'), derived from the other two to ensure a right-handed system
        U = np.cross(F, R)
        rot_matrix = np.array([F, R, U]).T

        off_range = list(range(-rng, rng + 1, inc))
        for ang_deg in off_range:
            ang_rad = np.radians(ang_deg)
            val_str = f"+{ang_deg}" if ang_deg >= 0 else f"{ang_deg}"
            
            if direction.lower() in ("horizontal", "hor_vert"):
                p_local = np.array([default_r * np.cos(ang_rad), default_r * np.sin(ang_rad), 0])
                p_global = rot_matrix @ p_local
                r_s, th_s, ph_s = cartesian_to_spherical(p_global[0], p_global[1], p_global[2])
                coords_spherical.append((np.degrees(th_s), np.degrees(ph_s), r_s))
                prefixes.append((f"{frd_prefix}_{int(default_r*1000)}mm_hor{val_str}", ""))
            if direction.lower() in ("vertical", "hor_vert"):
                if direction.lower() == "hor_vert" and ang_deg == 0: continue
                p_local = np.array([default_r * np.cos(ang_rad), 0, default_r * np.sin(ang_rad)])
                p_global = rot_matrix @ p_local
                r_s, th_s, ph_s = cartesian_to_spherical(p_global[0], p_global[1], p_global[2])
                coords_spherical.append((np.degrees(th_s), np.degrees(ph_s), r_s))
                prefixes.append((f"{frd_prefix}_{int(default_r*1000)}mm_ver{val_str}", ""))

    # Apply static offsets here in stage5, unless in manual mode where coords are absolute
    pts_sph = np.array(coords_spherical, dtype=float)
    r_in, th_in_rad, ph_in_rad = pts_sph[:, 2], np.radians(pts_sph[:, 0]), np.radians(pts_sph[:, 1])
    x_b, y_b, z_b = spherical_to_cartesian(r_in, th_in_rad, ph_in_rad)

    if not use_coord_list:
        off_x, off_y, off_z = offset_xyz
        x_b += off_x; y_b += off_y; z_b += off_z

    r_final, th_final_rad, ph_final_rad = cartesian_to_spherical(x_b, y_b, z_b)
    coords_spherical_final = np.column_stack((np.degrees(th_final_rad), np.degrees(ph_final_rad), r_final)).tolist()

    tof_ref_dist = None
    if subtract_tof:
        # NOTE: Using `r_in` (the radius before cartesian mic offsets are applied) 
        # ensures the TOF reference point shifts *with* the mic offset. For example, 
        # if the mic offset aligns with a tweeter, the TOF is calculated back to that 
        # offset point. Previously, this used `r_final` which always calculated TOF 
        # back to the absolute 0,0,0 origin regardless of offset.
        r_acous = r_in
        tof_ref_dist = np.min(r_acous)
        print(f"TOF subtraction ON (ref {tof_ref_dist:.6f} m)")
    else:
        print("TOF subtraction: OFF")

    result_raw = evaluate_she_field(
        coords_sph=coords_spherical_final,
        she_input=coeff_path,
        obs_mode=obs_mode,
        c_sound=c_sound,
        use_optimized_origins=use_optimized_origins
    )
    
    freqs = result_raw["freqs"]
    p_raw_all = result_raw["complex"] 
    fs_val = result_raw.get("fs")

    apply_cal_val = apply_mic_cal if apply_mic_cal is not None else getattr(config_process, 'APPLY_MIC_CALIBRATION', False)
    if apply_cal_val:
        cal_file = mic_cal_file if mic_cal_file else getattr(config_process, 'MIC_CALIBRATION_FILE', 'MM1_Mic_Cal.txt')
        cal_mode = mic_cal_mode if mic_cal_mode is not None else getattr(config_process, 'MIC_CALIBRATION_MODE', 'subtract')
        fade_oct = float(mic_cal_fade_octaves) if mic_cal_fade_octaves is not None else getattr(config_process, 'MIC_CALIBRATION_FADE_OCTAVES', 1.0)
        print(f"Applying mic calibration: {cal_file} (Mode: {cal_mode}, Fade: {fade_oct} oct)")
        p_raw_all = apply_mic_calibration(p_raw_all, freqs, cal_file, cal_mode, fade_oct)

    if subtract_tof and tof_ref_dist is not None:
        tof_phasor = get_tof_phasor(freqs, tof_ref_dist, c_sound)
    else:
        tof_phasor = None

    if save_to_disk:
        print("Writing files...")
    coords_log = []
    extracted_data = {}
    
    # SHE data is capped at 24kHz, so IRs are strictly generated at standard rates
    target_fs = 44100 if freqs[-1] < 23000.0 else 48000
    
    for idx, (prefix, subdir) in enumerate(prefixes):
        p_raw = p_raw_all[:, idx]
        th_in, ph_in, r_in = coords_spherical_final[idx]
        coords_log.append(f"{prefix}    {th_in:.2f}    {ph_in:.2f}    {r_in:.6f}")

        p_frd = p_raw.copy()
        if tof_phasor is not None:
            p_frd *= tof_phasor

        # --- IR Generation ---
        if save_to_disk and gen_ir:
            ir_out_dir = ir_dir / subdir if subdir else ir_dir
            
            ir_audio = complex_to_ir(
                p_complex=p_raw,
                freqs=freqs,
                target_fs=target_fs
            )
            
            write_wav(
                ir_audio,
                target_fs,
                ir_out_dir / f"{prefix}.wav",
                "Impulse Response"
            )

        eps = np.finfo(float).eps
        mags = 20 * np.log10(np.abs(p_frd) + eps)
        phases = np.angle(p_frd, deg=True)
        
        extracted_data[prefix] = {
            "complex": p_raw,
            "mag": mags,
            "phase": phases,
            "theta": th_in,
            "phi": ph_in,
            "r": r_in
        }

        if save_to_disk:
            out_subdir = output_dir / subdir if subdir else output_dir
            comp_subdir = complex_dir / subdir if subdir else complex_dir
            
            write_complex_npz(
                freqs, p_raw, 
                comp_subdir / f"{prefix}_complex.npz",
                theta_in=th_in, phi_in=ph_in, r_in=r_in
            )
            write_frd(freqs, mags, phases, out_subdir / f"{prefix}.frd", obs_mode)
        
    if save_to_disk:
        with open(output_dir / "coordinates.txt", "w") as f:
            f.write("# prefix    theta(deg)    phi(deg)    r(m)\n")
            f.write("\n".join(coords_log))
            
        print(f"Success. Files written to {output_dir}")
        
    elapsed = time.time() - start_time
    print(f"\nStage 5 processing completed in {elapsed:.2f} seconds.")

    return {"freqs": freqs, "data": extracted_data}

# -------------------------------------------------


# -------------------------------------------------
# Main CLI Execution
# -------------------------------------------------
def main():
    try:
        import config_process
        
        if getattr(config_process, 'CTA_MODE', False):
            run_cta2034_extraction()
        else:
            run_sweep_extraction()
    except ImportError:
        print("Error: config_process.py not found.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()