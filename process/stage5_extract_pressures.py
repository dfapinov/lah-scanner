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
from pathlib import Path
import numpy as np
from utils import spherical_to_cartesian, cartesian_to_spherical

from extract_pressures_core import evaluate_she_field

# -------------------------------------------------
# CTA-2034 Helpers
# -------------------------------------------------

def generate_cta2034_coords(dist_m, zero_th, zero_ph):
    """
    Generates the 70 unique spherical coordinates required for CTA-2034-A.
    """
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

    def add_point(key_name, th_phys, ph_phys, zero_th_offset, zero_ph_offset):
        th_final = th_phys + (zero_th_offset - 90) 
        ph_final = ph_phys + zero_ph_offset
        
        r_out, th_rad, ph_rad = cartesian_to_spherical(*spherical_to_cartesian(1.0, np.radians(th_final), np.radians(ph_final)))
        
        k_t = round(np.degrees(th_rad), 2)
        k_p = round(np.degrees(ph_rad), 2)
        coord_key = (k_t, k_p)
        
        if coord_key not in coord_to_idx:
            coord_to_idx[coord_key] = len(unique_coords)
            unique_coords.append( (k_t, k_p, dist_m) )
            coord_deviations[len(unique_coords)-1] = th_phys
            
        map_indices[key_name] = coord_to_idx[coord_key]

    for a in angles:
        th, ph = get_phys_coords('H', a)
        add_point(f"H{a}", th, ph, zero_th, zero_ph)

    for a in angles:
        th, ph = get_phys_coords('V', a)
        add_point(f"V{a}", th, ph, zero_th, zero_ph)
        
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
    zero_theta=None, zero_phi=None, offset_xyz=None, c_sound=None, save_to_disk=True
):
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
        
    print("Observation Mode: Forced to 'Internal' (Anechoic)")
    obs_mode = "Internal"
    offsets = (float(offset_xyz[0]), float(offset_xyz[1]), float(offset_xyz[2]))
    
    print("Generating 70 CTA-2034 measurement angles...")
    coords_spherical, map_indices, coord_deviations = generate_cta2034_coords(
        eval_dist, float(zero_theta), float(zero_phi)
    )
    
    result_raw = evaluate_she_field(
        coords_sph=coords_spherical,
        she_input=coeff_path,
        obs_mode=obs_mode,
        offsets_xyz=offsets,
        subtract_tof=False,
        c_sound=c_sound
    )
    
    freqs = result_raw["freqs"]
    p_raw_all = result_raw["complex"] 
    
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
        
    return {"freqs": freqs, "metrics": metrics}

# -------------------------------------------------
# Standard Sweep Driver Function
# -------------------------------------------------
def run_sweep_extraction(
    coeff_path=None, output_dir=None, use_coord_list=None, coord_list=None,
    direction=None, range_deg=None, increment_deg=None, zero_theta=None,
    zero_phi=None, dist_mic=None, obs_mode=None, offset_xyz=None,
    subtract_tof=None, frd_prefix=None, c_sound=None, save_to_disk=True
):
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

    complex_dir = output_dir / "complex"
    if save_to_disk:
        output_dir.mkdir(parents=True, exist_ok=True)
        complex_dir.mkdir(parents=True, exist_ok=True)

    coords_spherical = []
    prefixes = []
    default_r = float(dist_mic)
    offsets = (float(offset_xyz[0]), float(offset_xyz[1]), float(offset_xyz[2]))

    if use_coord_list:
        print(f"List Mode: {len(coord_list)} coordinate points")
    else:
        print(f"Sweep Mode: Direction={direction.upper()}, ±{range_deg}°, step={increment_deg}°")
    print(f"Observation mode: {obs_mode}")

    if use_coord_list:
        for entry in coord_list:
            if len(entry) == 2:
                th, ph = entry; r = default_r
            else:
                th, ph, r = entry
            coords_spherical.append((th, ph, r))
            r_str = f"{int(round(r*1000))}mm"
            prefixes.append(f"th{th}_ph{ph}_{r_str}")
    else:
        rng = int(range_deg); inc = int(increment_deg)
        zero_th = int(zero_theta); zero_ph = int(zero_phi)
        off_range = list(range(-rng, rng + 1, inc))
        if direction.lower() in ("horizontal", "hor_vert"):
            for off in off_range:
                coords_spherical.append((zero_th, zero_ph + off, default_r))
                val_str = f"+{off}" if off > 0 else f"{off}"
                prefixes.append(f"{frd_prefix}_hor{val_str}_{int(default_r*1000)}mm")
        if direction.lower() in ("vertical", "hor_vert"):
            for off in off_range:
                coords_spherical.append((zero_th + off, zero_ph, default_r))
                val_str = f"+{off}" if off > 0 else f"{off}"
                prefixes.append(f"{frd_prefix}_ver{val_str}_{int(default_r*1000)}mm")

    tof_ref_dist = None
    if subtract_tof:
        pts = np.array(coords_spherical, dtype=float)
        th_rad = np.radians(pts[:, 0])
        ph_rad = np.radians(pts[:, 1])
        r_val  = pts[:, 2]
        x_t, y_t, z_t = spherical_to_cartesian(r_val, th_rad, ph_rad)
        x_t += offsets[0]
        y_t += offsets[1]
        z_t += offsets[2]
        r_acous = np.sqrt(x_t**2 + y_t**2 + z_t**2)
        tof_ref_dist = np.min(r_acous)
        print(f"TOF subtraction ON (ref {tof_ref_dist:.6f} m)")
    else:
        print("TOF subtraction: OFF")

    result_raw = evaluate_she_field(
        coords_sph=coords_spherical,
        she_input=coeff_path,
        obs_mode=obs_mode,
        offsets_xyz=offsets,
        subtract_tof=False, 
        c_sound=c_sound
    )
    
    freqs = result_raw["freqs"]
    p_raw_all = result_raw["complex"] 

    if save_to_disk:
        print("Writing files...")
    coords_log = []
    extracted_data = {}
    
    for idx, prefix in enumerate(prefixes):
        p_raw = p_raw_all[:, idx]
        th_in, ph_in, r_in = coords_spherical[idx]
        coords_log.append(f"{prefix}    {th_in:.2f}    {ph_in:.2f}    {r_in:.6f}")

        p_frd = p_raw.copy()
        if subtract_tof and tof_ref_dist is not None:
            t_delay = tof_ref_dist / c_sound
            p_frd *= np.exp(-1j * 2.0 * np.pi * freqs * t_delay)

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
            write_complex_npz(
                freqs, p_raw, 
                complex_dir / f"{prefix}_complex.npz",
                theta_in=th_in, phi_in=ph_in, r_in=r_in
            )
            write_frd(freqs, mags, phases, output_dir / f"{prefix}.frd", obs_mode)
        
    if save_to_disk:
        with open(output_dir / "coordinates.txt", "w") as f:
            f.write("# prefix    theta(deg)    phi(deg)    r(m)\n")
            f.write("\n".join(coords_log))
            
        print(f"Success. Files written to {output_dir}")
        
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