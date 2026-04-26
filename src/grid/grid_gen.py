#!/usr/bin/env python3
"""
Cylindrical Measurement Grid Generator with Volumetric Variable Density
==========================================================================
Uses Deterministic Sine-Hashing for charge distribution and reverts to 
strict Cylindrical Expansion (XY walls, Z caps) to protect gradients.

Note: This one
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_cylinder_spiral(N, R, H, cap_frac,
                             reverse=False, rotate_deg=0.0, flip_z=False,
                             wall_thickness_mm=0.0,
                             vd_power_side=0.5, vd_power_caps=0.5,
                             index_offset=0, bottom_cutoff=0.0,
                                 azimuth_density_ratio=1.0,
                             azimuth_weight_center_deg=0.0):
    """Generate one Fibonacci spiral with Hashed Cylindrical Variable Density."""
    phi = (-1 if reverse else 1) * np.pi * (3. - np.sqrt(5))
    n_side = int(round(N*(1-cap_frac)))
    n_caps = N - n_side
    n_top = n_caps//2 + (n_caps%2)
    n_bot = n_caps - n_top
 
    z_shift_m = wall_thickness_mm / 1000.0
    pts = []
 
    # Convert center to radians, accounting for any subsequent rotation
    # so the concentrated area ends up at the requested global angle
    azimuth_weight_center_rad = np.radians(azimuth_weight_center_deg - rotate_deg)

    # Calculate the conformal mapping coefficient 'alpha'
    # Ratio = 1 / alpha^2  =>  alpha = 1 / sqrt(Ratio)
    alpha = 1.0 / np.sqrt(max(1.0, azimuth_density_ratio))

    for i in range(n_side):
        t = i/max(n_side-1,1)
        z = -H/2 + H * t
        
        # Start with a uniform spiral angle
        u = (i * phi) % (2 * np.pi)

        # Warp using a conformal mapping for a perfectly smooth, continuous density gradient
        u_shifted = u - azimuth_weight_center_rad
        theta_half = np.arctan2(alpha * np.sin(u_shifted / 2.0), np.cos(u_shifted / 2.0))
        θ = (2.0 * theta_half + azimuth_weight_center_rad) % (2 * np.pi)

        pts.append([R*np.cos(θ), R*np.sin(θ), z])

    for j in range(n_top):
        r = R*np.sqrt((j+0.5)/n_top)
        θ = (j*phi)%(2*np.pi)
        pts.append([r*np.cos(θ), r*np.sin(θ), +H/2 - z_shift_m])

    for j in range(n_bot):
        r = R*np.sqrt((j+0.5)/n_bot)
        θ = (j*phi)%(2*np.pi)
        pts.append([r*np.cos(θ), r*np.sin(θ), -H/2 + z_shift_m])

    arr = np.array(pts)

    # --- Apply Variable Density (Deterministic Hash + Cylindrical Expansion) ---
    if wall_thickness_mm > 0:
        d_max = wall_thickness_mm / 1000.0
        
        # High-frequency constants for the deterministic hash
        HASH_A = 12.9898
        HASH_B = 43758.5453
        
        for k in range(arr.shape[0]):
            x, y, z = arr[k]
            global_k = k + index_offset
            
            # 1. The Deterministic Sine-Hash (breaks the screw thread)
            u_k = (np.abs(np.sin(global_k * HASH_A) * HASH_B)) % 1.0
            
            # 2. Strict Cylindrical Expansion & Index-Based Power Law
            is_cap = (k >= n_side)

            if is_cap:
                # Point is on an end cap -> Push strictly vertically (Z) using P_caps
                dr = d_max * (u_k ** vd_power_caps)
                if k < n_side + n_top:
                    # Top cap expands upwards
                    arr[k, 2] += dr
                else:
                    # Bottom cap expands downwards
                    arr[k, 2] -= dr
            else:
                # Point is on the side wall -> Push strictly radially (X, Y) using P_side
                dr = d_max * (u_k ** vd_power_side)
                mag_xy = np.hypot(x, y)
                if mag_xy > 0:
                    arr[k, 0] += (x / mag_xy) * dr
                    arr[k, 1] += (y / mag_xy) * dr

    # --- Apply Transformations AFTER Density Expansion ---
    if flip_z:
        arr[:,2] *= -1

    if rotate_deg != 0:
        ang = np.radians(rotate_deg)
        c,s = np.cos(ang), np.sin(ang)
        x2 = c*arr[:,0] - s*arr[:,1]
        y2 = s*arr[:,0] + c*arr[:,1]
        arr[:,0], arr[:,1] = x2, y2


    df = pd.DataFrame(arr, columns=['x','y','z'])

    tol = (wall_thickness_mm/1000.0) if wall_thickness_mm>0 else 0
    # Update bottom cutoff mask to account for the new cap_z_shift
    mask_bottom = (
        (np.abs(df['z'] - (-H/2 + z_shift_m)) <= tol + 1e-8) &
        (np.hypot(df['x'], df['y']) <= bottom_cutoff)
    )
    if mask_bottom.any():
        df = df[~mask_bottom].reset_index(drop=True)

    return df

def generate_measurement_grid(
    cyl_radius,
    cyl_height,
    num_points,
    wall_thickness_mm,
    bottom_cutoff_mm,
    cap_fraction=None,
    P_side=0.5,
    P_caps=0.5,
    generate_reverse_spiral=True,
    z_rotation_deg=90.0,
    flip_poles=False,
    z_midpoint_zero=True,
    phi_min_deg=-180.0,
    phi_max_deg=180.0,
    azimuth_density_ratio=1.0,
    azimuth_weight_center_deg=0.0
):
    bottom_cutoff = bottom_cutoff_mm / 1000.0

    # Adjust cyl_height so the requested value represents the internal clearance.
    # Adding 2x the variable density max (in meters) means the expanded caps 
    # will end exactly at the new outer cyl_height boundary.
    cyl_height_working = cyl_height + 2.0 * (wall_thickness_mm / 1000.0)

    # ============================================================
    # Automatic / manual end-cap fraction handling
    # ============================================================
    if cap_fraction is None:
        side_area = 2.0 * np.pi * cyl_radius * cyl_height_working
        cap_area  = 2.0 * np.pi * cyl_radius**2
        cap_fraction_eff = cap_area / (side_area + cap_area)
        print(f"cap_fraction = Auto (None) → computed cap fraction = {cap_fraction_eff:.4f}")
    elif isinstance(cap_fraction, (int, float)):
        cap_fraction_eff = float(cap_fraction)
        if not (0.0 <= cap_fraction_eff <= 1.0):
            raise ValueError("cap_fraction must be in [0, 1] or None")
        print(f"cap_fraction (manual) = {cap_fraction_eff:.4f}")
    else:
        raise TypeError("cap_fraction must be None (Auto) or a float in [0, 1]")

    # ===============================
    # Deterministic Generation
    # ===============================
    print("Generating Variable Density Grid (Hash + Cylindrical Expansion)...")

    if generate_reverse_spiral:
        pts_forward = (num_points + 1) // 2
        pts_reverse = num_points // 2
    else:
        pts_forward = num_points
        pts_reverse = 0

    df_f = generate_cylinder_spiral(
        pts_forward, cyl_radius, cyl_height_working, cap_fraction_eff,
        reverse=False, 
        wall_thickness_mm=wall_thickness_mm, 
        vd_power_side=P_side, vd_power_caps=P_caps,
        index_offset=0,
        bottom_cutoff=bottom_cutoff,
        azimuth_density_ratio=azimuth_density_ratio,
        azimuth_weight_center_deg=azimuth_weight_center_deg
    )   

    if generate_reverse_spiral and pts_reverse > 0:
        df_r = generate_cylinder_spiral(
            pts_reverse, cyl_radius, cyl_height_working, cap_fraction_eff,
            reverse=True, rotate_deg=z_rotation_deg, flip_z=flip_poles,
            wall_thickness_mm=wall_thickness_mm, 
            vd_power_side=P_side, vd_power_caps=P_caps,
            index_offset=len(df_f),
            bottom_cutoff=bottom_cutoff,
            azimuth_density_ratio=azimuth_density_ratio,
            azimuth_weight_center_deg=azimuth_weight_center_deg
        )
        df = pd.concat([df_f, df_r], ignore_index=True)
    else:
        df = df_f

    df[['x','y','z']] *= 1000.0

    # ===============================
    # Cylindrical coordinates
    # ===============================
    r_xy = np.hypot(df['x'].to_numpy(), df['y'].to_numpy())
    phi_deg_cyl = np.degrees(np.arctan2(df['y'], df['x']))

    cyl_df = pd.DataFrame({
        'r_xy_mm': r_xy,
        'phi_deg': phi_deg_cyl,
        'z_mm': df['z'].to_numpy()
    })

    phi_mask = (
        (cyl_df['phi_deg'] >= phi_min_deg) &
        (cyl_df['phi_deg'] <= phi_max_deg)
    )

    cyl_df = cyl_df[phi_mask].reset_index(drop=True)

    # ===============================
    # Quantisation
    # ===============================
    cyl_df['r_xy_mm'] = np.round(cyl_df['r_xy_mm']).astype(int)
    H_mm = cyl_height_working * 1000.0

    if z_midpoint_zero:
        cyl_df['z_mm'] = np.round(cyl_df['z_mm']).astype(int)
    else:
        cyl_df['z_mm'] = np.round(cyl_df['z_mm'] + H_mm/2.0).astype(int)
        cyl_df['z_mm'] = np.clip(cyl_df['z_mm'], 0, int(round(H_mm)))

    cyl_df['phi_deg'] = np.round(cyl_df['phi_deg'], 1)

    # ===============================
    # Append Generation Settings
    # ===============================
    cyl_radius_working = cyl_radius + (wall_thickness_mm / 1000.0)

    settings_list = [
        f"cyl_radius_internal={cyl_radius:.3f}",
        f"cyl_radius_external={cyl_radius_working:.3f}",
        f"cyl_height_internal={cyl_height:.3f}",
        f"cyl_height_external={cyl_height_working:.3f}",
        f"num_points={num_points}",
        f"cap_fraction={cap_fraction}",
        f"cap_fraction_eff={cap_fraction_eff:.4f}",
        f"wall_thickness_mm={wall_thickness_mm}",
        f"P_side={P_side}",
        f"P_caps={P_caps}",
        f"generate_reverse_spiral={generate_reverse_spiral}",
        f"z_rotation_deg={z_rotation_deg}",
        f"flip_poles={flip_poles}",
        f"bottom_cutoff_mm={bottom_cutoff_mm}",
        f"z_midpoint_zero={z_midpoint_zero}",
        f"phi_min_deg={phi_min_deg}",
        f"phi_max_deg={phi_max_deg}",
        f"azimuth_density_ratio={azimuth_density_ratio}",
        f"azimuth_weight_center_deg={azimuth_weight_center_deg}"
    ]

    cyl_df['gen_settings'] = ""
    for i, setting in enumerate(settings_list):
        if i < len(cyl_df):
            cyl_df.loc[i, 'gen_settings'] = setting

    return cyl_df


if __name__ == "__main__":
    from config_grid import (
        OUTPUT_GRID_GEN,
        cyl_radius,
        cyl_height,
        num_points,
        cap_fraction,            
        wall_thickness_mm,      
        P_side,                  
        P_caps,                   
        generate_reverse_spiral,
        z_rotation_deg,
        flip_poles,
        bottom_cutoff_mm,
        z_midpoint_zero,
        phi_min_deg,
        phi_max_deg,
        azimuth_density_ratio,
        azimuth_weight_center_deg,
    )

    grid_dict = generate_measurement_grid(
        cyl_radius=cyl_radius,
        cyl_height=cyl_height,
        num_points=num_points,
        wall_thickness_mm=wall_thickness_mm,
        bottom_cutoff_mm=bottom_cutoff_mm,
        cap_fraction=cap_fraction,
        P_side=P_side,
        P_caps=P_caps,
        generate_reverse_spiral=generate_reverse_spiral,
        z_rotation_deg=z_rotation_deg,
        flip_poles=flip_poles,
        z_midpoint_zero=z_midpoint_zero,
        phi_min_deg=phi_min_deg,
        phi_max_deg=phi_max_deg,
        azimuth_density_ratio=azimuth_density_ratio,
        azimuth_weight_center_deg=azimuth_weight_center_deg
    )

    cyl_df = pd.DataFrame(grid_dict)
    cyl_df.to_csv(OUTPUT_GRID_GEN, index=False)
    print(
        f"Saved {OUTPUT_GRID_GEN} "
        f"(phi {phi_min_deg:.1f}…{phi_max_deg:.1f}°, mm units)"
    )