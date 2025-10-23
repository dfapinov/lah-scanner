#!/usr/bin/env python3
"""
Cylindrical Measurement Grid Generator with Jitter and Best Pattern Search
==========================================================================

This script generates a cylindrical measurement grid for Spherical Harmonic Expansion (SHE).
Coordinates are distributed on the surfaces of two cylindrical layers, each following a Fibonacci spiral pattern
with user-adjustable parameters for radius, height, and cap coverage.
Radial jitter can be applied to each point to break up perfectly regular spacing.
Runs multiple grid trials, flags points with excessive radial spacing, and saves the set with the fewest flags.

Purpose of jitter and spacing:
------------------------------
In SHE, evenly spaced or perfectly symmetric measurement grids can cause numerical
singularities in the least-squares solve. This happens because many points share the
same radius or height, making some spherical basis functions nearly linearly dependent.

To reduce the risk of numerical singularities, a Fibonacci spiral is used instead of a regular grid.
The spiral orientation of each cylindrical layer can be offset to minimize correlation between layers.
Additionally, random jitter can be applied to each point’s radius to avoid identical radial distances.
This breaks up regular spacing and prevents “double-layer shell” effects, where entire rings of points
at the same radius reinforce singularities at specific frequencies.

The nearest-neighbor spacing threshold (typically 8 mm) is chosen to keep point
separation above roughly half a wavelength at 20 kHz (≈17 mm in air), maintaining
stable behavior across the full audible band.

Grid Density Spacing
--------------------

To accurately capture spatial (angular) details up to a chosen maximum frequency f_max,
the minimum spacing between measurement points must satisfy the spatial Nyquist criterion:

   Δr ≤ λ / 2 = c / (2 f_max)
    
 where:
     Δr      = spacing between adjacent measurement points (metres)
     λ       = acoustic wavelength at frequency f_max (metres)
     c       = speed of sound in air (≈ 343 m/s)
     f_max   = highest frequency to be represented without spatial aliasing (Hz)

Example:
   • For f_max = 20 kHz → Δr ≤ 343 / (2 × 20,000) ≈ 8.6 mm  → use ≤ 8 mm
   • For f_max = 24 kHz → Δr ≤ 343 / (2 × 24,000) ≈ 7.15 mm → use ≤ 7 mm

 This ensures that the measurement grid resolves spatial detail up to the intended frequency
 without aliasing. A nearest-neighbour threshold check is used to verify this condition.
 Note: regular or perfectly uniform spacing should still be avoided to prevent geometric aliasing —
 this check is a safety guardrail, not a true optimiser for grid design.



Usage
-----

Configured via:  config_capture.py

Run from the command line:

    python stage1_grid_gen.py

By default, it reads parameters from `config_capture.py` and writes:
    • best_coordinates.csv — cylindrical coordinates in mm and degrees

Code Pipeline Overview
----------------------

1) Convert threshold and cutoff from mm → m.  
2) Generate Fibonacci spiral on cylinder:  
   – Side: golden-angle azimuth, linear height.  
   – Caps: √r radial spacing for uniform density.  
   – Optional Z flip, rotation, and radial/z jitter.  
   – Remove points near bottom cap center.  
3) Build grid (forward + optional reverse spiral).  
4) Test nearest-neighbor spacing on radius r = √(x²+y²+z²).  
   – Flag points exceeding threshold.  
   – Keep run with fewest flagged points.  
5) Repeat until zero flags or max runs reached.  
6) Convert best run to cylindrical and spherical coordinates.  
7) Quantise (mm, 0.1°) and save CSV outputs.

"""

import numpy as np                    # numerical operations on arrays
import pandas as pd                   # table-style data manipulation
import matplotlib.pyplot as plt       # plotting (imported but unused here)
from mpl_toolkits.mplot3d import Axes3D  # enables 3-D plotting if needed

# import user settings such as radius, height, jitter, etc.
from config_capture import (
    cyl_radius,              # cylinder radius in metres
    cyl_height,              # cylinder height in metres
    num_points,              # points per spiral (forward + reverse)
    cap_fraction,            # fraction of points placed on end-caps
    jitter_enabled,          # True/False toggle for random jitter
    jitter_range_mm,         # ± range of jitter in mm
    generate_reverse_spiral, # whether to make a mirrored second spiral
    z_rotation_deg,          # rotation of second spiral around Z axis
    flip_poles,              # flip Z coordinates of the second spiral
    bottom_cutoff_mm,        # drop points near bottom-cap centre
    k_neighbors,             # number of nearest neighbours to test
    threshold_mm,            # distance threshold in mm for flagging
    num_runs,                # how many random trials to perform
    run_until_zero,          # keep looping until no points are flagged
)

# --- unit conversions from mm → m for internal calculations ---
threshold     = threshold_mm / 1000.0
bottom_cutoff = bottom_cutoff_mm / 1000.0

def generate_cylinder_spiral(N, R, H, cap_frac,
                             reverse=False, jitter=False, jitter_mm=0,
                             rotate_deg=0.0, flip_z=False):
    """Generate one Fibonacci spiral around a cylinder."""
    phi = (-1 if reverse else 1) * np.pi * (3. - np.sqrt(5))  # golden-angle increment
    n_side = int(round(N*(1-cap_frac)))  # number of side-surface points
    n_caps = N - n_side                  # remaining points go to top/bottom
    n_top = n_caps//2 + (n_caps%2)       # split between top and bottom
    n_bot = n_caps - n_top

    pts = []  # list to collect xyz points

    # ---- side surface points ----
    for i in range(n_side):
        t = i/max(n_side-1,1)                # normalized height 0→1
        z = -H/2 + H*t                       # linear interpolation along height
        θ = (i*phi)%(2*np.pi)                # azimuth using golden angle
        pts.append([R*np.cos(θ), R*np.sin(θ), z])  # cylindrical → Cartesian

    # ---- top cap points ----
    for j in range(n_top):
        r = R*np.sqrt((j+0.5)/n_top)         # radial spacing for uniform density
        θ = (j*phi)%(2*np.pi)                # azimuth using golden angle
        pts.append([r*np.cos(θ), r*np.sin(θ), +H/2])  # z fixed at top

    # ---- bottom cap points ----
    for j in range(n_bot):
        r = R*np.sqrt((j+0.5)/n_bot)
        θ = (j*phi)%(2*np.pi)
        pts.append([r*np.cos(θ), r*np.sin(θ), -H/2])  # z fixed at bottom

    arr = np.array(pts)                      # convert list → NumPy array

    # optional flip in Z
    if flip_z:
        arr[:,2] *= -1

    # optional rotation about Z axis
    if rotate_deg != 0:
        ang = np.radians(rotate_deg)
        c,s = np.cos(ang), np.sin(ang)
        x2 = c*arr[:,0] - s*arr[:,1]
        y2 = s*arr[:,0] + c*arr[:,1]
        arr[:,0], arr[:,1] = x2, y2

    # ---- apply radial jitter ----
    if jitter and jitter_mm>0:
        jrad = jitter_mm/1000.0              # convert mm → m
        for k in range(arr.shape[0]):        # iterate all points
            x,y,z = arr[k]
            if abs(abs(z)-H/2)<1e-6:         # if on cap, perturb z
                arr[k,2] += np.random.uniform(-jrad/2, jrad/2)
            else:                             # if on side, perturb radius
                mag = np.hypot(x,y)
                if mag == 0:
                    continue
                dr = np.random.uniform(-jrad/2, jrad/2)
                arr[k,0] += (x/mag)*dr
                arr[k,1] += (y/mag)*dr

    df = pd.DataFrame(arr, columns=['x','y','z'])  # to DataFrame for convenience

    # ---- remove points near bottom-cap centre ----
    tol = (jitter_mm/1000.0)/2 if jitter and jitter_mm>0 else 0
    mask_bottom = (
        (np.abs(df['z'] + H/2) <= tol + 1e-8) &      # at bottom cap
        (np.hypot(df['x'], df['y']) <= bottom_cutoff) # within cutoff radius
    )
    if mask_bottom.any():
        df = df[~mask_bottom].reset_index(drop=True)  # drop flagged rows
    return df

# initialize best result trackers
best_flags = np.inf
best_df = None
run_count = 0

def evaluate_run():
    """Generate forward/reverse spirals, test spacing, update best result."""
    global best_flags, best_df, run_count
    run_count += 1

    # create forward spiral
    df_f = generate_cylinder_spiral(
        num_points, cyl_radius, cyl_height, cap_fraction,
        reverse=False, jitter=jitter_enabled, jitter_mm=jitter_range_mm
    )
    df_f['spiral'] = 'forward'

    # optionally add reversed second spiral
    if generate_reverse_spiral:
        df_r = generate_cylinder_spiral(
            num_points, cyl_radius, cyl_height, cap_fraction,
            reverse=True, jitter=jitter_enabled, jitter_mm=jitter_range_mm,
            rotate_deg=z_rotation_deg, flip_z=flip_poles
        )
        df_r['spiral'] = 'reverse'
        df = pd.concat([df_f, df_r], ignore_index=True)
    else:
        df = df_f

    # ---- spacing test ----
    r = np.linalg.norm(df[['x','y','z']].values, axis=1)  # radial distances
    flagged = np.zeros_like(r, dtype=bool)                # flag array
    for i in range(len(r)):
        diffs = np.abs(r - r[i])       # absolute distance differences
        diffs[i]=np.inf                # ignore self
        nnd = np.partition(diffs, k_neighbors)[:k_neighbors]  # nearest neighbours
        if np.any(nnd > threshold):    # flag if any exceed threshold
            flagged[i] = True

    n_flagged = flagged.sum()
    print(f"Run {run_count}: {n_flagged} flagged")  # report count

    # keep run with fewest flags so far
    if n_flagged < best_flags:
        best_flags = n_flagged
        best_df = df.copy()
    return (n_flagged == 0), df

# ---- main search loop ----
if run_until_zero:
    while True:
        zero, _ = evaluate_run()
        if zero:
            print(f"Zero-flag run on attempt {run_count}")
            break
else:
    for _ in range(num_runs):
        evaluate_run()

print(f"\nBest run flagged {int(best_flags)} points")

# === Output section ===
df = best_df.copy()

# convert from metres → millimetres for easier viewing
df[['x','y','z']] *= 1000.0

# ---- build cylindrical coordinate CSV ----
r_xy = np.hypot(df['x'].to_numpy(), df['y'].to_numpy())         # radial distance in XY
phi_deg_cyl = np.degrees(np.arctan2(df['y'].to_numpy(), df['x'].to_numpy()))  # azimuth angle
phi_deg_cyl = np.mod(phi_deg_cyl, 360.0)                        # wrap 0–360°
cyl_df = pd.DataFrame({
    'r_xy_mm': r_xy,
    'phi_deg': phi_deg_cyl,
    'z_mm': df['z'].to_numpy(),
    'spiral': df['spiral'].to_numpy()
})

# ---- compute spherical equivalents ----
r = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
with np.errstate(invalid='ignore', divide='ignore'):
    theta = np.arccos(np.clip(df['z'] / np.where(r==0, 1, r), -1.0, 1.0))
phi_sph = np.arctan2(df['y'], df['x'])
phi_deg_sph = np.degrees(phi_sph)
phi_deg_sph = np.mod(phi_deg_sph, 360.0)
theta_deg = np.degrees(theta)
sph_df = pd.DataFrame({
    'r_mm': r,
    'theta_deg': theta_deg,
    'phi_deg': phi_deg_sph,
    'spiral': df['spiral'].to_numpy()
})

# ---- quantise to whole mm and 0.1° ----
cyl_df['r_xy_mm'] = np.round(cyl_df['r_xy_mm']).astype(int)
H_mm = cyl_height * 1000.0
cyl_df['z_mm'] = np.round(cyl_df['z_mm'] + H_mm/2.0).astype(int)  # shift origin
cyl_df['z_mm'] = np.clip(cyl_df['z_mm'], 0, int(round(H_mm)))     # clamp bounds
cyl_df['phi_deg'] = np.round(cyl_df['phi_deg'], 1)
cyl_df.loc[cyl_df['phi_deg'] >= 360.0, 'phi_deg'] = 0.0

sph_df['r_mm']      = np.round(sph_df['r_mm']).astype(int)
sph_df['theta_deg'] = np.round(sph_df['theta_deg'], 1)
sph_df['phi_deg']   = np.round(sph_df['phi_deg'], 1)
sph_df.loc[sph_df['phi_deg'] >= 360.0, 'phi_deg'] = 0.0

# ---- write outputs ----
cyl_df.to_csv("best_coordinates.csv", index=False)
print("Saved best_coordinates.csv (phi in degrees, mm units)")

# optional spherical output (disabled by default)
# sph_df.to_csv("best_spiral_spherical.csv", index=False)
# print("Saved best_spiral_spherical.csv (theta/phi in degrees, mm units)")

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