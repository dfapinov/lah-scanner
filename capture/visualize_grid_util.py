#!/usr/bin/env python3
"""
visualize_grid_util.py

Checks a 3D coordinate grid for irregular radial spacing and plots the coordinates.

Usage:
    python visualize_grid_util.py --coordinates <filename.csv>
    e.g. "python visualize_grid_util.py --coordinates best_coordinates.csv"

Supported column formats (case-insensitive):
  • Cartesian:   x, y, z                          (also x_m, y_m, z_m)
  • Cylindrical: r_xy, phi_deg, z                 (phi in degrees)
  • Spherical:   r_m, theta_deg, phi_deg          (theta = colatitude, deg)
"""

import sys                     # system-related utilities (e.g., exiting the program)
import argparse                # command-line argument parsing
import pandas as pd            # read/write CSV and tabular data
import numpy as np             # numerical operations on arrays
import matplotlib.pyplot as plt # plotting library
from mpl_toolkits.mplot3d import Axes3D  # enables 3D plotting in Matplotlib (import needed even if unused)


def _colmap(df):
    """Return a dict mapping lowercased column names to original names."""
    # Create a dictionary mapping lowercase column names to their exact names in the CSV
    return {c.lower().strip(): c for c in df.columns}


def _get_xyz(df):
    """
    Detect columns and return x, y, z as numpy arrays.
    Supports Cartesian (x,y,z), Cylindrical (r_xy,phi_deg,z), Spherical (r_m,theta_deg,phi_deg).
    """
    cm = _colmap(df)           # Get lowercase → original column name mapping
    cols = set(cm.keys())      # Set of lowercase column names for quick membership testing

    # ── Try Cartesian formats first ────────────────────────────────────────────
    cart_sets = [
        {"x", "y", "z"},       # plain x,y,z (already in meters)
        {"x_m", "y_m", "z_m"}, # explicitly in meters
        {"x_mm", "y_mm", "z_mm"},  # explicitly in millimeters
    ]
    for needed in cart_sets:
        if needed.issubset(cols):       # If all required columns exist
            if needed == {"x_mm", "y_mm", "z_mm"}:
                # Convert mm to meters
                x = df[cm["x_mm"]].to_numpy().astype(float) / 1000.0
                y = df[cm["y_mm"]].to_numpy().astype(float) / 1000.0
                z = df[cm["z_mm"]].to_numpy().astype(float) / 1000.0
                print("Detected Cartesian columns in mm (x_mm, y_mm, z_mm) → converted to meters.")
            else:
                # Use as-is (meters)
                x = df[cm[list(needed)[0]]].to_numpy()
                y = df[cm[list(needed)[1]]].to_numpy()
                z = df[cm[list(needed)[2]]].to_numpy()
                print("Detected Cartesian columns.")
            return x, y, z              # Return results immediately if found

    # ── Try Cylindrical format: r_xy, phi_deg, z ───────────────────────────────
    cyl_needed = {"r_xy", "phi_deg", "z"}
    if cyl_needed.issubset(cols):
        r_xy = df[cm["r_xy"]].to_numpy().astype(float)
        phi = np.deg2rad(df[cm["phi_deg"]].to_numpy().astype(float))  # convert degrees to radians
        z = df[cm["z"]].to_numpy().astype(float)
        # Convert cylindrical → Cartesian
        x = r_xy * np.cos(phi)
        y = r_xy * np.sin(phi)
        print("Detected Cylindrical columns (r_xy, phi_deg, z) → converted to Cartesian.")
        return x, y, z

    # ── Cylindrical in millimeters ─────────────────────────────────────────────
    cyl_needed_mm = {"r_xy_mm", "phi_deg", "z_mm"}
    if cyl_needed_mm.issubset(cols):
        r_xy = df[cm["r_xy_mm"]].to_numpy().astype(float) / 1000.0
        phi = np.deg2rad(df[cm["phi_deg"]].to_numpy().astype(float))
        z = df[cm["z_mm"]].to_numpy().astype(float) / 1000.0
        x = r_xy * np.cos(phi)
        y = r_xy * np.sin(phi)
        print("Detected Cylindrical columns in mm (r_xy_mm, phi_deg, z_mm) → converted to Cartesian (m).")
        return x, y, z

    # ── Try Spherical format: r_m, theta_deg, phi_deg ──────────────────────────
    sph_needed = {"r_m", "theta_deg", "phi_deg"}
    if sph_needed.issubset(cols):
        r = df[cm["r_m"]].to_numpy().astype(float)
        th = np.deg2rad(df[cm["theta_deg"]].to_numpy().astype(float))  # convert to radians
        ph = np.deg2rad(df[cm["phi_deg"]].to_numpy().astype(float))
        # Convert spherical → Cartesian
        x = r * np.sin(th) * np.cos(ph)
        y = r * np.sin(th) * np.sin(ph)
        z = r * np.cos(th)
        print("Detected Spherical columns (r_m, theta_deg, phi_deg) → converted to Cartesian.")
        return x, y, z

    # ── Spherical in millimeters ───────────────────────────────────────────────
    sph_needed_mm = {"r_mm", "theta_deg", "phi_deg"}
    if sph_needed_mm.issubset(cols):
        r = df[cm["r_mm"]].to_numpy().astype(float) / 1000.0
        th = np.deg2rad(df[cm["theta_deg"]].to_numpy().astype(float))
        ph = np.deg2rad(df[cm["phi_deg"]].to_numpy().astype(float))
        x = r * np.sin(th) * np.cos(ph)
        y = r * np.sin(th) * np.sin(ph)
        z = r * np.cos(th)
        print("Detected Spherical columns in mm (r_mm, theta_deg, phi_deg) → converted to Cartesian (m).")
        return x, y, z

    # ── If none of the above matched, raise an error ───────────────────────────
    raise ValueError(
        "CSV must contain either Cartesian (x,y,z or x_m,y_m,z_m), "
        "Cylindrical (r_xy,phi_deg,z), or Spherical (r_m,theta_deg,phi_deg) columns. "
        f"Found: {list(df.columns)}"
    )


def main():
    # ── Parse command-line arguments ──────────────────────────────────────────
    p = argparse.ArgumentParser(description="Check 3D coordinate grid spacing irregularities.")
    p.add_argument(
        "--coordinates",                 # name of command-line flag
        dest="coord_file",               # variable name to store it in
        help="Path to coordinates CSV file (e.g., path_planned_coordinates.csv)",
    )
    args = p.parse_args()                # read arguments from CLI

    # If user forgot to supply a file path, print usage help and exit
    if not args.coord_file:
        print("Usage: python plot_grid.py --coordinates <filename.csv>")
        sys.exit(1)

    # ── Load CSV file and extract Cartesian coordinates ───────────────────────
    df = pd.read_csv(args.coord_file)    # read the CSV into a DataFrame
    x, y, z = _get_xyz(df)               # detect coordinate system and get x,y,z arrays

    # Center the Z-axis for distance calculations
    z_centered = z - 0.5 * (z.max() + z.min())  # shift z so that mean height is zero
    r = np.sqrt(x**2 + y**2 + z_centered**2)    # compute full 3D radius from origin

    # ── Parameters for irregularity check ─────────────────────────────────────
    k_neighbors = 20      # how many nearby points to check for each point
    threshold = 8e-3      # threshold in meters (8 mm) for uneven spacing

    # ── Initialize arrays for results ─────────────────────────────────────────
    flag_any_far = np.zeros(len(r), dtype=bool)     # True if any neighbor too far
    max_nearest_dist = np.zeros(len(r))             # largest distance among nearest neighbors
    avg_nearest_dist = np.zeros(len(r))             # average distance among nearest neighbors

    # ── Loop over each point and check its radial spacing ─────────────────────
    for i in range(len(r)):
        diffs = np.abs(r - r[i])       # absolute distance to all other points
        diffs[i] = np.inf              # ignore self (distance 0)
        nearest_diffs = np.partition(diffs, k_neighbors)[:k_neighbors]  # get k smallest
        max_nearest_dist[i] = nearest_diffs.max()  # store max among nearest
        avg_nearest_dist[i] = nearest_diffs.mean() # store average
        if np.any(nearest_diffs > threshold):      # check if any exceeds threshold
            flag_any_far[i] = True

    # ── Add diagnostic columns back to the DataFrame ─────────────────────────
    df_out = df.copy()                                 # make a copy for safety
    df_out["any_far_radial_neighbor"] = flag_any_far   # True/False column
    df_out["max_nearest_radial_dist_m"] = max_nearest_dist
    df_out["avg_20_nearest_radial_dist_m"] = avg_nearest_dist

    # ── Report results to console ─────────────────────────────────────────────
    num_flagged = int(flag_any_far.sum())              # how many were flagged
    print(f"{num_flagged} out of {len(r)} points have ≥1 of their {k_neighbors} nearest radial neighbors > {threshold*1000:.1f} mm.\n")

    # If any were flagged, print which indices
    if num_flagged > 0:
        print("Flagged points and their maximum nearest-neighbor distances:")
        for idx in np.where(flag_any_far)[0]:
            dist_mm = max_nearest_dist[idx] * 1000
            print(f" Point index {idx}: max nearest-neighbor Δr = {dist_mm:.3f} mm")

    # ── Plot the 3D coordinate grid ───────────────────────────────────────────
    fig = plt.figure(figsize=(8, 6))                      # create figure
    ax = fig.add_subplot(111, projection='3d')            # 3D axis

    # Plot normal (OK) points in gray
    ax.scatter(x[~flag_any_far], y[~flag_any_far], z[~flag_any_far],
               c='gray', s=10, label=f'All neighbors ≤ {threshold*1000:.1f} mm')

    # Plot flagged (irregular) points in red
    ax.scatter(x[flag_any_far], y[flag_any_far], z[flag_any_far],
               c='red', s=20, label=f'>=1 neighbor > {threshold*1000:.1f} mm')

    # ── Enforce equal axis scaling ────────────────────────────────────────────
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Label axes and set title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Points — Flagged if Any of {k_neighbors} Neighbors > {threshold*1000:.0f} mm')
    ax.legend()                   # show legend
    plt.tight_layout()            # adjust spacing to fit
    plt.show()                    # display the figure window


if __name__ == "__main__":
    main()  # only run if script is called directly, not imported

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