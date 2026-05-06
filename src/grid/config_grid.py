# config_capture.py
"""
User settings for the capture scripts that generate measruemtn grids, drive robotic microphone, sweep and IR capture.
"""

""" Grid Generator """
# ───────────────────────────────────────────────
# Cylinder and pattern settings:
OUTPUT_GRID_GEN         = "jan_cylinder_test.csv" # Default "measurment_grid.csv"
cyl_radius_mm           = 200.0    # Cylinder internal radius (mm)
cyl_height_mm           = 500.0    # Cylinder internal height (mm)
num_points              = 1000     # Total points for the generated grid (forward + reverse spirals combined)

# Keep Out areas. 
phi_min_deg = -170.0 # Phi cut limits (degrees, cylindrical azimuth)
phi_max_deg =  180.0 # Phi cut limits (degrees, cylindrical azimuth)
bottom_cutoff_mm  = 30   # Remove bottom‐cap points within this radius from center (mm) for support pole

# --- Variable Density Grid ("Magnetic Pull") ---
wall_thickness_mm  = 50.0   # The maximum distance points can be pulled outwards (D_max)
P_side = 0.5 # Magntic pull bias on cylinder sides - >0.5 
P_caps = 0.8 # Power for magntic pull on cylinder caps

# --- Azimuthal Weighting (for non-uniform density on cylinder walls) ---
azimuth_density_ratio     = 1.0     # Front-to-back point density ratio. 1.0 = uniform. 5.0 = front is 5x denser than back. 
                                    # Smaller values (2.0 - 10.0) create a wide, smooth fade. Large values (>20) create a narrow band.
azimuth_weight_center_deg = 0.0     # Angle (deg) for the center of the high-density zone.

tweeter_pos               = None    # Optional tweeter coordinate for downstream processing. Cylindrical tuple: (r_mm, phi_deg, z_mm) e.g., (150.0, 180.0, 300.0)

# Optional: Define grid using physical waypoints instead of radius/height directly.
# Provide as a cylindrical tuple: (r_mm, phi_deg, z_mm) e.g., (200.0, 180.0, 150.0)
top_crit_pos              = None
bot_crit_pos              = None


generate_reverse_spiral = True    # Make second (reverse) spiral
z_rotation_deg          = 90   # Rotate second spiral around Z (deg)
flip_poles              = False   # Flip Z sign of second spiral
z_midpoint_zero         = True    # True = Z axis centred at 0 mm (equal negative and positive values). False = Z axis all positive like physical robot axis.
cap_fraction            = None    # Fraction of points on both end‐caps combined. None = Auto (end cap to side wall area based). 0-1 = Manually enter a fraction.

""" Path Planner """
# ───────────────────────────────────────────────
INPUT_PATH_PLAN          = OUTPUT_GRID_GEN # Default "best_coordinates.csv"
OUTPUT_PATH_PLAN         = "path_planned.csv" # Default "scan_path.csv"

#Advanced:
DELTA_THETA_DEG    = 7.5   # θ-bin width (deg). Smaller → more bins, smoother θ progression.
CAP_TOL_MM         = wall_thickness_mm + 1   # Points within ±CAP_TOL_MM of min/max z are treated as caps (top/bottom).
SIDE_SNAKE_START   = "up"  # Initial z traversal direction for sidewall bins: "up" or "down".
