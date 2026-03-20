# config_capture.py
"""
User settings for the capture scripts that generate measruemtn grids and plan the robotic path.
"""

""" Grid Generator """
# ───────────────────────────────────────────────
# Cylinder and pattern settings:
OUTPUT_GRID_GEN         = "jan_cylinder_test.csv" # Default "measurment_grid.csv"
cyl_radius              = 0.10    # Cylinder radius (m)
cyl_height              = 0.43     # Cylinder height (m)
num_points              = 750     # Points per spiral (total = forward + reverse)
cap_fraction            = None    # Fraction of points on both end‐caps combined. None = Auto (end cap to side wall area based). 0-1 = Manually enter a fraction.
jitter_enabled          = True    # Enable radial jitter
jitter_range_mm         = 25      # Jitter range in mm (± half)
generate_reverse_spiral = True    # Make second (reverse) spiral
z_rotation_deg          = 90   # Rotate second spiral around Z (deg)
flip_poles              = False   # Flip Z sign of second spiral
bottom_cutoff_mm        = 35     # Remove bottom‐cap points within this radius from center (mm) for support pole
z_midpoint_zero         = True    # True = Z axis centred at 0 mm (equal negative and positive values). False = Z axis all positive like physical robot axis.
cap_z_shift_mm = 50
# --- Coprime Shell Offsets (mm) ---
# Total span: 50mm. Calculated to protect 20Hz - 20kHz.
coprime_offsets_mm = [0.0, 7.0, 19.0, 50.0]

# --- Variable Density Grid ("Magnetic Pull") ---
var_density_enabled = True
var_density_max_mm  = 50.0   # The maximum distance points can evaporate outwards (D_max)
var_density_power   = 0.5    # Power law exponent (1.0 = linear smear, 3.0 = dense core/sparse cloud)
P_side = 0.5
P_top = 0.5
# Phi cut limits (degrees, cylindrical azimuth)
phi_min_deg = -175.0
phi_max_deg =  175.0

# Optimisation settings:
k_neighbors   = 20      # how many points to group for nearest neighbour proximity check
threshold_mm  = 5       # radial distance to neighbour threshold (mm)
num_runs      = 100     # max runs, if not run_until_zero
run_until_zero = False  # if True, loop until zero flagged

""" Path Planner """
# ───────────────────────────────────────────────
INPUT_PATH_PLAN          = OUTPUT_GRID_GEN # Default "best_coordinates.csv"
OUTPUT_PATH_PLAN         = "path_planned.csv" # Default "scan_path.csv"

#Advanced:
DELTA_THETA_DEG    = 7.5   # θ-bin width (deg). Smaller → more bins, smoother θ progression.
CAP_TOL_MM         = var_density_max_mm + 1   # Points within ±CAP_TOL_MM of min/max z are treated as caps (top/bottom).
SIDE_SNAKE_START   = "up"  # Initial z traversal direction for sidewall bins: "up" or "down".
CAP_ORDER          = "side_then_caps"  # Handle all side bins first, then caps; or "interleave_caps".
CAP_RADIAL_ORDER   = "outer_to_inner"  # For caps: sweep radially "inner_to_outer" or "outer_to_inner".
PRINT_SUMMARY = True  # Whether to print counts and θ-bin info after writing the path
