# Measurement Grid Generator

The Measurement Grid Generator script creates a CSV file containing coordinates for every measurement point in a 3D grid. While the mathematical stages of the solver typically use spherical coordinates, this generator uses **cylindrical coordinates** because they align directly with the physical movement of standard measurement hardware:

* **Phi ($\phi$):** The rotating axis (e.g., a turntable).
* **Radius ($R$):** The linear radial distance of the microphone arm.
* **Vertical ($Z$):** The linear height of the microphone on its rail.

---

## Grid Dimensions and Clearances

When configuring the grid, the parameters `cyl_radius` and `cyl_height` represent the **internal dimensions**.

* **Internal vs. External Bounds:** Sizing the grid precisely around the physical dimensions of the Device Under Test (DUT). Note that the absolute maximum external dimensions will be the specified height and radius **plus** the `wall_thickness_mm` (default 50mm). Ensure these dimensions clear any cables or accessories.
* **Point Density:** The `num_points` parameter dictates the total number of measurement locations. A value between **1000 and 2000** is generally a reasonable default.
* **Keep-Out Zones:** * `phi_min_deg` and `phi_max_deg` create an angular keep-out range (useful if an endstop prevents a full 360-degree rotation).
    * `bottom_cutoff_mm` creates a circular keep-out area at the center of the bottom cap to avoid collisions with the DUT support pole.

---

## CSV Column Definitions

| Column | Description |
| :--- | :--- |
| `r_xy_mm` | The radial distance in millimeters. **Note:** $xy$ denotes the distance within the cylindrical plane, not a spherical straight-line distance from the origin. |
| `phi_deg` | The azimuth angle of the point in degrees. |
| `z_mm` | The vertical height of the point in millimeters. |
| `gen_settings` | Metadata containing the exact configuration parameters used for this specific grid. |
| `order_idx` | The sequence in which points should be measured. This is added by `path_plan.py` to ensure efficient motion and prevent collisions. |

---

## Constructing the Base Grid: The Fibonacci Spiral

The grid is initially laid down as a cylindrical surface wrapped by a **Fibonacci spiral**.

1.  **Surface Area Equality:** The Fibonacci spiral is mathematically desirable because each point represents an equal surface area, avoiding discrete bands or rings of points.
2.  **The Reverse Spiral:** To further distribute points, a second Fibonacci spiral is wrapped in the opposite direction and rotated by 90 degrees. It is recommended to leave `generate_reverse_spiral`, `z_rotation_deg`, and `flip_poles` at their default settings.
3.  **Cap vs. Wall Allocation:** The script automatically calculates how many points belong on the end caps versus the side walls based on their surface area ratio. This can be overridden by `cap_fraction`, though it is recommended to leave this as `None`.

---

## Sound Field Separation Requirements

After the base cylinder is wrapped, points are assigned a "magnetic attraction" value and pulled outward to create a volumetric "thickness" (defined by `wall_thickness_mm`). This radial depth is critical for sound field separation.

### The Blind Spot Problem (Bessel Nulls)
If a measurement grid consists of only a single radius (a single layer), the Spherical Harmonic Expansion (SHE) solver cannot track the behavior of the sound field through space. Even a dual-layer approach is vulnerable: at certain characteristic frequencies, both layers may coincide with the zero-velocity nodes of the wave. 

In mathematics, this is a **Bessel Null**. To solve this, the generator distributes points across a continuous "thickness" (default 50mm). If one specific radius falls into a null for a given wavelength, nearby points remain effective, spreading the error spectrally.

### Mathematical Stability: Grouping and Coupling
* **The Grouping Problem:** To ensure radial variation is uniform, the script prevents "grouping" of points with similar radii. 
* **Spatial Coupling (Mode Leakage):** The SHE matrix assumes radial distance and angle are independent variables. Any correlation between the two causes spatial coupling, leading to ill-conditioning. To minimize this, the script uses a **deterministic sine-hash**.

### The Sine-Hash Solution
The sine-hash assigns radial pull values based on the point's sequence in the Fibonacci spiral (its index) rather than its spatial location. This effectively de-correlates the radius and angle while maintaining a repeatable, high-performance distribution.

---

## Advanced Settings

| Setting | Description |
| :--- | :--- |
| **Z-Axis Datum** | `z_midpoint_zero` sets the origin to either the middle of the cylinder (positive/negative Z) or the bottom (all Z positive). |
| **Pull Strength** | `P_caps` and `P_side` dictate the mathematical strength of the hashed pull. These are tuned differently to maintain constant volumetric density as the side wall area increases with radius. |

### Pull Strength Formula
The radial displacement is calculated as:

$$dr = d_{max} \cdot (u_k^P)$$

**Where:**
* $dr$ = Radial Displacement
* $d_{max}$ = Maximum Thickness (`wall_thickness_mm`)
* $u_k$ = The Sine-Hash Value
* $P$ = The Power Constant (`P_caps` or `P_side`)

> **Note on $P_{side}$:** For exceptionally large grids, the surface area of the side walls increases aggressively with radius. $P_{side}$ may need adjustment to prevent the point cloud from becoming too sparse.

---

## Point Quantization
To ensure human-readable logs, coordinates are rounded to the nearest millimeter and angles to 0.1 degrees. This quantization level is significantly smaller than the wavelength of 20kHz, ensuring no loss of acoustic integrity.