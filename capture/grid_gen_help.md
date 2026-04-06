# Measurement Grid Generator

This script generates a CSV file containing the coordinates for every measurement point in the grid. Cylindrical coordinates are used at this stage (instead of spherical coordinates used by the math stages) because they align directly with the physical movement of the measurement hardware.

---

## CSV Column Definitions

| Column | Description |
| :--- | :--- |
| **r_xy_mm** | The radial distance in millimeters. **Note:** *xy* denotes the distance within the cylindrical plane, not a spherical straight-line distance from the origin. |
| **phi_deg** | The azimuth angle of the point in degrees. |
| **z_mm** | The vertical height of the point in millimeters. |
| **gen_settings** | A metadata column containing the exact configuration parameters used to generate this specific grid. |
| **order_idx** | Represents the sequence in which points should be measured. This is added by the subsequent script, `path_plan.py`, to ensure efficient mechanical motion and prevent hardware collisions. |

---

## Grid Dimensions and Clearances

When configuring the grid, the specified `cyl_radius` and `cyl_height` represent the internal dimensions of the cylinder.

* **Internal vs. External Bounds:** By sizing the grid precisely around the physical dimensions of the Device Under Test (DUT), you ensure accuracy. The absolute maximum external dimensions of the generated grid will be the specified height and radius plus the `wall_thickness_mm` (default 50mm). You must ensure the internal dimensions clear any cables or accessories connected to the DUT.
* **Point Density:** The `num_points` parameter dictates the total number of measurement locations. A value between 1000 and 2000 is generally a reasonable default for standard operations.
* **Keep-Out Zones:** The script includes parameters to define physical limits where the microphone cannot travel:
    * `phi_min_deg` and `phi_max_deg`: Create an angular keep-out range for mechanical limits/endstops.
    * `bottom_cutoff_mm`: Creates a circular keep-out area at the center of the bottom cap to avoid collisions with the DUT support pole.

---

## Constructing the Base Grid: The Fibonacci Spiral

The grid points are initially laid out as a cylindrical surface wrapped by a Fibonacci spiral.

* **Surface Area Equality:** The Fibonacci spiral is mathematically desirable because each point represents an equal surface area. This avoids discrete bands or tightly packed rings of points typical of simple grid divisions.
* **The Reverse Spiral:** To further distribute points uniformly, a second Fibonacci spiral is wrapped onto the cylinder in the opposite direction and rotated by 90 degrees. It is highly recommended to leave these at default values.
* **Cap vs. Wall Allocation:** The script automatically calculates point distribution for end caps versus side walls based on the ratio of the surface area they represent.

---

## Grid Requirements for Sound Field Separation

After the base cylinder is wrapped, the script assigns a "magnetic attraction" value to each point, pulling it outward to give the cylinder walls and caps a volumetric "thickness." The maximum distance of this outward expansion is defined by the `wall_thickness_mm` parameter.

### The Blind Spot Problem (Bessel Nulls)
If a measurement grid consists of only a single radius (a single layer), the Spherical Harmonic Expansion (SHE) solver cannot track the behavior of the sound field as it travels through space, rendering field separation physically impossible.

While academic texts frequently depict a "dual-layer" approach, a grid with only two radii remains vulnerable. At certain frequencies, both layers may coincide with the zero-velocity nodes of the sound wave (Bessel Nulls), causing the system to lose its ability to resolve the acoustic field.

To solve this, the generator distributes points across a **continuous layer thickness** (e.g., 50mm). Because points exist at varied distances, if one radius falls into a null, nearby points at different radii remain effective, spreading the mathematical error spectrally.

---

## Mathematical Stability: Grouping and Coupling

### The Grouping Problem
To ensure blind spots are counteracted, there must be no "grouping" of points with similar radii. Radial variation must be uniformly distributed across the entire grid to maintain a consistent "second opinion" for the solver.

### The Spatial Coupling Problem (Mode Leakage)
The SHE matrix assumes that radial distance and the angle of every measurement point are entirely independent variables. Any correlation between a point’s angle and its radius causes spatial coupling, leading to ill-conditioning. The goal of the generator is to minimize this correlation as much as possible.

---

## The Sine-Hash Solution

To guarantee that the radius and angle remain uncorrelated without causing "clumping," the script utilizes a deterministic sine-hash.

1.  **Decoupling:** This hash assigns the radial pull value sequentially based on the point's order in the Fibonacci spiral. This effectively severs the link between a point's spatial coordinate and its distance from the origin.
2.  **Determinism vs. Randomization:** Pure randomization can result in "unlucky" correlations and non-repeatable performance. The sine-hash ensures high performance is consistent and repeatable.

---

## Advanced Settings

| Setting | Description |
| :--- | :--- |
| **Z-Axis Datum** | Controlled by `z_midpoint_zero`. Enabled: origin is in the middle (±Z). Disabled: datum is at the bottom (all +Z). |
| **Reverse Spiral** | `generate_reverse_spiral`, `z_rotation_deg`, and `flip_poles` adjust the secondary spiral orientation. |
| **Cap Fraction** | `cap_fraction` manually sets the ratio of points for caps. Leave as `None` for automatic calculation. |
| **Pull Strength** | `P_caps` and `P_side` dictate the mathematical strength of the hashed pull. |

**Note on $P_{side}$:** If using an exceptionally large grid, the side wall surface area increases more aggressively with radius. $P_{side}$ may need adjustment to prevent the point cloud from becoming too sparse.

### Pull Strength Formula
The formula for the pull strength is:

$$dr = d_{max} \cdot (u_k^P)$$

**Where:**
* $dr$ = The Radial Displacement
* $d_{max}$ = The Maximum Thickness
* $u_k$ = The Sine-Hash Value
* $P$ = The Power Constant

### Point Quantization
To ensure human-readable filenames and logs, coordinates are rounded:
* **Radial/Vertical:** Nearest millimeter.
* **Angles:** 0.1 degrees.

This quantization remains significantly smaller than the wavelength of 20kHz, ensuring no loss in acoustic resolution.