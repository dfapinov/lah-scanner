# The Measurement Grid Generator

The script generates a CSV file containing the coordinates for every measurement point in the grid. **Cylindrical coordinates** are used at this stage (instead of spherical coordinates used by the math stages) because they align directly with the physical movement of the measurement hardware:

* **Phi ($\phi$):** The rotating axis (turntable).
* **Radius ($R$):** The linear radial distance of the microphone arm.
* **Vertical ($Z$):** The linear height of the microphone on its rail.

---

### CSV Column Definitions

| Column | Description |
| :--- | :--- |
| **r_xy_mm** | The radial distance in millimeters. Note: *xy* denotes the distance within the cylindrical plane, not a spherical straight-line distance from the origin. |
| **phi_deg** | The azimuth angle of the point in degrees. |
| **z_mm** | The vertical height of the point in millimeters. |
| **gen_settings** | A metadata column containing the exact configuration parameters used to generate this specific grid. |
| **order_idx** | Represents the sequence in which points should be measured. This is added by the subsequent script, `path_plan.py`, to ensure efficient mechanical motion and prevent hardware collisions. |

---

### Grid Dimensions and Clearances

When configuring the grid, the specified `cyl_radius` and `cyl_height` represent the internal dimensions of the cylinder.

* **Internal vs. External Bounds:** Sizing the grid precisely around the physical dimensions of the DUT. The absolute maximum external dimensions will be the specified height and radius plus the `wall_thickness_mm` (default 50mm). You must ensure internal dimensions clear any cables or accessories.
* **Point Density:** The `num_points` parameter dictates the total number of measurement locations. A value between **1000 and 2000** is generally a reasonable default.
* **Keep-Out Zones:** * `phi_min_deg` and `phi_max_deg` create an angular keep-out range for mechanical endstops.
    * `bottom_cutoff_mm` creates a circular keep-out area at the center of the bottom cap to avoid collisions with the DUT support pole.

---

### Constructing the Base Grid: The Fibonacci Spiral

The grid points are initially laid out as a cylindrical surface wrapped by a Fibonacci spiral.

* **Surface Area Equality:** The Fibonacci spiral ensures each point represents an equal surface area, avoiding discrete bands or tightly packed rings.
* **The Reverse Spiral:** A second spiral is wrapped in the opposite direction and rotated by 90 degrees to further distribute points. It is recommended to leave these at default values.
* **Cap vs. Wall Allocation:** The script automatically calculates point distribution based on the ratio of surface area between end caps and side walls.

---

## Grid Requirements for Sound Field Separation

After the base cylinder is wrapped, the script assigns a **"magnetic attraction"** value to each point, pulling it outward to give the cylinder volume. The maximum distance of this expansion is defined by `wall_thickness_mm`.

#### The Blind Spot Problem (Bessel Nulls)
If a measurement grid consists of only a single radius, the SHE solver cannot track sound field behavior as it travels, making field separation impossible. 

While academic texts depict "dual-layer" approaches, two-layer grids remain vulnerable to **Bessel Nulls**—characteristic frequencies where both layers coincide with zero-velocity nodes. To solve this, the generator distributes points across a continuous layer thickness (default 50mm). Because points exist at varied distances, if one radius falls into a null, nearby points remain effective, spreading the mathematical error spectrally.

---

## Mathematical Stability: Grouping and Coupling

#### The Grouping Problem
To ensure blind spots are counteracted, there must be no "grouping" of points with similar radii. Radial variation must be uniformly distributed to maintain a consistent "second opinion" for the solver.

#### The Spatial Coupling Problem (Mode Leakage)
The SHE matrix assumes that radial distance and the angle of every point are entirely **independent variables**. Any correlation causes spatial coupling, leading to ill-conditioning. The generator's goal is to minimize this correlation.

---

### The Sine-Hash Solution
To guarantee radius and angle remain uncorrelated without "clumping," the script utilizes a deterministic sine-hash:

1.  **Decoupling:** The hash assigns radial pull values sequentially based on the point's order in the Fibonacci spiral, severing the link between spatial coordinates and distance from the origin.
2.  **Determinism vs. Randomization:** Unlike pure randomization, which can produce "unlucky" correlations, the sine-hash ensures high performance is repeatable across generations.

---

### Advanced Settings

| Setting | Description |
| :--- | :--- |
| **Z-Axis Datum** | Controlled by `z_midpoint_zero`. **Enabled:** Origin is in the middle (±Z). **Disabled:** Datum is at the bottom (all +Z). |
| **Reverse Spiral** | `generate_reverse_spiral`, `z_rotation_deg`, and `flip_poles` adjust secondary spiral orientation. |
| **Cap Fraction** | `cap_fraction` manually sets the ratio of points for caps. Leave as `None` for automatic calculation. |
| **Pull Strength** | `P_caps` and `P_side` dictate the strength of the hashed pull. These are tuned differently to maintain constant volumetric density. |

> **Note on $P_{side}$:** If using an exceptionally large grid, the side wall surface area increases aggressively with radius. $P_{side}$ may need adjustment to prevent the point cloud from becoming too sparse.

#### Pull Strength Formula
The formula for the pull strength is:

$$dr = d_{max} \cdot (u_k^P)$$

**Where:**
* $dr$ = The Radial Displacement
* $d_{max}$ = The Maximum Thickness
* $u_k$ = The Sine-Hash Value
* $P$ = The Power Constant

#### Point Quantization
To ensure human-readable logs, coordinates are rounded to the nearest **millimeter** and angles to **0.1 degrees**. This quantization is significantly smaller than the wavelength of 20kHz.