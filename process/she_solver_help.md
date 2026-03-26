# Understanding the SHE Solver: Physics, Limits, and Linear Algebra

This stage of the pipeline involves taking the windowed, smoothed, and origin-aligned audio data in the complex frequency domain and solving for the Spherical Harmonic (SH) coefficients. This is the mathematical engine of the project. It translates discrete microphone measurements into a continuous 3D sound field and separates the loudspeaker's direct sound (outgoing field) from the room's acoustic reflections (incoming field, or more accurately, the standing wave component on the measurement surface).

To use this solver effectively, it is helpful to understand how to balance its settings, why certain limits are enforced, and how the underlying processing choices impact your final results.

---

## 1. Matrix Condition vs. Fit Residual Error

When evaluating the quality of a solve, the user can monitor two metrics that are displayed as the solve progresses and as summary plots at completion.

* **Fit Residual Error (%):** This is the difference between your raw microphone data and the sound field rebuilt by the solver. A lower percentage means the solver's mathematical model closely matches reality.
* **Matrix Condition Number:** This represents the stability of the linear algebra. A well-conditioned matrix (low number) means the data is over-determined and robust. An ill-conditioned matrix (high number) means the math is fragile.

**The Balancing Act:** It is very easy to drive the Fit Residual Error down to near zero simply by cranking the Order $N$ higher. However, doing so rapidly destroys the Matrix Condition. The solver begins applying massive, non-physical values to the coefficients. This occurs if the solver attempts to fit components not supported by the level of spatial sampling (noise or aliasing), leading to large, non-physical coefficients. You must balance Order $N$ to achieve the lowest residual error while keeping the matrix well-conditioned, indicating the math is describing real, physical, coherent sound fields.

---

## 2. Establishing the Order N Ceiling

Before the solver can run, it requires a maximum Order $N$ to act as a hard cap on what spatial detail the input data can support. As established in the Stage 3 Optimizer process, this "Tipping Point" is identified specifically by looking at the RFT range for the point where the internal-to-external fields energy ratio is at its peak before declining due to ill-conditioning. Once this stable ceiling is established, we still need to decide how the Stage 4 solver "Climbs" to that target: the rate at which Order $N$ grows with frequency.

---

## 3. The Rate of Growth: $kr$ vs. Source Complexity

Determining how rapidly $N$ increases is a balance between the physical size of the grid and the actual complexity of the loudspeaker radiation.

### The Grid Size Demand ($kr + x$)
Spatial complexity grows faster with frequency for larger measurement grids. If we compare two spheres—one small and one large—a fixed angle range projected from the center (for example, 10 degrees) corresponds to a greater physical arc-distance along the surface of the larger sphere.

This increased distance allows for greater phase variation in the sound on the surface of the larger sphere compared with the smaller one, enabling tighter lobes and more complex spatial structure per degree. This is the meaning of angular spatial detail. This behavior is governed by the wavenumber $k$ and the grid radius $r$:

$$kr = \frac{2\pi r}{\lambda}$$

The industry standard for a stable, high-resolution solve is $kr + 2$.

### The Internal Source vs. External Field
It is critical to distinguish between the complexity of the Internal Source (the speaker) and the External Field (the room):

* **The External Field:** Reflections arrive from all angles across the entire sampled volume, requiring high spatial resolution ($N$) to map correctly. A larger measurement grid will contain a more complex "room field."
* **The Internal Source:** The speaker's complexity, on the other hand, is not tied to the measurement grid size. It is defined by the source's physical size relative to the wavelength. Driver size, baffle dimensions, horns, and cabinet edges all contribute. In fact, as the distance from the source increases, the complexity typically drops because we move toward far-field plane-wave conditions at high frequencies.

In many cases, the internal source is spatially "simpler" than the room field, especially as the grid gets larger. While the grid radius grows (increasing $kr$ demand), the speaker remains a constant size. It is therefore optimal to keep measurement grids as tight to the DUT dimensions as practically possible to minimize this complexity gap.

This internal-to-external complexity relationship makes it difficult to define an ideal rate of climb for Order $N$ that applies to all measurement grids and DUTs. Be reassured that in practice, provided the maximum supported Order $N$ is not exceeded, the rate of climb vs. frequency primarily serves as a fine-tuning parameter rather than a factor that fundamentally determines the quality of the result.

---

## 4. User Controls: Adjusting the "Climb"

The solver provides two primary ways to adjust the growth rate of Order $N$ to prevent over-fitting simple sources while maintaining room rejection.

### The $kr + x$ Offset
The user can adjust the "Stability Offset" (the $+2$ in $kr + 2$).
* **Higher Offset (e.g., +2):** Pushes the Order $N$ growth higher earlier in the frequency range. This is the default and prioritizes room field separation. The cost is possibly over-fitting simple internal sources, leading to artifice in the response.
* **Lower Offset (e.g., +0):** Damps the growth rate, protecting simple internal sources from being over-fitted by high-order harmonics too early. This may occur if the measurement grid is large or the source is spatially simple. The cost may be reduced room field separation.

### The Manual Order Table
For advanced control, a manual frequency-to-order mapping can be used.

---

## 5. Verification and Interpretation

To assess the quality of the solve beyond simple residual error and matrix condition metrics, users can study the quality of sound field separation by exporting pressures for both internal and external fields:

* **External Field above RFT Range:** Above the RFT, the external field should be near the environment noise floor (typically -25dB to -30dB) as the reflections should have been removed by windowing and only noise remains.
* **External Field below RFT Range:** There are room modes and reflections in this range. A higher SPL here suggests the solver was able to separate more of the room field, provided the following:
    * **Internal Field:** The internal field response appears anechoic, smooth, and physically plausible.
    * **Watch out for:** Peaks or dips that show up in both external and internal responses. This suggests ill-conditioning where the solver incorrectly assigned energy to both fields to make the total fit error lower at the measurement surface.

---

## 6. Linear vs. Logarithmic Frequency Resolution

Many acoustic analysis tools downsample high-resolution FFT data, which is intrinsically on a linear frequency axis, into logarithmic bins (like 1/6th octave), both for visual reasons and to save CPU time in processing. This solver, however, computes the full linear FFT bins across the entire spectrum.

### The Advantages of Linear Processing:
* **Phase Coherence:** Solving every linear bin preserves the continuous phase relationship of the data originating from the FFT process.
* **Time-Domain Reversibility:** Because we maintain the full linear complex frequency response, we can easily perform an Inverse FFT (IFFT) on the separated outgoing field to generate a clean, reflection-free Impulse Response (IR). If we solved logarithmically, the results would need to be interpolated from log back to a linear scale; this back-and-forth between dense and sparse frequency grids is a lossy process that introduces error.

### The HF Cap:
While we process linear bins, the solver implements an optional High-Frequency (HF) cap. If you input a 96 kHz sample rate measurement, there is no value in solving thousands of ultrasonic frequency bins. At those ultra-short wavelengths, the physical measurement grid is far too sparse to support field separation. Capping the maximum frequency saves CPU time. A modern multi-core processor is able to process the dense linear solve in a reasonable timeframe—expect from 5 to 30 minutes depending on the maximum length of window used in the Stage 1 FDW processing.

---

## 7. The Physics and the Math

For those studying the code, you may notice apparent discrepancies between this script and published academic math texts on Spherical Harmonics. This is a deliberate choice to align the mathematics with standard audio engineering practices.

### The FFT Time Convention and Hankel 2
In theoretical literature, harmonic wave propagation is often defined using the $e^{-i\omega t}$ time convention, which pairs with Hankel functions of the first kind to describe outgoing waves. However, modern signal processing, audio engineering, and standard FFT algorithms utilize the $e^{+j\omega t}$ time convention.

To avoid convoluting the code with multiple complex conjugations just to satisfy canonical math texts, this solver uses the $e^{+j\omega t}$ time convention consistent with FFT throughout the pipeline. As a direct result, Hankel functions of the second kind ($h_n^{(2)}$) are used to represent the outgoing wave field. While this may look "backward" to a pure mathematician, it is the correct physical representation for our chosen time convention.

### Mixed Dual Basis: Separating Wave Fields
To effectively isolate direct sound from room reflections, we employ a dual basis that can represent both waves traveling outward (diverging) from the origin and waves traveling inward (converging) toward the origin from the environment.

* **The Outgoing (diverging) Field:** This represents the sound traveling outward from the source. It is modeled using Spherical Hankel functions of the second kind ($h_n^{(2)}$).
* **The Inward (converging) Field:** This represents the room's response. It is modeled using Spherical Bessel functions of the first kind ($j_n$).

**Why use $j_n$ instead of a pure Hankel basis?**
In many texts, a field is decomposed into incoming ($h_n^{(1)}$) and outgoing ($h_n^{(2)}$) Hankel functions. However, in practical interior acoustics, using the Spherical Bessel function $j_n$ is more physically intuitive.

A room reflection passing through a measurement sphere does not disappear at the origin; it enters the volume, passes through, and exits the other side. A pure incoming Hankel function ($h_n^{(1)}$) is singular at the origin, meaning it does not describe the "exit wound" at the other side. By contrast, the Spherical Bessel function ($j_n$) is regular (finite) at the origin. It can be interpreted as a superposition of the balanced inward and outward propagating components of the room field, making it suitable for representing fields that pass through the measurement volume. This is frequently referred to as the "standing wave component" because the inward and outward energy is balanced, resulting in net zero radial flux across the measurement surface.

---

## 8. RFT and Sound Field Separation

A common misconception is that data within the Reflection-Free Time (RFT) window can be treated as a "raw" anechoic response and simply merged with separated data from lower frequencies. While splicing magnitude responses is standard for 2D graphs, it is mathematically incompatible with 3D spatial reconstruction.

To propagate a sound field to new coordinates, the full frequency range must be fitted with spherical harmonic coefficients. This leaves two choices for the RFT range:

1.  **Forced Outgoing:** Trust that the windowed data is 100% anechoic and force the solver to use exclusively Outgoing (Hankel) terms.
2.  **Dual Basis (SFS):** Allow the solver to use the dual basis ($h_n^{(2)}$ and $j_n$) to perform active field separation across the entire frequency range.

### The Case for SFS in the RFT Range
Utilizing the dual basis in the RFT zone is significantly more robust for three key reasons:

1.  **Environmental Noise as an Incoming Field:** Even if the RFT zone is reflection-free, it contains environmental noise (HVAC, fans, traffic). Physically, this is a converging field. If you force the solver to fit this using only outgoing terms, you introduce fit errors.
2.  **The Propagation:** If an incoming field is incorrectly fitted using outgoing terms, the resulting model is only valid at the exact measurement radius. Outgoing Hankel functions decay as they move away from the origin—the opposite of a physical incoming field. When you propagate the field outwards (e.g., from 0.2m to 2.0m), the SPL response may "blow up" non-physically.
3.  **The Transition:** If SFS was not used above the RFT range, a merge would be required between the dual and single basis data. Ensuring such a transition is always accurate is a significant challenge.

## Conclusion: Robustness Through Field Separation

To ensure a stable, coherent reconstruction, it is most robust to fit the full frequency range using the Dual Basis ($h_n^{(2)}$ and $j_n$). In real-world applications, this means your External Field will not drop to theoretical zero in the RFT range; instead, it will sit at a realistic baseline representing the ambient background noise floor. This provides a more accurate, physically grounded internal source field that remains stable across all propagation distances.