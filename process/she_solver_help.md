# Understanding the SHE Solver: Physics, Limits, and Linear Algebra

This stage of the pipeline involves taking the windowed, smoothed and origin-aligned audio data in the complex frequency domain and solving for the **Spherical Harmonic (SH) coefficients**. This is the mathematical engine of the project. It translates discrete microphone measurements into a continuous 3D sound field and separates the loudspeaker's direct sound from the room's acoustic reflections.

To use this solver effectively, it is helpful to understand how to balance its settings, why certain limits are enforced, and how the underlying processing choices impact your final results.

---

### 1. Matrix Condition vs. Fit Residual Error
When evaluating the quality of a solve, the user can monitor two metrics that are displayed as the solve progresses and as summary plots at completion.

* **Fit Residual Error (%):** This is the difference between your raw microphone data and the sound field rebuilt by the solver. A lower percentage means the solver's mathematical model closely matches reality.
* **Matrix Condition Number:** This represents the stability of the linear algebra. A well-conditioned matrix (low number) means the data is over-determined and robust. An ill-conditioned matrix (high number) means the math is fragile.

**The Balancing Act:** It is very easy to drive the Fit Residual Error down to near zero simply by cranking the **Order N** higher. However, doing so rapidly destroys the Matrix Condition. The solver begins applying massive, non-physical values to the coefficients. This occurs if the solver attempts to fit components not supported by the level of spatial sampling (noise or aliasing). You must balance Order N to achieve the lowest residual error while keeping the matrix well-conditioned, indicating the math is describing real, physical coherent sound fields.

### 2. Establishing the Order N Ceiling
Before the solver can run, it requires a maximum Order N to act as a hard cap on what spatial detail the input data can support. As established in the Stage 3 Optimizer process, this **"Tipping Point"** is identified specifically by looking at the RFT range for the point where the internal-to-external fields energy ratio is at its peak before declining due to ill-conditioning. Once this stable ceiling is established, we still need to decide how the Stage 4 solver "Climbs" to that target: the rate at which order N grows with frequency.

### 3. The Rate of Growth: $kr$ vs. Source Complexity
Determining how rapidly $N$ increases is a balance between the physical size of the grid and the actual complexity of the loudspeaker radiation.

**The Grid Size Demand ($kr + x$)**
Spatial complexity grows faster with frequency for larger measurement grids. If we compare two spheres—one small and one large—a fixed angle range projected from the centre (for example 10 degrees) corresponds to a greater physical arc-distance along the surface of the larger sphere.

This increased distance allows for greater phase variation in the sound on the surface of the larger sphere compared with the smaller one—enabling tighter lobes and more complex spatial structure per degree. This is the meaning of angular spatial detail. This behaviour is governed by the wavenumber $k$ and the grid radius $r$:

$$kr = \frac{2\pi r}{\lambda}$$

The industry standard for a stable, high-resolution solve is $kr + 2$.

**The Internal Source vs. External Field**
It is critical to distinguish between the complexity of the **Internal Source** (the speaker) and the **External Field** (the room):

* **The External Field:** Reflections arrive from all angles across the entire sampled volume, requiring high spatial resolution ($N$) to map correctly. A larger measurement grid will contain a more complex ‘room field’.
* **The Internal Source:** The speaker's complexity on the other hand is not tied to the measurement grid size. It is defined by the source's physical size relative to the wavelength. Driver size, baffle dimensions, horns, and cabinet edges all contribute. In fact, as the distance from the source increases, the complexity typically drops because we move toward far-field plane-wave conditions at high frequencies.

In many cases, the internal source is spatially "simpler" than the room field, especially as the grid gets larger. While the grid radius grows (increasing $kr$ demand), the speaker remains a constant size. It is therefore optimal to keep measurement grids as tight to the DUT dimensions as practically possible to minimize this complexity gap.

It is this internal to external complexity relationship that makes it difficult to define an ideal rate of climb for order N that applies to all measurement grids and DUTs. Be reassured that in practice, provided the maximum supported order N is not exceeded, the rate of climb vs. frequency primarily serves as a fine-tuning parameter, rather than a factor that fundamentally determines the quality of the result.

### 4. User Controls: Adjusting the "Climb"
The solver provides two primary ways to adjust the growth rate of Order $N$ to prevent over-fitting simple sources while maintaining room rejection.

**The $kr + x$ Offset**
The user can adjust the "Stability Offset" (the $+2$ in $kr+2$).

* **Higher Offset (e.g., $+2$):** Pushes the Order $N$ growth higher earlier in the frequency range. This is the default and prioritises room field separation. The cost is possibly over-fitting of the simple internal sources leading to artifice in the response.
* **Lower Offset (e.g., $+0$):** Damps the growth rate, protecting simple internal sources from being over-fitted by high-order harmonics too early. This may occur if the measurement grid is large or the source spatially simple. The cost may be reduced room field separation.

**The Manual Order Table**
For advanced control, a manual frequency-to-order mapping can be used.

### 5. Verification and Interpretation
To assess the quality of the solve beyond simple residual error and matrix condition metrics, users can study the quality of sound field separation by exporting pressures for both internal and external fields:

* **External Field above RFT Range:** Above the RFT, the external field should be near the environment noise floor (typically -25dB to -30dB) as the reflections should have been removed by windowing and only noise remains.
* **External field below RFT Range:** There are room modes and reflections in this range. A higher SPL here suggests the solver was able to separate more of the room field, but only provided the following:
* **Internal Field:** The internal field response appears anechoic, smooth and physically plausible.
* **Watch out for:** Peaks or dips that show up in both external and internal responses. This suggests ill-conditioning where the solver incorrectly assigned energy to both fields to make the total fit error lower at the measurement surface.

### 6. Linear vs Logarithmic Frequency Resolution
Many acoustic analysis tools downsample high-resolution FFT data, that is intrinsically on a linear frequency axis, into logarithmic bins (like 1/6th octave), both for visual reasons and to save CPU time in processing. This solver, however, computes the full linear FFT bins across the entire spectrum.

**The Advantages of Linear Processing:**
* **Phase Coherence:** Solving every linear bin preserves the continuous phase relationship of the data originating from the FFT process.
* **Time-Domain Reversibility:** Because we maintain the full linear complex frequency response, we can easily perform an Inverse FFT (IFFT) on the separated outgoing field to generate a clean, reflection-free Impulse Response (IR). If we solved logarithmically, the results would need to be interpolated from log back to a linear scale—this back and forth of dense-to-sparse frequency grids is a lossy process introducing error.

**The HF Cap:**
While we process linear bins, the solver implements an optional High-Frequency (HF) cap. If you input a 96 kHz sample rate measurement, there is no value in solving thousands of ultrasonic frequency bins. At those ultra-short wavelengths, the physical measurement grid is far too sparse to support field separation. Capping the maximum frequency saves CPU time. A modern multi-core processor is able to process the dense linear solve in a reasonable timeframe – expect from 5 minutes to 30 minutes depending on the maximum length of window (and thus FFT) used in the stage 1 FDW processing.

### 7. The Physics and the Mathematics
**Mixed Dual Basis: Separating Wave Fields**
While it is common to describe Sound Field Separation (SFS) using "time of arrival" intuition—where sound from an internal source hits the inner measurement grid before the outer grid, and vice versa for an external source—this intuition can be misleading. Timing and phase changes over small radial differences (e.g., 50 mm) are microscopic at low frequencies. If SFS relied solely on timing, performance would degrade at low frequencies; in practice, the opposite is true.

Instead of "timing" the sound, the solver identifies radiation characteristics and the specific shape of the pressure decay across the measurement volume.

**The Internal Source (The Loudspeaker)**
* **Radial Decay:** Sound from an internal source decays as it moves outward, away from the origin.
* **Net Energy Flow:** There is a net positive flow of energy leaving the measurement surface.

**The External Source (The Room)**
* **Radial Decay:** The external source field decays as it moves inward toward the origin, but then exits through the opposite side.
* **The "Standing" Component:** Because the inward-moving energy is matched by the outward-moving "exit" energy, they superimpose. On the measurement surface, these components are balanced.
* **Net Zero Energy Flow:** There is no total energy entering or exiting the measurement volume for the room field; it is oscillating rather than radiating.

By measuring at multiple radii, the solver identifies whether the shape of the pressure decay matches a radiating source or a non-radiating, balanced room field. The solver employs a dual basis to represent each radiation characteristic:
* **A Radiating Field (Internal Source):** Sound traveling outward from the source, represented by Spherical Hankel functions of the second kind ($h_n^{(2)}$).
* **A Non-Radiating Field (External Source):** Represented by Spherical Bessel functions of the first kind ($j_n$).

**Addressing Discrepancies with Academic Texts**
In many physics texts, spherical harmonics and sound field separation are represented by incoming ($h_n^{(1)}$) and outgoing ($h_n^{(2)}$) Hankel functions. However, as established, the external room field is not purely incoming. In a "pure Hankel basis," the physically relevant room field is a combination of $h_n^{(1)}$ and $h_n^{(2)}$. For interior acoustics, the Spherical Bessel function ($j_n$) is more physically intuitive because it is regular (finite) at the origin and represents both parts of the external source as it travels through the measurement volume.

**The FFT Time Convention and $h_n^{(2)}$**
In academic literature, harmonic wave propagation is often defined using the $e^{-i\omega t}$ time convention, which pairs with Hankel functions of the first kind ($h_n^{(1)}$) to describe outgoing waves. However, DSP and FFT algorithms typically utilize the $e^{+j\omega t}$ convention.

To avoid cluttering the code with multiple complex conjugations when moving between the "audio world" and the "math world," this solver uses the $e^{+j\omega t}$ convention consistent with the FFT pipeline. Consequently, **Spherical Hankel functions of the second kind ($h_n^{(2)}$)** are used to represent the outgoing wave field. While this may appear "backward" to a pure mathematician, it is the correct physical representation for this specific time convention.

### 8. The Case for SFS in the RFT Range
It might seem counter-intuitive to perform sound field separation on data that has already been windowed to remove reflections, but utilizing the dual basis in the RFT zone is significantly more robust for three key reasons:

1.  **Environmental Noise as an Incoming Field:** Even if the windowed RFT zone is 100% reflection free, it still contains environmental noise—HVAC systems, computer fans, or distant traffic. Physically, this noise is an external (converging) field. If you force the solver to fit this noise using only outgoing Hankel terms, you introduce fit errors because the mathematical model does not match the physical reality of the noise.
2.  **The Propagation:** If an incoming field is incorrectly fitted using outgoing terms, the resulting model is only valid at the exact measurement radius.
    * **The Mismatch:** Outgoing Hankel functions are designed to decay as they move outwards away from the origin. This is opposite of the physical incoming field, that decays as it propagates inward to the origin.
    * **The Result:** When you propagate the sound field outwards (e.g., from a 0.2m grid to a 2.0m observation point), the radiation characteristics applied to that "hidden" incoming noise become invalid. The coefficients will cause the pressure level to grow in a non-physical way, causing the SPL response to "blow up" at the evaluation point.
3.  **The Transition:** SFS is desirable below the RFT range; if it was not used above the RFT range, a merge would be required between the dual and single basis data. Ensuring such a transitional merge is always accurate presents a challenge.

### Conclusion: Robustness Through Field Separation
To ensure a stable coherent reconstruction, it is most robust to fit the full frequency range using the **Dual Basis ($h_n^{(2)}$ and $j_n$)**, performing sound field separation, even if the windowed RFT range appears extremely clean and high in SNR.

In real-world applications, this means your External Field will not drop to theoretical zero in the RFT range; instead, it will sit at a realistic baseline representing the ambient background noise floor of your environment. This provides a more accurate, physically grounded internal source field that remains stable across all propagation distances.
