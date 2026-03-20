# Understanding Acoustic Origins: Aligning Math with Physical Reality

In **Spherical Harmonic Expansion (SHE)**, modelling a sound field in 3D space relies on a fundamental assumption: the mathematical centre of the coordinate system needs to match the actual source of the sound.

However, a loudspeaker's acoustic origin is not static; it shifts depending on the frequency being played. 
* At **10 kHz**, the sound originates from the tweeter. 
* At **100 Hz**, it originates from the woofer or a bass reflex port.

If we don't dynamically track and align our math with this moving acoustic origin, our calculations become unnecessarily complex, mathematically unstable, and highly susceptible to noise. This is called **“ill-conditioning”**.

---

## Spherical Harmonics and Order N

To understand why the origin matters, we first need to understand how Spherical Harmonic Expansion works. You may know of Fourier, who proved that a complex audio signal can be described as a sum of its component parts—simple sine waves. 

Spherical Harmonics can be thought of as a 3D version of a Fourier transform, breaking a complex 3D shape (the sound field) into a series of standardized 3D shapes (spherical harmonics, or the ‘harmonic basis’). It is like using various standardized Lego bricks to build up the complex shape of a Batman figure.

The complexity of these shapes (like simple or complex bricks) is defined by the **Harmonic Degree, or Order (N)**:

* **N=0 (Monopole):** A perfectly uniform, pulsating sphere. This is the simplest possible sound source.
* **N=1 (Dipole):** A figure-eight shape, with sound pushing forward and pulling backward.
* **N>=2 (Quadrupole and beyond):** Increasingly complex shapes with multiple lobes that describe tight, angular details in the sound field.

### Degrees of Freedom
The Lego brick analogy extends to the degrees of freedom (DOF) of each harmonic degree. A simple order is like a square brick, in that it is the same at every rotation. A complex brick like an L-piece has more DOF—it is different when rotated, flipped, or mirrored. Higher harmonic degrees also have more DOFs, so the complexity they can describe scales quickly.

In this respect, **Order N represents our spatial resolution.** Higher orders allow us to model finer, sharper details in the sound field's directivity.

---

## The Origin Problem: The "Explosion" of Complexity

Imagine a perfectly spherical, pulsating balloon (an N=0 monopole). If we place the center of this balloon exactly at the center of our mathematical universe (0,0,0), the math is incredibly simple. The SHE solver only needs a single coefficient (a single radius) at N=0 to perfectly describe its surface.

Now, imagine we move that same balloon 1 meter to the left, but we keep our mathematical origin at (0,0,0). 

From the perspective of our mathematical origin, the balloon no longer looks like a simple, single radius. To describe this shifted balloon using math anchored at (0,0,0), the solver has to account for the near surface, the far surface, and everything in between. It is forced to use a massive combination of N=1, N=2, N=3, and so on, stacking complex "lobes" and "dents" on top of each other just to describe a simple balloon from a distant perspective.

### The Consequences of Mismatched Origins
When the acoustic origin is far from the mathematical origin, the required Order N explodes. This causes two massive problems for separating the internal speaker source from external room reflections:

1.  **Noise Amplification:** Higher-order spherical harmonics are incredibly sensitive to tiny variations. If the solver is forced to use high orders just to map an off-center source, it will amplify background noise or slight measurement errors in the microphone positioning and data.
2.  **Numerical Instability (Singularities):** The math required to calculate high-order wave propagation becomes fundamentally unstable. The matrices we use to solve the sound field lose their simplicity and become overly dependent on each other (a form of ill-conditioning called "linear dependence"), leading to mathematical singularities where the simulation falls apart.

---

## The "Phantom" Origin and Phase Shifts

To further understand why a dynamic 3D origin search is necessary, it helps to look at how different sound sources on a loudspeaker interact. The acoustic origin isn't always a physical component you can touch; it is often a "phantom" point in space created by the summation of multiple sources or the physics of air motion where velocity and pressure are not in phase.

* **The Crossover Handover (The Mid-way Point):** When a loudspeaker transitions from the tweeter down to the woofer, the acoustic center doesn't simply jump from one to the other. Through the crossover region, where both drivers are producing the exact same frequencies at similar amplitudes, the mathematical "center" of the combined sound field will resolve to a phantom mid-way point suspended between the two drivers.
* **Rapid Shifting and Diffraction:** At the specific frequency where a bass reflex port is tuned, the port takes over as the dominant radiator, causing the origin to shift rapidly toward the opening. Furthermore, when sound waves hit the cabinet edges, those edges act as secondary radiators (diffraction). Cabinet panels may vibrate as sympathetic resonators creating further sound sources. These interfere with the direct sound, pulling the calculated acoustic origin in unexpected directions.

Because of these complex interactions, it is impossible to manually pick a single coordinate that works for the whole speaker, or even one per driver.

---

## Optimization Strategies: The Simplex vs. The Brute Force Grid

To map this shifting path and keep the required Order N as low as physically possible, the script provides two distinct methods for finding the "global minimum" error in 3D space.

### 1. The 3D Simplex (The "Amoeba" Descent)
The primary search tool is the **Nelder-Mead Simplex** algorithm.
* **How it works:** It creates a small 3D tetrahedron (the simplex) in space and walks it through the error landscape by flipping, stretching, or shrinking the shape.
* **Strengths:** It is incredibly fast and efficient once it is in the correct neighborhood of the goal.
* **The Weakness:** It is a local optimizer. If it is dropped into a local valley (a false minimum), it will get stuck there and never find the true acoustic origin.

### 2. The Volumetric Grid Scan (The "Brute Force" Map)
For cases where the sound field is particularly chaotic, or simply to visualize the acoustic landscape, the script can perform a full volumetric grid scan.
* **How it works:** It divides the 3D search area into a grid and calculates the SHE residual error at every single point (to a limited order_N, e.g., N=6).
* **Strengths:** It provides a "true" map of the error landscape, ensuring the region of the global minimum is found regardless of where you start.
* **The "Cost":** This method is computationally massive. While a Simplex search might require 50 solves per frequency, a full 3D landscape rendering for a single frequency might require 200,000. A single frequency full scan can take 5+ minutes. Running this across the entire spectrum could easily exceed hours of CPU time.

---

## Seeded Intelligence: Why Human Input is Critical

Because the brute force method is too slow for everyday use, the script is designed to use human intelligence as the bridge. The optimization requires the user to provide the physical coordinates of the high-frequency driver as a starting point. This tends to have the smallest ‘acoustic origin’ footprint and is therefore most difficult to locate in a large volume of space without a hint. Dropping the Simplex close to the HF driver bypasses the need for a costly brute-force scan.

### Parallel Dual-Path Search

#### 1. High-to-Low Search (HF-to-LF)
* **Initial Lock:** The algorithm begins at the highest frequencies, using the user-provided tweeter coordinates to lock onto the origin where wavelengths are smallest and most sensitive.
* **The History Seed:** The script uses the origin coordinates from the previous frequency to seed the next.
* **Path Tracking:** As the frequency drops and the origin shifts, the simplex follows the acoustic path downward.

#### 2. Low-to-High Search (LF-to-HF)
* **Initial Lock:** The algorithm begins at the lowest frequencies, where large wavelengths create a broad, stable gradient that is easy to "lock on" to.
* **The History Seed:** Similar to the HF search, each successful solve seeds the next step upward.
* **Path Tracking:** As the frequency increases, the simplex tracks the shifting origin upward toward the crossover region and tweeter.

While each individual search branch is inherently sequential, they are independent of each other. It is therefore possible to process both in parallel on their own CPU cores, adding no extra time penalty compared to a single search. 

Since each branch approaches the acoustic centre from a different starting position, they provide a "second opinion" on the origin's location. One search can occasionally find a lower error origin than the other.

Once both branches complete, the script compares the results and selects the most accurate origin point from either path.

These best-fit results are then interpolated across the full frequency resolution of your input data set for a seamless final output.


### When to use the Grid Scan?
The brute force grid scan is best reserved as a last resort for erroneous results or as a diagnostic tool. If the High-Frequency and Low-Frequency search branches fail to "meet in the middle" or provide results that do not seem to match the reality of the DUT, running a grid scan on the problematic frequencies can help determine if the simplex is getting stuck in a local valley.

### A Note on Reported Results
Because the origin search is performed using a limited spatial resolution (typically Order N=6) to maintain high solve speeds, the error percentages reported during this stage may appear higher than those of the final, high-order SHE reconstruction. You should expect to see a reasonably consistent error curve that rises slightly during complex spatial transitions—such as the crossover region—and generally trends upward toward the high frequencies.