# Extracting Pressures: Evaluating the 3D Sound Field

The purpose of this script is to take the mathematical description of the 3D sound field (stored in the .h5 coefficients file) and translate it into actionable acoustic measurements. By utilizing wave propagation physics, the script can reconstruct the sound field and extract complex pressure data (magnitude and phase) at any coordinate in 3D space. Whether you need to evaluate the response 1 meter in front of the speaker or 100 meters above it, the underlying mathematical model allows for precise virtual microphone placement.

---

## Core Operations
Before generating measurements, the script must define what it is evaluating and where.

### 1. Field Selection
The **Spherical Harmonic Expansion (SHE)** solver separates the sound field into distinct mathematical components. You can choose which portion to evaluate:

* **Internal Source:** Extracts the purely anechoic data originating from the Device Under Test (DUT), stripped of room reflections.
* **External Source:** Extracts the room field and reflection data (typically used for analysis and inspection of solve quality rather than speaker evaluation).
* **Full:** Evaluates both fields combined. This effectively reconstructs the raw measurement in the room and is mostly used for debugging or verification.

### 2. Frame of Reference
To ensure your virtual measurements align with the physical reality of the speaker, you must define the correct frame of reference.

**Cartesian Offset (XYZ)**
This applies a physical shift to the virtual microphone relative to the measurement origin. For example, if your physical measurement grid was centered on the midrange driver of a 3-way speaker, but you want to perform an arc sweep rotating around the tweeter's acoustic center, you can apply a Z-axis (height) offset to raise the measurement origin accordingly.

**Spherical Orientation**
Spherical harmonic mathematics rely on a specific coordinate system where **Theta = 0** aims "straight up" vertically (the North Pole of the sphere). However, in acoustic measurements, a loudspeaker is typically facing "forwards" toward the equator, not pointing at the ceiling.

* **Zero Theta:** Setting this to 90 degrees rotates the primary measurement axis from the North Pole down to the equator, aligning measurement arcs sweeps with the front of the DUT.
* **Zero Phi:** Functions similarly, allowing you to rotate the frame of reference horizontally around the equator.

---

## Evaluation Modes
The script offers three primary methods for placing virtual microphones to extract data:

### Arc Sweep
This mode allows for automated polar measurements. You can define a radius (distance), an angular range (e.g., +/- 90 degrees), and an increment step (e.g., 10 degrees). The sweep can be performed horizontally, vertically, or both simultaneously to generate a comprehensive polar map.

### CTA-2034 Mode
This fully automates the evaluation of the 70 specific spatial coordinates required by the CTA-2034 standard (commonly known as the **"Spinorama"**). The script automatically extracts the complex pressures at these 70 points, calculates the standard curves (On-Axis, Listening Window, Early Reflections, Sound Power, Predicted In-Room Response (PIR), and Directivity Indices (DI)), and outputs the final standardized response data. To keep the output clean, the original 70 raw measurement files used for the calculation are discarded.

### Custom Coordinate Table
For specific, non-standard evaluations, you can input a custom list of spherical coordinates, and the script will extract the data exactly at those points.

---

## Exporting the Data
The script provides several ways to format and export the results.

### FRD Export
This exports standard Frequency Response Data (FRD) files containing magnitude and phase.

* **Linear Resolution:** Unlike typical FRD files from acoustic software that use logarithmically spaced frequency bins, this script exports the full linear frequency axis data from the original input set. This preserves maximum resolution in the mid and high frequencies and significantly reduces phase wrapping artifacts.
* **Subtract TOF:** You can choose to subtract the phase slope caused by the Time of Flight (TOF) propagation delay. This pure delay removal makes the phase response much easier to analyze visually. (Note: If more than one mic distance is used in the evaluation set, only the shortest TOF delay will be subtracted from all files equally to preserve relative phase differences).

### Impulse Response (IR) WAV
The extracted data can be used to generate impulse response WAV files by means of an Inverse Fast Fourier Transform (IFFT).

* These files always include the full TOF propagation delay, exactly like a physical impulse response measurement.
* They can be imported into acoustic viewing software, crossover simulators, or used directly in a convolution engine to auralize the measured system.
* **Sample Rates:** If the input data for the SHE was 44.1kHz, the IR generation will be 44.1kHz. If the input was 48kHz or higher, the IR export is capped at 48kHz, as the SHE processing is inherently limited to a 24kHz bandwidth.

---

## Microphone Calibration
If you utilized a measurement microphone that requires correction, a calibration file can be applied during extraction to improve the accuracy of the final data. This correction is applied to both the FRD and IR WAV exports.

### Complex Phase Derivation
Standard calibration files only provide magnitude adjustments. However, this script automatically derives the minimum-phase response that directly corresponds to those magnitude adjustments. This ensures that the system corrects not just the amplitude, but the phase of your measurements as well, maintaining magnitude and phase coherence.

* **Add vs. Subtract:** Depending on the source, some files contain the raw microphone response (requiring subtraction), while others contain the inverted correction curve itself (requiring addition). You can toggle the script to "add" or "subtract" to accommodate either format.

* **Fade-Out:** Microphone calibration files rarely cover the entire frequency spectrum. To prevent a sudden "step" or shelf in the measurement data where the calibration file ends, the script automatically applies a fade-out to the correction above and below the limits of the provided file. The bandwidth of this fade can be user-set by `MIC_CALIBRATION_FADE_OCTAVES`.


---

## Inverse Acoustic Origin Transform
In earlier processing stages, the system dynamically shifted the mathematical coordinate system to align with the optimal acoustic origin at every individual frequency. This was necessary to solve the Spherical Harmonic Expansion (SHE) with maximum accuracy.

Evaluating that optimized SHE data "as is" would result in a sound field that no longer aligns with the physical reality of the Device Under Test (DUT). The script notes the exact coordinate transforms applied during the earlier solve and performs the exact opposite shift at every frequency during extraction. 

This restores the true behavior of the DUT, ensuring that each frequency radiates from the correct physical location and the time delay and relative phase relationships are perfectly preserved.

> **Note:** There are no user settings for this function. It operates automatically in the background, but it is a critical mathematical step that curious users may wish to understand.