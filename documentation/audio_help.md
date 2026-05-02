# Audio Measurement Engine: Hardware Setup and Signal Processing

The `audio.py` module serves as the acoustic measurement engine for the system. The job of this script is to drive the audio hardware and capture clean, accurate impulse response measurements. Understanding how the script's settings interact will help you optimize your measurements, protect your hardware, and ensure the best possible data for the **Spherical Harmonic Expansion (SHE)** solver.

---

## 1. Hardware Interface and Configuration

Before taking measurements, the system must properly interface with your audio hardware.

### Device Selection and ASIO
The script utilizes the `sounddevice` library with **ASIO** support enabled by default on Windows. ASIO is strongly recommended as it bypasses the operating system's audio mixer, providing bit-perfect, low-latency streams.

* **Configuration:** You must configure the correct input and output device indices and channel mappings in `config.ini`.
* **Dynamic IDs:** Because device IDs can dynamically change when hardware is added or removed (such as connecting a Bluetooth headset), it is highly recommended to use the provided `list_sound_devices.py` CLI script to confirm your IDs after any change.
* **Initialization Tip:** When an audio device is first initialized, the very beginning of the audio stream can sometimes be cut off by the interface driver. It is good practice to run at least one test sweep after starting the system to allow the hardware to lock onto the sample rate and protocol.

### The 48kHz Sweet Spot
While modern interfaces support extremely high sample rates, **48kHz** is generally the optimal setting for this system.
* It provides a comfortable buffer above the 20kHz audible band for the anti-aliasing filters to roll off smoothly.
* The measurement grids used in this system physically cannot support sound field separation at the short wavelengths of ultrasonic frequencies, meaning anything above ~24kHz will be truncated by the SHE solver regardless.

---

## 2. Measurement Sweeps and The Farina Method

The core of the measurement process relies on **Exponential Sine Sweeps (ESS)**.

### Separating Linear and Non-Linear Data
We use the **Farina sweep method** rather than periodic noise or standard spectral deconvolution because of its unique ability to push harmonic distortion products to negative time offsets, cleanly separating them from the main linear Impulse Response (IR).

* **The LTI Assumption:** The SHE solver is intrinsically a mathematical model for Linear Time-Invariant (LTI) systems. Harmonic distortion violates this physical assumption. If distortion is included in the IR, the solver will struggle to fit it, resulting in higher residual errors. By feeding the SHE solver only the separated linear portion of the IR, we maintain mathematical coherence.
* **Future Proofing:** The system saves the IR with full non-linear parts in the 'distortion' directory. By the full IR including distortion products, the system paves the way for potential future updates capable of extracting anechoic distortion data.

### Optimizing Sweep Parameters
* **Sweep Duration:** A sweep length of **1.5 to 2.0 seconds** is recommended. This is long enough to get useful data above the noise floor in the low-frequency range, but not so long as to cause excessive thermal buildup in drivers. 
* **FFT Length:** Note that longer sweeps do not increase the FFT length (unlike in traditional acoustic software), as the downstream Frequency Dependent Windowing (FDW) stage during post-processing will set the FFT length of the IR appropriately.
* **Averaging:** You can define multiple sweeps per measurement point. This is the best way to suppress non-coherent environmental noise (HVAC, traffic, etc.). The SHE solver cannot fit random noise, so lowering the noise floor directly improves the final fit error.
* **SNR vs. Safety:** While pushing higher SPLs improves SNR, always be mindful of your Device Under Test (DUT). Do not push drivers past their linear excursion or thermal limits.

---

## 3. Driver Protection and LF Correction

When measuring raw, un-crossovered drivers (particularly tweeters), low-frequency energy from the sweep can easily cause mechanical damage.

### The Protection HPF
You can apply a High-Pass Filter (HPF) directly to the playback signal. You can select the corner frequency, slope order (1st to 4th), and filter correction details.

### Inverse Correction and SNR Limits
The system can apply an inverse filter during IR generation to mathematically "undo" the HPF and reconstruct the driver's natural response. 

* **The Gain Limit:** Because the physical DUT receives an HPF-filtered signal, the acoustic output at low frequencies will dive into the room's noise floor. To avoid massively amplifying noise, the system applies a maximum gain limit (defaulting to **12dB**, user-adjustable). A 12dB boost provides exactly two additional octaves of fully corrected response for a 1st order (6dB/oct) filter, and one additional octave for a 2nd order (12dB/oct) filter.

![Driver Protection HPF](./driver%20protection%20hpf.png)

---

## 4. Synchronization and Latency

System latency (delay introduced by audio drivers and AD/DA converters) is actively compensated for using a physical loopback channel.

### Barker Code Alignment
Before every sweep, the loopback channel emits a band-limited **Barker-13 sequence**. Barker codes allow the audio engine to pinpoint the exact starting sample via cross-correlation.

### Alignment Strategies (`align_to_first_marker`)
* **True (Sample Cutting):** The system locks onto the very first marker in a set and relies on strict sample counting for subsequent sweeps. This introduces fewer potential points of misalignment if your interface has stable latency.
* **False (Per-Sweep Alignment):** The system re-syncs to the Barker marker at the beginning of every individual sweep in the average. This provides robustness against minor clock drift or driver glitches.

*Note: Ideally both methods should produce the same results, but in case of issues with sweep averaging that only present ‘in the field’ this setting is retained*

---

## 5. Deconvolution and Masking

During the mathematical derivation of the Impulse Response, the engine applies specific spectral masks to ensure a clean, oscillation-free IR without temporal smearing.

* **DC Kill:** A 5Hz Butterworth filter ensures the absolute DC bin (0Hz) contains zero energy, preventing baseline drift.
* **Anti-Aliasing HF Mask:** A high-frequency taper is applied near the Nyquist limit. 
    * **Logic:** If the sample rate is **44.1KHz**, the filter starts at exactly **20KHz**. If the sample rate is **48kHz or greater**, the filter starts at **22kHz** and ends exactly one bin before Nyquist to ensure absolute zero at the ceiling.

---

## 6. Automated DSP Verification and Quality Metrics

After deconvolution, the script analyzes the resulting IR to calculate quality metrics, logged in `_metrics.json`.

* **Signal-to-Noise Ratio (SNR):** EThe system estimates the environmental noise floor by analyzing the very end of the full, uncropped IR record (where the acoustic energy should have decayed). It compares this noise floor to the peak amplitude of the linear IR. A warning is logged if SNR < 30dB.
* **Total Harmonic Distortion (THD) Estimation:** Compares the energy of the "negative time" distortion products against the main linear impulse. A warning is logged if **THD > 10%**.
* **Peak Sharpness Ratio (PSR):** Evaluates the quality of the Barker code in the loopback channel by comparing the height of the primary correlation peak to the sidelobes. A low PSR indicates poor alignment. A warning is logged if PSR < 3.0
* **Crest Factor:** The ratio of the peak amplitude of the linear IR to its RMS energy. A low crest factor suggests temporal smearing or alignment failures. **This metric is only logged in _metrics.json**

---

## 7. Diagnostics and Testing

These features are intended for pipeline testing and verification:

* **Mock Interfaces:** The `mock_interface` mode simulates hardware latency, DAC FIR anti-aliasing filter group delay, and high-pass characteristic of DC blocking capacitors to test the alignment engine without physical hardware attached.
* **Harmonic Injection:** Injects artificial 2nd and 3rd order harmonics (`H2_TEST_DB`, `H3_TEST_DB`) to verify the Farina separation logic.
* **Debug Saves:** Saves raw microphone captures and loopback tracks to a debug folder for manual waveform inspection.