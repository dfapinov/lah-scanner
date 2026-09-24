# Stage 1 Help - FDW & Smoothing

Stage 1 turns your captured impulse response WAV files into a single complex frequency-domain dataset for the rest of the HALS processing pipeline. In the app this is the **Stage 1: FDW & Smoothing** tab.

This help file is split into two parts:

- **Usage Guide:** what to put where in the app, what each setting does, and what to check after running.
- **Understand the Processing:** the original explanation of FDW, RFT, and complex smoothing.

---

# Usage Guide

## What Stage 1 Does

Stage 1:

- finds the direct sound arrival in each impulse response
- applies Frequency Dependent Windowing (FDW)
- applies complex smoothing
- saves the processed complex pressure data for later stages
- opens the interactive FDW viewer when processing finishes to view the results

The output from Stage 1 is used by Stage 2, Stage 3 and Stage 4.

## Why We Do This
FDW and Complex Smoothing are preparations for advanced processing like Spherical Harmonic Expansion. By minimizing the impact of room reflections up front and reducing the chaos of the data (like comb filtering) through smoothing, we create more coherent data that the processing can more accurately model (fit).

## Input And Output Files

Stage 1 expects the current project folder to contain one of these folders:

- `project_folder/measurement_set`
- `project_folder/recordings`

The WAV filenames should include the measurement position using the HALS coordinate convention:

```text
anything_r<radius_mm>_ph<azimuth_deg>_z<height_mm>.wav
```

For example:

```text
point001_r300_ph-45p5_z120.wav
```

The app uses these filenames later to recover each microphone position. The `p` character is used as the decimal point in coordinate values, so `ph-45p5` means -45.5 degrees. Files captured by the HALS Capture app should already follow this convention.

When you click **Run Stage 1**, the app creates:

- output folder: `project_folder/outputs`
- main output file: `project_name_complex_data.npz`

## Quick Start

For a normal first run:

1. Select an existing project folder created by the HALS Capture app.
2. Set **Reflection Free Time (ms)** from your measurement geometry.
3. Set **Octave Resolution (1/x)** for the frequency resolution you want.
4. Leave **Max Window Cap (ms)** at `200-400` for most datasets.
5. Leave **Enable Auto Gain** disabled if you use calibrated SPL from the HALS Capture app.
6. Leave **Enable Smoothing** and **Sliding HF smoothing** enabled, with **Smoothing Octave Res (1/x)** set to `Auto`.
7. Click **Run Stage 1**.

The progress of Stage 1 processing is displayed in the console pane. The viewer always opens once processing completes. Note: these results are only to inspect that the data looks reasonable. Unlike standard acoustic measurements, this is only one step of the HALS speaker measurement process.

## Main Settings

### Reflection Free Time (ms)

The clean time window before the first significant reflection reaches the microphone. It should be determined and calculated based on the nearest reflecting boundary to the HALS measurement system. Use the built-in calculator to help.

At high frequencies, the frequency dependent windowing uses this as the fixed window length.

Note that the Reflection Free Time and the Octave Resolution settings interact to set the reflection free frequency range.



Typical starting value:

- `5 ms` when the closest boundary is about 1m away.
- The closest boundary is usually the floor, so consider optimizing the height of the speaker to maximise distance from the floor and ceiling.

Achieving a longer valid reflection-free time puts less pressure on the sound field separation DSP, which is most effective at removing room influence at low and mid frequencies.

### Octave Resolution (1/x)

The target FDW resolution, expressed as a fraction of an octave.

Examples:

- `3` = 1/3 octave
- `6` = 1/6 octave
- `12` = 1/12 octave
- `24` = 1/24 octave

Higher octave-resolution values need more cycles per frequency, so for the same RFT they push the frequency floor higher.

### Max Window Cap (ms)

The maximum allowed FDW window length at low frequencies.

FDW naturally wants very long windows at low frequencies to maintain the target octave resolution, but this will dramatically increase processing time in Stage 4. This cap prevents the low-frequency window from becoming extremely long.

The default `200-400 ms` is a practical starting point.

### Enable Auto Gain

Applies one shared gain value across the full batch so the loudest detected peak reaches **Target Peak (dB)**.

This keeps relative levels between measurement positions intact because every file receives the same gain change.

Leave it off if you want to preserve the recorded WAV levels and apply the SPL calibration that can be set up in the HALS Capture app. Enable it if you want a consistent global peak level in the processed dataset.

### Target Peak (dB)

The target peak level used when **Enable Auto Gain** is on.

The value is in dBFS. For example, `-3.0` means the loudest peak in the batch is scaled to -3 dBFS.

### Enable Smoothing

Turns complex frequency-domain smoothing on or off. Enabled by default.

For normal HALS processing, leave this enabled. Turn it off when you want to inspect or compare the raw FDW result. With smoothing disabled, **Sliding HF smoothing** has no effect.

### Smoothing Octave Res (1/x)

The base resolution of the complex smoothing pass.

Stage 1 has two smoothing effects. FDW naturally smooths the response through time windowing. Complex smoothing is a separate pass that averages the complex response, affecting both magnitude and phase.

`Auto` means the app uses twice the FDW octave-resolution denominator. For example, if **Octave Resolution (1/x)** is `12`, the base complex smoothing is `24` (1/24 octave). This adds a light touch of complex smoothing where windowing already provides most of the smoothing.

### Sliding HF smoothing

Always used when complex smoothing is enabled. Below the fixed-window RFT transition, complex smoothing keeps the base bandwidth. Above the transition, its bandwidth gradually increases toward the selected FDW octave bandwidth as frequency increases.

For example, with FDW set to `12` and smoothing set to `Auto`, complex smoothing stays at 1/24 octave in the FDW range and widens toward 1/12 octave in the high-frequency fixed-window range.

The adjustment follows the estimated number of cycles within the fixed window, using the same cycles-to-resolution relationship as FDW. There is no separate eased-onset frequency band. As the fixed window contributes less smoothing in octave terms, complex smoothing progressively picks up the difference.

Sliding smoothing is always used when complex smoothing is enabled. A manually selected base denominator must be at least the FDW denominator (for example, 24 for FDW 12). Older project settings cannot select fixed smoothing.


## Advanced Settings

Click **Show Advanced Settings** to reveal these controls.

### Alpha HF / LF

Alpha controls how abruptly the time window closes. A lower value gives a more rectangular window, which keeps more of the available reflection-free time but can increase spectral leakage and ringing. A higher value gives a smoother taper, which reduces leakage but also softens the effective time cutoff.

The Alpha value can be defined for HF and LF with the value graduated between.
- `0.0` = rectangular
- `1.0` = Hann


### Windows per Octave

Controls how many FDW analysis windows are generated per octave.

The default `3` gives smooth transitions without excessive processing cost. Increasing it can make the window schedule more finely sampled, but processing takes longer.

Each window is smoothly crossfaded in the complex domain, so the result does not stair step from one window length to the next regardless of the number of windows used.

### Peak Detect Threshold (dB)

Controls how Stage 1 chooses the direct sound peak.

The detector finds the loudest peak, then searches earlier in the impulse response for significant peaks above this threshold. This helps because while the loudest peak is usually the direct sound arrival, it is sometimes a reflection, which places the window in the wrong place.

The default `-12.0 dB` means an earlier peak can be accepted if it is within 12 dB of the loudest peak.

The FDW results viewer shows the IR waveform and the detected peak for user inspection.

If the window aligns too late, try a more negative value such as `-18`. If it locks onto noise before the true impulse, try a stricter value such as `-6`.

### Debug / Inspection

These controls are useful when checking or comparing Stage 1 behavior. For normal processing, the defaults are usually appropriate.

### Keep Raw & Smoothed

When off, Stage 1 saves the final selected result to the main NPZ file.

When on, Stage 1 saves both:

- raw FDW data: `project_name_complex_data.npz`
- smoothed FDW data: `project_name_complex_data_smoothed.npz`

Use this when comparing settings for debugging issues. Leave it off for a normal workflow.

---

# Understanding FDW: Windowing as Smoothing

In standard gated measurements, we use a fixed-length window (e.g., 5ms). However, the relationship between that window length ($T$) and our frequency resolution is governed by how many wave cycles ($m$) can actually fit inside it.

Because low frequencies have long wavelengths, a fixed window captures fewer cycles as the frequency drops. This makes a standard window a "frequency-variant" process: you get high resolution at the top end, but very little at the bottom.

![fixed window changing frequency](./docu_images/fixed%20window%20changing%20frequency.png)

To achieve a specific octave resolution, the window must capture a minimum number of cycles for any given frequency ($f$):



$$T(f) = \frac{m}{f}$$



For example, at 1KHz to achieve 1/3rd octave resolution requires 4 cycles. To achieve 1/12 octave resolution required 17 cycles.

![fixed frequency changnig window](./docu_images/fixed%20frequency%20changnig%20window.png)

### The Reflections vs. Resolution Trade-off

* **A Long Window:** Captures more cycles per frequency. This provides high resolution and better detail but risks letting in room reflections.
* **A Short Window:** Captures fewer cycles. This keeps the measurement "clean" of reflections but results in a smoothed, lower-resolution plot.

### The FDW Approach
Most acoustic analysis benefits from constant octave resolution (e.g., 1/6th octave) across the entire spectrum. Instead of using a fixed time window that provides inconsistent resolution, FDW varies the window length based on the frequency being measured.

By keeping the number of cycles ($m$) constant, the window automatically shrinks at high frequencies to stay tight and expands at low frequencies to capture enough wave cycles. This allows us to exclude as many reflections as possible while maintaining a consistent resolution across the entire sweep.

![fdw infographic](./docu_images/fdw%20infographic.png)

---

## Practicality

In a perfect mathematical world, we would calculate a unique window length for every single frequency bin in the FFT. However, this brute-force method is not only CPU intensive, but it also introduces phase discontinuities. If each frequency is treated in total isolation, the transitions between them can become "choppy," ruining the phase data we rely on.

### The Solution: Multi-Window Interpolation
To solve this, the practical approach uses a series of overlapping windows rather than an infinite number of unique ones.

* **A Bank of Windows:** The software generates a number of windows (e.g., 3 per octave, from long to short).
* **Complex Data Processing:** It calculates the complex data (both the Real and Imaginary parts, which represent Magnitude and Phase) for these windows.
* **Interpolation:** The software then interpolates across these windows.

  This produces a smoothly varying mix, where the transition from one window length to the next is seamless. The result is a measurement with near-constant octave resolution that maintains its phase coherence, giving you an optimally windowed look at the DUT response with minimal reflections.

  ![fdw screenshot](./docu_images/fdw%20screenshot.png)

---

## THE RFT CONSTRAINT AND THE FREQUENCY FLOOR

The **Reflection-Free Time (RFT)** is the window of "clean" data available in your specific physical setup. It is the time gap between the direct sound arriving at the microphone and the very first reflection (usually from the floor or a nearby wall) hitting the capsule.

Since this period is intrinsically reflection-free, it defines our starting window length. Because this window is fixed in time, the number of wave cycles it contains changes with frequency. To maintain a target octave resolution, there is a **"Frequency Floor"**â€”the point below which a fixed RFT window simply doesn't have enough cycles to give you the detail you want.

To determine the Reflection-Free Time (RFT), you need to calculate the difference between the direct sound path and the shortest reflected path (usually the floor). If your speaker and microphone are at the same height ($h$) and separated by a distance ($d$), the reflected sound travels a longer, triangular path. By calculating this distance difference and dividing it by the speed of sound ($c \approx 343\text{ m/s}$), you find the "time window" available before the first reflection corrupts your measurement.

### The RFT Formula

To calculate the **Reflection-Free Time (RFT)** in milliseconds:

$$\text{RFT (ms)} = \left( \frac{\sqrt{D^2 + 4D_r^2} - D}{343} \right) \times 1000$$

**Where:**
* **$D$** = Distance in meters between the speaker (DUT) and the microphone.
* **$D_r$** = Distance in meters from the speaker/mic to the reflecting boundary.
* **$343$** = Speed of sound in m/s (approximate for room temperature).

![RFT calc](./docu_images/RFT%20calc.png)

(Image of calculator in VituixCAD)

### Examples:
1) 5ms RFT, 1/3 target Oct Res = 4 cycles needed = 760Hz frequency floor.
2) 5ms RFT, 1/12 target Oct res = 17 cycles needed = 3300Hz frequency floor.
3) 10ms RFT, 1/12 target Oct res = 17 cycles needed = 1650Hz frequency floor.

Above this Frequency Floor, the fixed RFT window is actually longer than necessary for the target resolution, so we keep it as is. Only below this RFT window do we begin to apply the FDW logic of expanding the window to maintain resolution. At this point, we are making a conscious trade-off: we accept a small amount of reflection data in exchange for maintaining the target octave resolution at lower frequencies.

---

## Understanding Complex Smoothing: Magnitude vs. Phase

### Why "Standard" Smoothing Fails
Typical smoothing in acoustic software is magnitude-only. The software simply averages the peaks and dips of the SPL curve. While this makes a graph look pretty and easier to read, it is intended for visual purposes only.

You cannot use magnitude-only smoothing for crossover design calculations or signal processing. This is because magnitude and phase are inextricably linked (in what engineers call a "Minimum Phase" relationship). When the magnitude changes, the phase must change accordingly. If you smooth the magnitude but leave the phase raw (or smooth it independently), that crucial relationship is broken, and your simulations will be inaccurate.

### The Solution: Complex Frequency Domain Smoothing
To keep the measurement representing physical reality, we must use Complex Frequency Domain Smoothing.

Think of the Complex Domain as a way to describe the full signal where phase and magnitude are joined at the hip. In technical terms, this data is represented by Real and Imaginary parts. (I created a fun interactive script to help understand complex frequency domain [Complex Visualizer](https://github.com/dfapinov/lah-scanner/blob/66cfdf15742e616362d79e6079e99741fabb2889/process/complex_visualizer.py))

By applying smoothing to the Real and Imaginary parts simultaneously, we smooth the overall response while maintaining the physical relationship between magnitude and phase. 

---

## How Windowing and Complex Smoothing Share the Work

The primary smoothing mechanism changes across the frequency range:

- **FDW range:** the window length follows frequency to maintain approximately the requested octave resolution. Windowing provides most of the smoothing, while the narrower complex-smoothing kernel adds a light touch.
- **Fixed-window RFT range:** the window length stays constant, so it contains more cycles as frequency rises. Its smoothing bandwidth becomes narrower in octave terms. Sliding complex smoothing gradually widens to compensate, approaching the requested FDW octave bandwidth at high frequencies.

For FDW 1/12 octave with `Auto` complex smoothing, the kernel therefore stays at 1/24 octave below the transition and approaches 1/12 octave above it. Its width follows the cycle-based estimate of the fixed window's remaining smoothing contribution, rather than the crossfade between FDW windows.

This is an approximate balance between two different smoothing mechanisms, not an exact combined octave resolution. In particular, FDW 1/12 plus complex smoothing 1/24 produces slightly stronger smoothing than FDW 1/12 alone.

## The Bonus Effect: Reflection Rejection
Beyond just making the data easier to process, Complex Smoothing provides a secondary, powerful benefit: it further cleans the measurement of room reflections.

### Coherence vs. Chaos
This works because of the fundamental difference between the direct sound and a reflection bouncing off a wall:

* **The Direct Sound:** Since it follows a "Minimum Phase" relationship with the frequency response, its phase changes smoothly across the frequency range. It is coherent.
* **The Reflection:** Because a reflection arrives at the microphone with a time delay, it introduces extremely steep, rapid phase shifts that vary wildly from one frequency to the next.

When we apply smoothing to the complex data (Real and Imaginary parts), we are essentially averaging the phase.

* **The coherent phase** of the direct sound is consistent across the smoothing window, so it remains strong and intact.
* **The chaotic phase** of the reflections varies so significantly by frequency that the peaks and valleys of those shifts average out to a small value.

**In short:** Complex smoothing acts as a filter that preserves the DUTs true signal while suppressing erratic interference.

