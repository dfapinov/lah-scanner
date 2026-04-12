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

Since this period is intrinsically reflection-free, it defines our starting window length. Because this window is fixed in time, the number of wave cycles it contains changes with frequency. To maintain a target octave resolution, there is a **"Frequency Floor"**—the point below which a fixed RFT window simply doesn't have enough cycles to give you the detail you want.

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

---

## Why We Do This
These two steps—FDW and Complex Smoothing—are preparations for advanced processing like Spherical Harmonic Expansion. By minimizing the impact of room reflections up front and reducing the chaos of data (like comb filtering) through smoothing, we create more coherent data that the processing can more accurately model (fit).