# Synthetic IR Generation (IR_Gen_piston)
Generates synthetic impulse responses (IRs) at specified coordinates from the input CSV.  
The results are physically accurate in time-of-flight, phase, and directivity, providing ideal test data for the pipeline.  
Because these IRs are free from room acoustics, the SHE solver may attempt very high harmonic orders unless a user limit is set.

---

## Measurement Grids (Grid Generator)
Creates a cylindrical measurement grid around the DUT (Device Under Test).  
A cylinder is used instead of a sphere because most loudspeakers are taller than they are wide.  
A spherical grid would push all points out to the same large radius, while a cylinder keeps the microphone closer (in the near field) more often.  
The aspherical SHE solver allows non-spherical or irregular shaped grids and patches to be utilised.

### Coordinate system (cylindrical)
- **r_xy_mm** – radial distance from the Z-axis  
- **phi_deg** – azimuth angle around the X-Y plane  
- **z_mm** – height along the Z-axis  

**CSV columns:**  
`r_xy_mm, phi_deg, z_mm, spiral, order_idx`

The cylinder is defined by its radius and height.  
The number of points on the caps (top and bottom) is set by `cap_fraction`.  
Adjust this as aspect ratio changes to keep cap density reasonable.  
Default settings (reverse spiral, rotation angle, etc.) minimize nearest-neighbour threshold warnings while keeping the point spacing irregular.  
The bottom “no-go” zone avoids microphone collisions with the support pole.

**Neighbour threshold guideline:**  
- ~8 mm for 20 kHz  
- ~7 mm for 24 kHz  

---

## Path Planner
The aim is to move the microphone up and down the walls of the cylinder in vertical lines, while the rotational arm advances in small steps around the perimeter.  
This approach is used because motion on the rotating axis is assumed to be more prone to backlash and inertia, making it slower.  
Vertical stripes of `DELTA_THETA_DEG` width define each sweep band.

---

## Audio Capture
Use **ASIO drivers** when available.  
Run `sweep_function.py` in the terminal to list audio devices.  
Channel defaults are usually fine (0 = L, 1 = R).  

**96 kHz** capture sample-rate is recommended. This provides clean phase data across the full solve range.  
At 48 kHz, filtering near the 24 kHz Nyquist limit can distort the phase and negatively affect the solve.  

`pre_sil` adds safety silence before the sweep.  
`post_sil` must be long enough for full latency capture (if the sweep cuts off early, increase this).  
Set `debug_saves = True` to save raw sweep recordings for inspection.  

If audio glitches occur, adjust the buffer size in your audio driver to match `BLOCKSIZE`, or reduce `BLOCKSIZE` to feed the buffer more frequently.

# Debug Output Files (`debug_saves = True`)

| File | Description |
|------|--------------|
| `excitation_signal.wav` | Original digital log sweep played through the loudspeaker channel.|
| `<base>_mic_raw.wav` | Unprocessed mic capture. | 
| `<base>_mic_aligned.wav` | Mic signal time-aligned to excitation, via loopback marker. |
| `<base>_mic_conditioned.wav` | Pre and post gated mic signal. |
| `<base>_loop_raw.wav` | Raw loopback signal (barker timeing marker. |
| `<base>_loop_aligned.wav` | Loopback signal time-aligned to excitation, via correlation filter. |
| `<base>_ir.wav` | Final gated impulse response generated from mic conditioned and exitation signal. |

---

## IR from Sweep (Make_IR_function)
Default settings usually work well.  
They minimize HF aliasing and ensure stable deconvolution.  
Set `ENABLE_GATE = False` to check the ungated IR for aliasing artifacts and optimisation of settings.  
Regularization acts like an adaptive filter that suppresses extreme values where SNR is low (outside the sweep band or near DC/Nyquist).  
Magnitude tapers are belt and braces – optional but harmless.

---

## Capture and Filename Convention
Each measured IR must follow this naming format so the solver can extract coordinates:  
`<orderID>_<r_xy_mm>_<phi_deg>_<z_mm>_ir.wav`  

Decimal points are replaced with “p”:  
Example → `id1_r253_ph4p7_z41_ir.wav`

---

## Crossover Frequency and Dual-Gate Strategy
The crossover frequency setting defines the short gate length for HF processing in stage 1 and merging of the HF / LF data in stage 4.  
Choose it so the short gate excludes the nearest reflection (floor, ceiling, etc.).  
Use `crossover_check.py` to find a safe frequency.

**Dual gate rationale** – Two gate lengths are used: a long gate for low frequencies and a short gate for high frequencies, and later their results are merged.  
At low frequencies, sound-field separation (SFS) is the most effective method for removing room effects, since the acoustic field changes slowly in space and can be accurately described with only a few spherical harmonics.  
These low-frequency impulse responses can even remain ungated, although applying a long gate is generally good practice.  

At higher frequencies, where wavelengths are much shorter, the sound field varies rapidly in space, and SFS becomes less reliable – small spatial errors or noise can corrupt the harmonic solve.  
In this range, simple time-gating works very effectively because reflections arrive well after the direct sound and can be cleanly removed.  

Both SHE and SFS are applied to the LF and HF datasets so they can be propagated together and merged smoothly, resulting in a final response that combines the low-frequency robustness of SFS with the clean, reflection-free precision of high-frequency time-windowing.

### Stop criteria for the harmonic solver
Selecting good stop criteria for the expansion is critical.  
If the solver stops at too low a harmonic order, important spatial information is lost.  
If it continues to too high an order, it starts fitting noise or redundant information, causing the coefficient matrix to become ill-conditioned (nearly linearly dependent).

- **KR-limit** – Theoretical limit based on the product `k r = 2πf r / c`.  
  It defines the maximum harmonic order that can be meaningfully resolved for a given frequency and measurement radius.  
  This limit is effective at low frequencies (where N is small) but grows quickly with frequency.  
  In practice, combining with a hard cap such as `N = 20` works well for typical loudspeakers.  
- **Condition-number monitoring** – A dynamic and practical method to detect instability.  
  Watch for a sudden rise in the matrix condition number; when this occurs, stop increasing N.  
  A hard upper limit (e.g., `N = 20`) can still be applied as a safeguard.  
- **Grid-density guardrail** – The grid geometry itself limits how many harmonics can be resolved.  
  The number of available measurement points sets an upper bound on N.  
- **Residual percentage** – The residual error between reconstructed and measured pressures can help track convergence but does not reliably indicate when instability begins.

---

## FRD Extraction
Cylindrical coordinates are used for the measurement grid and IR naming because they match the physical robotic system.  
Internally however, the SHE solver operates in spherical coordinates.  
When `USE_COORD_LIST = True`, the observation coordinates must also be spherical.  

FRD files wrap phase values at 360°.  
If the phase changes by more than 360° within a single frequency step, the wrap fails and the data becomes invalid.  
Large time-of-flight (TOF) delays create a steep phase slope, and at high frequencies on a logarithmic frequency grid, this makes phase wrap errors very likely.  

To prevent phase wrap errors in FRD exports, the TOF delay can be subtracted from the data before export:
- If all points are the same distance, this is fine.  
- If distances vary, only the smallest TOF is subtracted to preserve relative phase (some wrapping may remain).  

**FRD naming convention (VituixCAD/ARTA style):**  
`response_<plane><±angle>_<distance_mm>.frd`  
“response” is a pre-fixed name that can be changed in `config_process.py`.  

Examples:  
response_hor+10_2000mm.frd
response_ver-35_1000mm.frd

The script can export:
- **Internal** – outward radiation (direct sound, propagation valid outside the measurement grid)  
- **External** – inward radiation (room influence, propagation valid inside the measurement grid)  
- **Full** – sum of both (valid only at the measurement grid surface)  

In practical applications, **Internal** is the only useful setting, since it represents the direct radiation of the device under test propagated from the measurement grid without room or boundary effects.  
The External and Full options are retained mainly for completeness, testing, or research purposes.  

For debugging, note that the underlying coefficient matrix from the SHE stage contains both **C_nm** (outward going) and **D_nm** (inward going) terms.  
Inspecting these values directly can help verify whether sound field separation has been performed correctly.

---

## IR Generation from Propagated Complex Pressures
Converting propagated spherical harmonic expansion (SHE) data back into an impulse response (IR) is a delicate process.  
The source data is defined on a logarithmic frequency grid, whereas the inverse FFT requires a uniform linear grid.  
Moreover, the input data is band-limited, and the inverse FFT is extremely sensitive to any discontinuities in magnitude or phase — even slight mismatches or complex-valued asymmetry can cause time-domain smearing and spurious pre- or post-ringing.  

To minimize interpolation artefacts, it helps for the input data to extend close to DC (e.g. 5 Hz), so that only a small extrapolation is required.  
At the high-frequency end, life is easiest if the data extends beyond the Nyquist frequency of the intended output IR.  
Practically, this means using 96 kHz source IRs for the input to the SHE process and solving the SHE coefficients up to about 24 kHz.  
A 96 kHz source provides accurate phase information well beyond 24 kHz, while solving the expansion above 24 kHz would demand an unrealistically dense measurement grid.  
Thus, SHE up to 24 kHz is a practical and acoustically sufficient target.  
Once we have coefficients and pressures up to this frequency, we can safely set our output sample rate to 44.1 kHz, ensuring that the usable band lies well within Nyquist and avoiding the need for further interpolation.  

Finally, when generating the IR, it is possible that the phase data in the input is less than ideal.  
If this produces a time-smeared IR or excessive pre-ringing, there is an optional `enforce_causal_minphase` mode.  
When enabled, the phase is reconstructed from the magnitude response using a Hilbert transform, producing a minimum-phase (strictly causal) response.  
This guarantees a clean, physically causal IR with minimal or no pre-ringing.  
However, it also discards the original phase information, removing any non-minimum-phase features that may exist in reality — for instance, diffraction, port resonances, or deliberate linear-phase filtering.

---

## Extracting Anechoic Distortion Sweeps
A useful future feature would be the ability to isolate distortion measurements from room effects.  
This should be possible based on the Farina exponential sweep method, which highlights the important property that when a logarithmic sweep is deconvolved to the time domain, the linear and nonlinear components appear separated in time.  

Building on this, anechoic distortion data can be obtained using the following process, where (f) indicates frequency-domain operations and (t) indicates time-domain operations:

### Steps
1. **Propagate the internal sound field** from the SHE coefficients to the microphone position used for the distortion measurement (e.g., 1 m on the tweeter axis) and export as complex pressures `H_internal(f)`.  
   This represents the anechoic linear system transfer function (without nonlinear distortion).  
2. **Measure a logarithmic sweep** at that same mic position in the real room and deconvolve it to obtain the measured impulse response `h_meas(t)`.  
   This IR contains both the linear response and time-separated nonlinear distortion components (per the Farina method).  
3. **Window the fundamental (linear) part** of `h_meas(t)` around the main peak to isolate the linear response, and take its FFT to obtain `H_total(f)`.  
   This represents the measured linear transfer including both the loudspeaker and the room.  
4. **Compute the room transfer function:**  
   `H_room(f) = H_total(f) / H_internal(f)`  
5. **Build the inverse (“anti-room”) filter per frequency bin:**  
   `H_room_inv(f) = conj(H_room(f)) / |H_room(f)|²`  
6. **Convert the inverse filter to the time domain:**  
   `h_room_inv(t) = IFFT{ H_room_inv(f) }`  
   This produces the “anti-room” impulse response.  
7. **Apply the anti-room IR** by convolving it with the full measured IR:  
   `h_corrected(t) = h_meas(t) * h_room_inv(t)`  

- The linear component now represents the loudspeaker’s anechoic response.  
- The nonlinear components remain present but with their room coloration removed, giving near-anechoic distortion data.
