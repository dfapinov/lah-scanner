# Project Recommendations

## Hardware

### Measurement Concept
The **microphone must move around the speaker** while the **speaker remains stationary**.  
This ensures that the speaker–room interaction stays constant across all measurements and keeps the data compatible with **sound-field separation**.  
It will not work if the speaker rotates and the mic stays still.

Accurate positioning is critical — the solver assumes that each impulse response (IR) is captured at its exact intended coordinate, so that the high-frequency phase relationships remain consistent.

---

### Rotational Axis
Use a **strong, large-diameter thrust bearing** that allows the microphone arm to rotate around the speaker.  
Deep-groove ball bearings should be paird and axialy retained. Avoid 'lazy susan' turntable bearings, these are not stable.

A car **wheel hub and bearing** is an excellent option: precise, affordable, and widely available.  
For example:

- *VW Golf Mk4*:  
  - Bearing: `1J0598477`  
  - Hub: `1J0501117B`  

Choose a hub with a **central bore ≥30 mm** to allow a speaker-mounting pole to pass through.  
The central nut can be fixed to, or replaced by, the pole itself.

---

### Radial and Vertical Motion
Use **MGN15 profile linear rails** for all linear movement.  
Profile rails provide rigidity in all axes except the direction of travel, allowing a **single rail** to be used where **dual round rails** would otherwise be needed.

- For **vertical motion**: one profile rail is usually sufficient.  
- For **radial motion**: dual rails are recommended for stability over long spans.  

Profile rails are now affordable due to the 3D-printer component market.

---

### Drive System
Use **NEMA 17 stepper motors** with **GT2 belts and pulleys**.  
The main rotational axis can use a **large 3D-printed gear** bolted to the wheel-hub plate, driven by belt. This will provide high torque and good position accuracy.  
Speed is not critical in this application. Avoid direct drive ring gears, they have a lot of backlash and drive can become disengaged if there is movement in the platter.

---

### Structural Materials
Arms can be made from:
- **Aluminium extrusion** – easy to build and adjust  
- **Square aluminium tube** – stiffer and cheaper but requires accurate drilling  

The long radial arm should be **diagonally braced** to prevent sagging under load.

---

### Electronics and Control
Use a **3D-printer controller board** such as **SKR V1.4**, which supports up to five axes and uses **TMC silent stepper drivers**.  
This provides smooth, quiet motion at low cost.

The board runs **Marlin firmware**, which:
- Appears as a USB serial device  
- Accepts standard G-code commands directly (e.g. `G28`, `G0 X180`)  
- Can be easily controlled from Python  

To enable smoother motion, compile Marlin with **S-curve acceleration** enabled:  
[Video example](https://youtu.be/C0XjXqO6Ji8?si=XSGAy99Cbaehf7yj&t=12)

---

## Software

### GUI
A simple graphical interface would make the system more accessible.  
It should:
- Generate and visualize scan grids and paths  
- Save settings to the existing `config.py` files  
- Call the main scripts automatically  

This avoids embedding control logic inside the GUI itself.

---

### `Stage1_grid_gen.py`
Implement **angular weighting** so that:
- More measurement points are concentrated near the **front axis**, where HF directivity benefits from denser sampling  
- Fewer points are used behind the speaker (except for **dipoles**, which require full coverage)

---

### `Stage3_run_capture.py`
Update this script to control the robotic motion directly.  
It should:
- Read coordinates from `scan_path.csv`  
- Send **G-code commands** via USB serial to the Marlin-based controller  
  - Examples:  
    ```gcode
    G28        ; home all axes  
    G0 X180    ; move to position  
    ```

---

### `stage3_extract_frd.py`
Enhance this stage with optional **CEA-2034 outputs**, such as:
- On-axis response  
- Early reflections  
- Floor and ceiling bounces  

Plotting is not essential — FRD files can be viewed in **REW** or **VituixCAD** —  
but adding a **3D “bubble plot” visualization** of the sound field vs frequency would be a useful future feature.

---

## Extracting Anechoic Distortion Sweeps

A useful future feature would be the ability to isolate **distortion measurements** from **room effects**.  
This should be possible based on the **Farina exponential sweep method**, which highlights the important property that when a logarithmic sweep is **deconvolved to the time domain**, the **linear** and **nonlinear** components appear **separated in time**.

Building on this, anechoic distortion data can be obtained using the following process, where *(f)* indicates frequency-domain operations and *(t)* indicates time-domain operations:

### Steps

1. **Propagate the internal sound field** from the SHE coefficients to the microphone position used for the distortion measurement (e.g., 1 m on the tweeter axis) and export as complex pressures **H_internal(f)**.  
   This represents the *anechoic linear system transfer function* (without nonlinear distortion).

2. **Measure a logarithmic sweep** at that same mic position in the real room and **deconvolve** it to obtain the measured impulse response **h_meas(t)**.  
   This IR contains both the *linear response* and *time-separated nonlinear distortion components* (per the Farina method).

3. **Window the fundamental (linear) part** of **h_meas(t)** around the main peak to isolate the linear response, and take its FFT to obtain **H_total(f)**.  
   This represents the *measured linear transfer* including both the loudspeaker and the room.

4. **Compute the room transfer function** in the frequency domain:  
   **H_room(f) = H_total(f) / H_internal(f)**  
   *(Apply a small regularization floor to avoid divide-by-zero at spectral notches.)*

5. **Build the inverse (“anti-room”) filter** per frequency bin:  
   **H_room_inv(f) = conj(H_room(f)) / |H_room(f)|²**  
   *(Apply the same small regularization floor to prevent blow-ups at deep notches.)*

6. **Convert the inverse filter to the time domain** by IFFT:  
   **h_room_inv(t) = IFFT{ H_room_inv(f) }**  
   This produces the *“anti-room” impulse response.*

7. **Apply the anti-room IR** by convolving it with the full measured IR:  
   **h_corrected(t) = h_meas(t) * h_room_inv(t)**  

   - The **linear component** now represents the loudspeaker’s *anechoic response.*  
   - The **nonlinear components** remain present, but with their *room coloration removed,* giving near-anechoic distortion data.

---

*End of recommendations.*
