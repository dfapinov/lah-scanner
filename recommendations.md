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
Avoid deep-groove ball bearings, which are not ideal for axial load retention.

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
The main rotational axis can use a **large 3D-printed gear** bolted to the wheel-hub plate, providing high torque and good position accuracy.  
Speed is not critical in this application.

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

*End of recommendations.*
