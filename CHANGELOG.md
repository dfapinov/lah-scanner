# Changelog

## [2.2.11] - 2026-06-17
Added: Stage 3 now supports a custom frequency range for the order-N search, with a UI note that the selected range should sit entirely within the reflection-free time/range.
Fixed: Project JSON handling now preserves the discovered project file path, including project names that use `-` or `_`, and merges saves without overwriting unrecognized keys.

## [2.2.10] - 2026-06-16
Changed: Stage 2 speed-of-sound optimization now scores each candidate across six non-coherent probe bins from 5 kHz to 10 kHz. Speed-of-sound selection uses
mean fit error across the probe bins with a small worst-bin penalty.
The speed-of-sound value flows through the processing files themselves: Stage 2 writes `speed_of_sound_mps` to the NPZ, Stage 3 and Stage 4 read it from that NPZ, Stage 4 writes it into the coefficients H5, and Stage 5 reads it from the H5.
Added: Stage 3 now saves an Int/Ext ratio vs. Order N sweep plot and shows the sweep in the Stage 3 choice popup with suggested orders highlighted as points.
Added: Stage 3 now calculates an "Order N at roll-off start" option using the first sustained acceleration in the post-peak Int/Ext decline.

## [2.2.9] - 2026-05-30

Added: Project waypoint metadata loading for baffle points, tweeter position, reference origin, and related measurement reference points. These values are now imported from project JSON/CSV metadata and used to populate downstream GUI fields.
Added: Stage 5 visual preview now uses the project baffle waypoints, including tilted baffle planes, instead of assuming a simple axis-aligned cabinet reference.
Changed: Stage 1 now accepts either `project_folder/measurement_set` or `project_folder/recordings` as the IR WAV input folder, matching folder names case-insensitively for compatibility with both new and old capture systems.
Changed: Stage 5 SPL calibration is stored as `stage5_vars.frd_db_offset` so the capture and post-processing tools share one FRD export level key.
Changed: Stage 3 now only runs the order-N SFS test. The previous noise-floor and lambda sweep code is retained in the script as blocked-out legacy reference because it was found ineffective in practice.
Changed: Stage 3 GUI tab renamed to "Stage 3: Find Order N" to better match its purpose.
Changed: Stage 4 GUI/config defaults updated to Noise Floor Start = -30 dB, Noise Floor Max = -40 dB, and Max Lambda = 0.000001.
Fixed: Stage 2 could sometimes fail when saving adjusted validation results after the first automatic save. The save process now avoids reusing stale origin data and writes the updated results more safely.

## [2.2.8] - 2026-05-13

Fixed: Kword args out of sync between hlas_gui.py and grid_gen.py for cyl_radius and cyl_height after conversion from meters to millimeters. 

## [2.2.7] - 2026-05-08

Fixed: Spatial Error Viewer not compatible with phase compensation applied to 'extract_pressures_core.py'  in previous update. How: Added phase correction bypass mode in extract_pressures_core.py.
Added: "IR Peak" mode for the "Subtract Time Of Flight" feature. This mode automatically finds the phase timing reference by temporarily generating the impulse response for the point on the reference axis, locating the IR peak, and using that sample time to determine the exact amount of delay phase compensation to apply to all FRD exports in the measurement set.

## [2.2.6] - 2026-05-08

Added: FRD DB Offset. This feature scales the magnitude level of the exported FRD files to allow SPL calibration. It does not affect exported impulse response wavs.
Added: Microphone calibration files are saved to the project.jason file as a fall-back in case the original file is missing.
Removed: Redundant 'target_fs' argument from complex_to_ir_core.py Why: AI added it in previous version without developers intent.

## [2.2.5] - 2026-05-07

Added: Phase compensation in `extract_pressures_core.py` for the 5-sample padding introduced during upstream IR generation (`audio.py "split_idx = len(inv_data) - 5"). The input IR sample rate is now passed through the pipeline to enable precise 5-sample phase rotation.
Changed: The phase reference for measurements when "Subtract Time of Flight" is enabled now dynamically tracks the `mic_offset` values. This allows setting the phase reference at positions other than the origin of the measurement grid (e.g., if the tweeter is offset forward and above the origin). This change is also reflected visually as the origin point of the red reference axis in the Stage 5 GUI plots.

## [2.2.4] - 2026-05-07

Changed: FRD file name convention for better VituixCAD compatability.
Changed: Seperate Horizontal and Vertical response file output directories replaced with a single directory named after the project title. Why: VituixCAD only opens response files from a single directory.
Fixed: GUI did not pass use_coord_list to stage 5 when using sweep mode, resulting in variable pulls from config_process.py.
Fixed: CTA-2034 export mode did not support the 'Subtract Time of Flight' setting, resulting in the exported On-Axis phase including the full TOF delay. How: Added the subtract_tof argument to the CTA-2034 extraction function. Extracted the TOF phase rotation math into a shared helper function and applied it to the CTA-2034 data.

## [2.2.3] - 2026-05-06

Added: Completion timer for each processing stage.
Added: Auto-saving of Stage 2 validation results image.
Added: "Save View Image" button in Stage 5 for microphone position plots.
Changed: Transitioned Cylinder Height, Cylinder Radius, and Bottom Cutoff grid settings to millimeters (mm) across the UI and processing scripts.
Changed: Physical waypoints now override and auto-fill the numerical cylinder dimensions in the UI.

## [2.2.2] - 2026-05-05

Fixed: Severe performance bottleneck on Linux during parallel processing (Thread Thrashing). How: Enforced single-threading for underlying C math libraries (OpenBLAS, MKL, OMP, etc.) before importing NumPy, preventing CPU oversubscription.

Fixed: Potential deadlocks and memory bloat on Linux when using Python's `multiprocessing`. How: Explicitly set the multiprocessing context start method to `'spawn'` rather than the Linux default `'fork'` to ensure clean process initialization when interacting with multithreaded math libraries.

Fixed: `RuntimeError: main thread is not in main loop` crashing the application when opening result viewers. How: Enforced strict thread-safety by marshaling all Tkinter window creation (e.g., Validation UI, SHE Plotter) back to the main GUI thread using `.after()` callbacks after background processing threads complete.

## [2.2.1] - 2026-05-05

Added: Project file naming and auto-discovery in the HALS GUI. How: Project settings are now saved using the specific project name (e.g., `[project_name]_project.json`) rather than a generic filename. When browsing to a new project directory, the application automatically scans for any file ending in `_project.json`, loads the configuration, and updates the Project Name field in the UI to match the discovered file.

Added: Auto-saving of initial project settings in the HALS GUI. How: Save is triggered before running any processing stage or grid generation action. Checks for the existence of `scanner_project.json` and saves the current UI state if the file is not found, ensuring a baseline configuration is established without overwriting existing user saves.

Added: Comprehensive Debug Mode for the HALS GUI. How: Introduced a DEBUG_MODE toggle that intercepts and mirrors all CLI output to a hals_debug.log file and the original terminal. The debug log automatically captures system-level diagnostics, library versions, directory listings with file sizes, project settings, and detailed tracking of all major UI interactions.

Fixed: CLI output auto-scroll not engaging on Linux. How: Replaced fractional scroll threshold with calculation that checks if the view is within 3 lines of the bottom. Accommodates variations in font metrics and line-height.

Fixed: Toolbar buttons disappearing under the 3D viewer when entering full screen on Linux. How: Wrapped the Matplotlib FigureCanvasTkAgg in a dedicated ttk.Frame container to enforce strict layout boundaries and prevent it from painting over its sibling widgets during rapid window resizing.

Fixed: Stage 5 (evaluation mic positions) 3D viewer scale can change aspect ratio distorting display. How: Used Matplot set_box_aspect 1:1:1.

Fixed: Stage 2 discards optimised origins if plot window is closed without using 'accept & save' button'. How: Stage 2 now saves the acoustic origns immediatly on completion. 'Accept and save' is now only needed for adjustments.

## [2.2.0] - 2026-04-26
Refactor coord_viewer_gui.py as separate plotting core and gui scripts, plus code comments.
Update readme.md

## [2.1.0] - 2026-04-25
Gui script created to drive the measurement grid generation and processing pipeline. Some updates 
to processing scripts to better support gui integration.

## [2.0.0] - 2026-03-20
A compatability breaking release. All scripts have been significantly refactored with the aim of
becoming the key post-processing pipeline a comunity project called HALS (Holographic Acoustic
Loudspeaker Scanner). This is a beta release. Updates and documentation will follow.

---

## [1.1.0] – 2025-11-06
### Aim
Align captured impulse responses with the **Farina distortion measurement method**,  
using the original digital excitation sweep for deconvolution rather than the recorded loopback.  
This enables extraction of harmonic distortion components from the IR.  
Loopback has been replaced with a short Barker marker pulse for more precise time alignment  
via matched-filter correlation.

### Changed
- **sweep_function.py**
  - Now saves the original digital excitation sweep (`excitation_signal.wav`) for later deconvolution.
  - Replaced loopback log sweep with a short Barker marker pulse for improved alignment accuracy.

- **make_ir_function.py**
  - Updated to use the saved digital excitation signal for deconvolving the microphone recording.
  - Fully compliant with the Farina deconvolution approach.

- **run_capture.py**
  - Now correctly passes the updated recordings from `sweep_function.py` to `make_ir_function.py`.
  - Renamed configuration option `SAVE_MIC_LOOP_SIG` → `DEBUG_SAVES`.
  - Adjusted which files are saved when debug mode is enabled.

### Notes
Interface response compensation via the electrical loopback is no longer applied,  
as most modern audio interfaces exhibit flat frequency response.

---

## [1.0.0] – 2025-10-29
### Added
- Initial stable release of the capture and SHE processing pipeline.

---
