# Changelog

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

