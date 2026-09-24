# Viewer CTA-2034 methodology

Analysis and Export share `calculate_cta2034_energy_metrics` in the viewer-owned
`hals_engine/stage5_extract_pressures.py`. Its FRD adapter only assigns filenames
and reference phase. Both use the coordinate generator in `cta_coordinates.py`.
There is no second implementation in Analysis. HALS Post is unchanged.

Reference: [ANSI/CEA-2034-A](https://diy.midwestaudio.club/uploads/editor/3h/o3x5qp1ulkgh.pdf),
section 5.2, Appendix C Table 7 and the explanatory notes.

All directional averages use squared pressure, converted to dB afterwards.
Listening window uses nine directions. Early reflections equally weights the
five floor, ceiling, front, side and rear group energy means. The standard's
rear group uses three directions (horizontal -90, +90 and 180 degrees).
The separate horizontal-reflections diagnostic uses nineteen rear directions;
that larger group does not feed early reflections or predicted in-room response.

Sound power uses normalized Table 7 solid-angle weights, counting shared front
and rear directions once. Predicted in-room response uses 12% listening window,
44% early reflections and 44% sound power, in energy. Each DI subtracts its
composite level from listening-window level. SPL calibration offsets do not
apply to DI exports.

Analysis samples the reconstructed sphere using energy interpolation; Export
samples coefficients directly. Angular interpolation can therefore cause small
differences despite identical composite calculations. Composite FRD phase retains
the on-axis phase convention and is not an averaged physical phase.

Previously exported files need exporting again to receive these corrections.
