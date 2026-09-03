"""Shared, execution-model-independent Stage 5 pressure helpers.

Stage 5 has two distinct phase corrections which must not be confused:

* Capture-padding correction removes the known, artificial leading samples
  introduced while preparing captured IRs. It is applied before any TOF mode.
* TOF subtraction removes one physical propagation delay from the complex
  pressures by multiplying every frequency bin by ``exp(+j*omega*delay)``.

The selectable physical TOF modes are:

* Off: do not apply a physical propagation-delay correction.
* Ref Origin: use the configured observation radius. Mic/reference offsets move
  the origin and observation point together, so they do not change this radius.
* Min Phase Ref: work entirely in the frequency domain. Derive minimum phase
  from the on-axis magnitude, measure the robust linear excess group delay of
  measured/minimum-phase response, and apply that one delay to the whole batch.
* IR Peak: synthesize only the on-axis full-resolution IR, locate its earliest
  significant peak, convert the peak sample to delay/distance, and apply that
  one delay to the whole batch. Other observation points do not select a peak.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


DEFAULT_IR_CAPTURE_PADDING_SAMPLES = 50


def nearest_acoustic_origin(freqs, origins_mm, requested_hz):
    """Return the closest stored frequency and its Stage 2 origin in metres."""
    freqs = np.asarray(freqs, dtype=float)
    origins_mm = np.asarray(origins_mm, dtype=float)
    requested_hz = float(requested_hz)

    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("Stage 2 frequencies must be a non-empty 1D array.")
    if origins_mm.shape != (freqs.size, 3):
        raise ValueError("Stage 2 origins must contain one XYZ position per frequency bin.")
    if not np.isfinite(requested_hz) or requested_hz < 0.0:
        raise ValueError("Origin display frequency must be a finite, non-negative value.")
    if not np.all(np.isfinite(freqs)) or not np.all(np.isfinite(origins_mm)):
        raise ValueError("Stage 2 frequencies and origins must contain finite values.")

    index = int(np.argmin(np.abs(freqs - requested_hz)))
    return float(freqs[index]), origins_mm[index] / 1000.0


def centered_sweep_angles(range_deg, increment_deg):
    """Return a symmetric, centre-out sweep containing the on-axis point.

    The requested range is treated as a maximum excursion.  When it is not an
    exact multiple of the increment (for example 90 degrees with 20-degree
    steps), the outermost points are therefore +/-80 degrees rather than
    shifting the whole sweep by half a step and losing zero degrees.
    """
    sweep_range = int(range_deg)
    increment = int(increment_deg)
    if sweep_range < 0:
        raise ValueError("Sweep range must be zero or greater.")
    if increment <= 0:
        raise ValueError("Sweep increment must be greater than zero.")

    angles = [0]
    for angle in range(increment, sweep_range + 1, increment):
        angles.extend((-angle, angle))
    return angles


def clockwise_sweep_angles(range_deg, increment_deg):
    """Return preview numbering order, starting on-axis and moving clockwise.

    Positive sweep angles are the clockwise/right-hand side in the Stage 5
    reference-axis frame.  After reaching that edge, numbering continues from
    the opposite edge back towards the point immediately left of on-axis.
    """
    centered = centered_sweep_angles(range_deg, increment_deg)
    positive = sorted(angle for angle in centered if angle > 0)
    negative = sorted((angle for angle in centered if angle < 0))
    return [0, *positive, *negative]


def preview_sweep_sequence(range_deg, increment_deg, direction):
    """Return arc/angle pairs in the order used for preview numbering."""
    angles = clockwise_sweep_angles(range_deg, increment_deg)
    direction = str(direction).lower()
    if direction == "horizontal":
        return [("horizontal", angle) for angle in angles]
    if direction == "vertical":
        return [("vertical", angle) for angle in angles]
    if direction == "hor_vert":
        # The two arcs share their on-axis point. Number the whole horizontal
        # arc first, then continue around the vertical arc without duplicating it.
        return (
            [("horizontal", angle) for angle in angles]
            + [("vertical", angle) for angle in angles if angle != 0]
        )
    raise ValueError(f"Unknown sweep direction: {direction}")


def unique_cartesian_points(points, decimals=10):
    """Keep the first occurrence of each physical Cartesian position.

    Cartesian comparison avoids the multiple spherical representations at the
    front and rear poles (for example +180 and -180 degrees).
    """
    unique = []
    seen = set()
    for point in points:
        point_array = np.asarray(point, dtype=float)
        key = tuple(np.round(point_array, decimals=int(decimals)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(point_array)
    return unique


def smooth_fractional_octave_response(freqs, pressure, denominator):
    """Smooth magnitude and unwrapped phase on a uniform log-frequency grid."""
    freqs = np.asarray(freqs, dtype=float)
    pressure = np.asarray(pressure, dtype=np.complex128)
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError("Smoothing denominator must be greater than zero.")
    if freqs.ndim != 1 or pressure.ndim != 1 or len(freqs) != len(pressure):
        raise ValueError("Frequency and pressure inputs must be equal-length 1D arrays.")
    if len(freqs) < 3 or np.any(freqs <= 0.0) or np.any(np.diff(freqs) <= 0.0):
        raise ValueError("Smoothing requires at least three increasing positive frequencies.")

    log_freqs = np.log2(freqs)
    log_grid = np.linspace(log_freqs[0], log_freqs[-1], len(freqs))
    grid_step = float(log_grid[1] - log_grid[0])
    # Treat one fractional octave as the Gaussian full-width at half maximum.
    sigma_bins = (1.0 / denominator) / (2.354820045 * grid_step)

    magnitude_db = 20.0 * np.log10(np.abs(pressure) + np.finfo(float).eps)
    phase_rad = np.unwrap(np.angle(pressure))
    magnitude_grid = np.interp(log_grid, log_freqs, magnitude_db)
    phase_grid = np.interp(log_grid, log_freqs, phase_rad)
    magnitude_smooth = gaussian_filter1d(magnitude_grid, sigma_bins, mode="nearest")
    phase_smooth = gaussian_filter1d(phase_grid, sigma_bins, mode="nearest")

    magnitude_out = np.interp(log_freqs, log_grid, magnitude_smooth)
    phase_out = np.interp(log_freqs, log_grid, phase_smooth)
    phase_out = np.degrees(np.angle(np.exp(1j * phase_out)))
    return magnitude_out, phase_out


def get_tof_phasor(freqs, dist_m, c_sound):
    """Return the phase rotation used to subtract a propagation delay."""
    return np.exp(1j * 2.0 * np.pi * np.asarray(freqs) * (float(dist_m) / float(c_sound)))


def apply_ir_padding_phase(pressure, freqs, padding_samples, sample_rate):
    """Remove artificial capture padding from a 1D or frequency-major 2D response."""
    padding_samples = int(padding_samples)
    if padding_samples < 0:
        raise ValueError("IR capture padding samples must be zero or greater.")
    correction = np.exp(
        1j * 2.0 * np.pi * np.asarray(freqs, dtype=float) * (padding_samples / float(sample_rate))
    )
    values = np.asarray(pressure)
    return values * (correction[:, np.newaxis] if values.ndim > 1 else correction)


def get_min_phase_delay(p_complex, freqs, c_sound):
    """Estimate propagation distance from robust minimum-phase excess group delay.

    A constant phase rotation and the arbitrary 2*pi branch of unwrapped phase
    carry no timing information.  Taking adjacent phase differences of the
    measured/minimum-phase ratio removes both before estimating one physical
    delay.  The upper octave is excluded because a minimum-phase reconstruction
    from finite-band magnitude data is least reliable near its boundaries.
    """
    freqs = np.asarray(freqs, dtype=float)
    p_complex = np.asarray(p_complex, dtype=np.complex128)
    if freqs.ndim != 1 or p_complex.ndim != 1 or len(freqs) != len(p_complex):
        raise ValueError("Frequency and pressure inputs must be equal-length 1D arrays.")
    if len(freqs) < 2 or np.any(freqs <= 0.0) or np.any(np.diff(freqs) <= 0.0):
        raise ValueError("Minimum-phase delay requires increasing positive frequencies.")

    eps = np.finfo(float).eps
    mag_db = 20.0 * np.log10(np.abs(p_complex) + eps)
    fs_sim = 192000
    n_fft = 262144
    f_lin = np.fft.rfftfreq(n_fft, 1.0 / fs_sim)
    mag_lin_db = np.interp(f_lin, freqs, mag_db)

    f_min, f_max = float(freqs[0]), float(freqs[-1])
    lower_fade, upper_fade = f_min / 2.0, f_max * 2.0
    weights = np.ones_like(f_lin)
    lower_idx = (f_lin > lower_fade) & (f_lin < f_min)
    upper_idx = (f_lin > f_max) & (f_lin < upper_fade)
    weights[f_lin <= lower_fade] = 0.0
    weights[f_lin >= upper_fade] = 0.0
    weights[lower_idx] = 0.5 * (1.0 - np.cos(np.pi * np.log2(f_lin[lower_idx] / lower_fade)))
    weights[upper_idx] = 0.5 * (1.0 + np.cos(np.pi * np.log2(f_lin[upper_idx] / f_max)))

    cepstrum = np.fft.irfft(mag_lin_db * weights / 8.685889638, n=n_fft)
    cepstrum_window = np.zeros(n_fft)
    cepstrum_window[0] = 1.0
    cepstrum_window[1:n_fft // 2] = 2.0
    cepstrum_window[n_fft // 2] = 1.0
    min_phase = np.interp(freqs, f_lin, np.imag(np.fft.rfft(cepstrum * cepstrum_window)))

    excess = p_complex * np.exp(-1j * min_phase)
    delta_phase = np.angle(excess[1:] * np.conj(excess[:-1]))
    delta_omega = 2.0 * np.pi * np.diff(freqs)
    local_distances = -delta_phase / delta_omega * float(c_sound)
    midpoint_freqs = 0.5 * (freqs[1:] + freqs[:-1])

    fit_low = max(100.0, float(freqs[0]) * 2.0)
    fit_high = min(20000.0, float(freqs[-1]) / 2.0)
    if fit_high <= fit_low:
        fit_low, fit_high = float(freqs[0]), float(freqs[-1])
    fit_mask = (midpoint_freqs >= fit_low) & (midpoint_freqs <= fit_high)

    # Phase at deep response nulls is poorly defined. Retain a generous 40 dB
    # window so normal driver roll-off remains represented without allowing a
    # handful of null bins to dominate the timing estimate.
    pair_magnitude_db = np.minimum(mag_db[1:], mag_db[:-1])
    if np.any(fit_mask):
        fit_peak_db = float(np.max(pair_magnitude_db[fit_mask]))
        fit_mask &= pair_magnitude_db >= fit_peak_db - 40.0
    fit_mask &= np.isfinite(local_distances)
    candidates = local_distances[fit_mask]
    if candidates.size == 0:
        raise ValueError("No reliable frequency bins are available for minimum-phase delay estimation.")

    # A median/MAD pass rejects phase discontinuities without imposing an
    # arbitrary absolute phase intercept or unwrap branch.
    center = float(np.median(candidates))
    mad = float(np.median(np.abs(candidates - center)))
    if mad > np.finfo(float).eps:
        inliers = np.abs(candidates - center) <= 4.5 * 1.4826 * mad
        if np.any(inliers):
            candidates = candidates[inliers]
    return float(np.median(candidates))
