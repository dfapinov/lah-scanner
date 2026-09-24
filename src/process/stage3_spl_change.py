"""Successive-fit internal SPL changes on a fixed sphere in measurement coordinates."""
import numpy as np
from scipy.special import sph_harm_y

from utils import hankel2, translate_coordinates


def evaluation_sphere(point_count, radius_m):
    if int(point_count) != point_count or point_count < 4:
        raise ValueError('SPL-change sphere needs an integer point count of at least 4.')
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError('SPL-change sphere radius must be positive and finite.')
    index = np.arange(int(point_count))
    theta = np.arccos(1 - 2 * (index + .5) / point_count)
    phi = (index * np.pi * (3 - np.sqrt(5)) + np.pi) % (2 * np.pi) - np.pi
    return np.full(int(point_count), radius_m), theta, phi


def maximum_spl_change(previous, current, floor_db=-40.0, *, include_samples=False):
    """Use the previous fit's spatial peak at this frequency as a common reference.

    Return None for undefined comparisons (zero previous peak or nonfinite field).
    No phase difference enters this amplitude-only metric.
    """
    if not np.isfinite(floor_db) or floor_db >= 0:
        raise ValueError('SPL-change floor must be finite and negative (for example -40 dB).')
    previous = np.abs(np.asarray(previous))
    current = np.abs(np.asarray(current))
    if previous.shape != current.shape or previous.ndim != 1 or not previous.size:
        raise ValueError('SPL-change fields must be matching nonempty vectors.')
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        return None
    peak = float(previous.max())
    if peak <= 0:
        return None
    with np.errstate(divide='ignore'):
        # Logs separately avoid division overflow for very different fit scales.
        old_db = 20 * (np.log10(previous) - np.log10(peak))
        new_db = 20 * (np.log10(current) - np.log10(peak))
    change = np.maximum(new_db, floor_db) - np.maximum(old_db, floor_db)
    index = int(np.argmax(np.abs(change)))
    def finite_or_none(value):
        return float(value) if np.isfinite(value) else None
    result = dict(max_change_db=float(abs(change[index])), signed_change_db=float(change[index]),
                point_index=index, previous_peak_magnitude=peak,
                previous_magnitude=float(previous[index]), current_magnitude=float(current[index]),
                previous_relative_db=finite_or_none(old_db[index]),
                current_relative_db=finite_or_none(new_db[index]),
                previous_clamped_db=float(max(old_db[index], floor_db)),
                current_clamped_db=float(max(new_db[index], floor_db)))
    if include_samples:
        result['_absolute_change_db'] = np.abs(change)
    return result


def reconstruct_internal_fits(coefficients, f_hz, speed, sphere, origin_m):
    """Reuse the outgoing basis for all fits; block directions to bound memory."""
    r, theta, phi = translate_coordinates(*sphere, origin_m)
    if np.any(r <= 0):
        raise ValueError('Evaluation sphere intersects an expansion origin.')
    mode_count = max(len(c) for c in coefficients)
    max_n = int(round(np.sqrt(mode_count))) - 1
    degrees = np.repeat(np.arange(max_n + 1), 2*np.arange(max_n + 1) + 1)
    azimuth_orders = np.concatenate([np.arange(-n, n + 1) for n in range(max_n + 1)])
    padded = np.zeros((len(coefficients), mode_count), dtype=complex)
    for i, c in enumerate(coefficients):
        padded[i, :len(c)] = c
    fields = np.empty((len(coefficients), len(r)), dtype=complex)
    for start in range(0, len(r), 256):
        sl = slice(start, start + 256)
        angular = sph_harm_y(degrees[:, None], azimuth_orders[:, None], theta[None, sl], phi[None, sl])
        radial = hankel2(degrees[:, None], 2*np.pi*f_hz/speed*r[None, sl])
        fields[:, sl] = padded @ (radial * angular)
    return fields


def evaluate_frequency_changes(task):
    f_hz, orders, fits, speed, origin_m, point_count, radius_m, floor_db = task
    sphere = evaluation_sphere(point_count, radius_m)
    valid = {n: fit for n, fit in fits.items() if fit is not None and fit['usable']}
    if not valid:
        return []
    fields = reconstruct_internal_fits([fit['internal_coeffs'] for fit in valid.values()],
                                       f_hz, speed, sphere, origin_m)
    reconstructed = dict(zip(valid, fields))
    result = []
    for n in orders:
        if n not in valid or n-1 not in valid:
            continue
        maximum = maximum_spl_change(reconstructed[n-1], reconstructed[n], floor_db, include_samples=True)
        if maximum is None:
            continue
        index = maximum['point_index']
        effective_previous = len(valid[n-1]['internal_coeffs'])**.5 - 1
        effective_current = len(valid[n]['internal_coeffs'])**.5 - 1
        maximum.update(order_n=int(n), previous_order_n=int(n-1), frequency_hz=float(f_hz),
                       theta_deg=float(np.degrees(sphere[1][index])),
                       phi_deg=float(np.degrees(sphere[2][index])),
                       effective_previous_n=int(round(effective_previous)),
                       effective_current_n=int(round(effective_current)),
                       order_capped=effective_current == effective_previous)
        result.append(maximum)
    return result


def evaluate_frequency_chunk(tasks):
    """Reuse a solve pool while limiting reconstruction to four concurrent chunks."""
    return [evaluate_frequency_changes(task) for task in tasks]


def summarize_changes(orders, frequency_results, point_count, radius_m, floor_db, sample_count):
    peaks, counts, active_counts, percentiles, all_percentiles = [], [], [], [], []
    for n in orders:
        values = [v for group in frequency_results for v in group if v['order_n'] == n]
        peak = max(values, key=lambda v: v['max_change_db']) if values else None
        peaks.append({k:v for k,v in peak.items() if k != '_absolute_change_db'} if peak else None)
        counts.append(len(values))
        active_counts.append(sum(not v['order_capped'] for v in values))
        samples = [v['_absolute_change_db'] for v in values if '_absolute_change_db' in v]
        active_samples = [v['_absolute_change_db'] for v in values
                          if not v['order_capped'] and '_absolute_change_db' in v]
        all_percentiles.append(float(np.percentile(np.concatenate(samples), 99)) if samples else None)
        percentiles.append(float(np.percentile(np.concatenate(active_samples), 99)) if active_samples
                           else (0.0 if values and not active_counts[-1] else None))
    return dict(orders=list(orders), max_change_db=[p['max_change_db'] if p else None for p in peaks],
                p99_change_db=percentiles, p99_all_frequencies_change_db=all_percentiles,
                percentile=99, percentile_scope='All sampled directions pooled across valid frequencies that add a degree; capped comparisons excluded. Fully capped orders displayed as zero, not used for selection.',
                peaks=peaks, valid_frequency_counts=counts, added_degree_frequency_counts=active_counts,
                sample_frequency_count=sample_count, sphere_points=point_count, radius_m=radius_m,
                floor_db=floor_db, sphere_center_m=[0, 0, 0],
                direction_convention='theta: polar angle from +Z; phi: azimuth from +X toward +Y, in measurement coordinates')


def plot_spl_changes(ax, result):
    primary = result.get('p99_change_db', [None] * len(result['orders']))
    values = [np.nan if v is None else v for v in primary]
    ax.plot(result['orders'], values, 'o-', color='#2a9d8f', label='_nolegend_')
    from stage3_optimize_she_settings import format_stage3_order_axis
    format_stage3_order_axis(ax)
    ax.set_ylabel('Absolute SPL change (dB)')
    ax.set_title('Change to Directivity Pattern', fontsize=12, fontweight='bold', pad=32)
    ax.text(.5, 1.02, 'Source pressure is reconstructed over a sphere; the change in SPL\n'
            'across directions is assessed when one more order is added.',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8)
    ax.set_xticks(result['orders'])
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=.35)
    capped = [n for n, count, active in zip(result['orders'], result['valid_frequency_counts'], result['added_degree_frequency_counts']) if count and not active]
    if capped:
        ax.scatter(capped, [values[result['orders'].index(n)] for n in capped], marker='x', s=80,
                   color='#777777', label='All frequencies order-capped: no additional degree tested')
    for key, choice in result.get('order_choices', {}).items():
        ax.scatter([choice['n']], [choice['spl_db']], s=100, edgecolors='white', zorder=5,
                   color='#e45756', marker='o', label='within 1dB of min. -1N')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)


def save_spl_changes(result, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10, 5))
    FigureCanvasAgg(fig)
    plot_spl_changes(fig.add_subplot(111), result)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
