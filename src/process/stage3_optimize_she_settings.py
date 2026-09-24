#!/usr/bin/env python3
"""
Open-Branch Spherical Energy Optimizer
======================================
Compares order diagnostics to recommend a maximum solve order.
"""

import os
import sys
import time
import json

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import multiprocessing
import concurrent.futures
import schema

try:
    from she_solver_core import _solve_one_frequency
    from utils import load_and_parse_npz, translate_coordinates, get_grid_limit, get_kr_limit
except ImportError as e:
    sys.exit(f"Error: Could not import required solver modules. {e}")

# =============================================================================
SFS_ACCEPTABLE_RATIO_DB = 20.0
NO_SEPARATION_NOTE = (
    'No tested order produces Int/Ext separation greater than 20 dB. '
    'This may occur when the test frequency range is below the reflection-free time-window (RFT) range, '
    'or when the measurement grid or data needs attention. Check the test frequency range, windowing, '
    'measurement quality, and measurement-point count and spatial coverage; re-measure if needed.'
)
STAGE3_CHOICE_STYLES = {
    'knee': ('#e45756', 'o', 'Roll-off knee'),
    'tail': ('#e45756', 'o', 'Soft -25 dB tail'),
    'spl': ('#e45756', 'o', 'Directivity change'),
    'spl_safe': ('#9467bd', 's', 'SPL plateau minus one order'),
}
STAGE3_CHOICE_HELP = """Choosing an order for Stage 4

Choosing the test frequency band
Set the upper limit to the driver's usable response before it rolls off toward
the noise floor. Where possible, start at the RFT-derived lower frequency.
Stage 4 already limits order at lower frequencies, so assessing the usable upper
band avoids diluting high-order contributions. Octave resolution controls all
diagnostics; finer sampling can reveal narrower details and takes longer.

Top graph: Source to Room Field Ratio
Above the reflection-free time-window (RFT) boundary, time windowing removes
reflections. Little energy should remain in the external field, so Int/Ext should
be large. A ratio above 20 dB is our suitability rule of thumb, not proof of
accuracy or a direct numerical condition number. This sets the eligibility
threshold whenever the band permits its use and at least one tested order exceeds
20 dB. Below RFT, real environmental sound can remain in the external field;
the ratio stays visible for inspection but is not used for selection.

Bottom graph: Sound Power Discarded
This shows how much modeled internal power lies above each stopping order in
one reference fit. The reference defaults to the highest tested order above
20 dB Int/Ext. More negative means less discarded: -25 dB is about 0.32%, and
-40 dB is 0.01%. These are mean per-frequency shares within the tested band,
not guarantees for every direction or frequency. If nearby orders have similar
separation, a very small tail can support choosing the lower one. If appreciable
tail remains before the reference, consider whether you need more measurement
points or improved coverage. The tail cannot measure missing information beyond
its reference: it is zero at the reference by construction, and that endpoint
is omitted. A denser grid is not guaranteed to improve the result.
The dropdown changes this diagnostic and its backup choice, not the ratio knee.

The three choices
Roll-off knee: a candidate when Int/Ext is usable. It identifies where
the ratio's post-peak decline becomes sharper. Detection can be sensitive to
the shape and rate of decline, so check the curve rather than treating the knee
as proof of an optimum. If the main knee is at or below 20 dB, make one
gentler second pass and choose the latest earlier knee above 20 dB, confirmed
by two consecutive declines. This second pass does not use the geometric bend
fallback or repeatedly relax the criteria.

Soft -25 dB tail: the first order below the reference with tail <= -24 dB,
allowing 1 dB around the target. When Int/Ext is usable, all offered candidates must exceed 20 dB separation.

Directivity change: find the first valid order within 1 dB of the lowest SPL
change, then step back one order within the valid tested range. It becomes primary when Int/Ext is not usable. Each graph marks its
own candidate with a red dot; choose among the available methods below the graphs.

Incremental SPL change graph
The coloured curve shows the 99th percentile of absolute local SPL changes when
increasing order from N-1 to N. It pools sampled directions and frequencies that
actually add a degree. This is the level below which 99% of those changes lie;
isolated peaks have less influence. The maximum curve retains visibility of
narrow features affecting fewer than 1% of samples.
Both fields use the same fixed assessment floor, -40 dB below the previous
fit's spatial peak at each frequency. A change from -30 to -24 dB counts as 6 dB;
-46 to -40 counts as zero. Phase-only changes do not count. Capped comparisons
cannot dilute the percentile; fully capped orders are marked and cannot establish
a plateau. The default sphere has 1,000 points at 1 m, centred on the measurement
coordinate origin, and should enclose the source. These curves describe sampled
points, not a continuous-sphere guarantee. Radius and point count are in
Advanced Settings. The assessment floor is fixed at -40 dB. The reconstruction reuses the existing solves.

Below the reflection-free range
If the upper limit is below RFT, or no tested order exceeds 20 dB Int/Ext, use the
simple SPL-change rule to choose the actual solve order directly. Tail
power does not select or limit the order in this mode, and no tail reference is
needed. The maximum curve also does not choose the order. All available graphs share the results view; Int/Ext remains visible for inspection.
Choose the first valid order within 1 dB of the minimum SPL change, then step
back one order. Fully capped orders and missing values are excluded. If there
is no earlier valid order, use the lowest valid tested order. This always gives
a candidate when valid comparisons exist. Small changes do not prove accuracy.
Only a band whose upper limit is below RFT is reset to the two octaves below that
limit. An above-RFT band with poor separation retains the entered limits.

How we recommend an order
Int/Ext mode: recommend the highest order among the eligible knee, sound-power,
and directivity-change suggestions. Each must exceed 20 dB separation.
SPL mode: first order within 1 dB of the minimum, then one order back.
If acceptable separation is achieved only below N4, that is unusually limited
except perhaps for a subwoofer. Check point count, spatial coverage, positioning,
windowing and the test band; re-measure if needed. These are practical heuristics,
not stability guarantees.

Select a choice, then click Use in Stage 4. The radio buttons control the actual
order transferred. The combined results plot is saved for later inspection.
"""


def stage3_order_choices(orders, ratios, residuals, knee, tail_db, reference_n,
                         tail_only=False, manual_reference=False):
    if tail_only and not manual_reference:
        return {}  # SPL-mode choices come directly from the percentile curve.
    selected = {'knee': knee['n'] if knee and not tail_only else None}
    for key, threshold in [('tail', -24.0)]:
        tails = [n for n, db in zip(orders, tail_db) if reference_n is not None and n < reference_n and not np.isnan(db) and db <= threshold]
        selected[key] = min(tails) if tails else None
    result = {}
    for key, n in selected.items():
        if n is None:
            continue
        idx = list(orders).index(n)
        if (not tail_only and
                (not np.isfinite(ratios[idx]) or ratios[idx] <= SFS_ACCEPTABLE_RATIO_DB)):
            continue
        result[key] = {'n': n, 'ratio': ratios[idx], 'err': residuals[idx],
                       'tail_db': tail_db[idx],
                       'label': STAGE3_CHOICE_STYLES[key][2]}
    return result


def stage3_test_band(start_hz, end_hz, rft_hz=None):
    if not np.isfinite(start_hz) or not np.isfinite(end_hz) or start_hz <= 0 or end_hz <= 0:
        raise ValueError('Stage 3 frequency limits must be finite and positive.')
    tail_only = rft_hz is not None and np.isfinite(rft_hz) and rft_hz > 0 and end_hz < rft_hz
    if tail_only:
        return end_hz / 4, end_hz, True
    return min(start_hz, end_hz), max(start_hz, end_hz), False


def recommended_stage3_choice(options):
    """Choose the highest pre-qualified order; keep style priority for ties."""
    keys = [key for key in STAGE3_CHOICE_STYLES if key in options]
    return max(keys, key=lambda key: options[key]['n']) if keys else None


def select_spl_tail_reference(orders, ratios, diagnostic):
    """Use percentile evidence without letting it postpone a maximum-based warning."""
    maximum = _select_spl_reference_curve(orders, ratios, diagnostic, 'max_change_db')
    if not diagnostic or 'p99_change_db' not in diagnostic:
        return maximum
    percentile = _select_spl_reference_curve(orders, ratios, diagnostic, 'p99_change_db')
    candidates = [r for r in (maximum, percentile) if r['n'] is not None]
    # A maximum plateau must also be quiet in the percentile curve.
    if maximum.get('method') == 'SPL low-change plateau onset':
        ns = list(diagnostic['orders'])
        p99 = np.asarray(diagnostic['p99_change_db'], dtype=float)
        valid = np.isfinite(p99) & (np.asarray(diagnostic['added_degree_frequency_counts']) > 0)
        i = ns.index(maximum.get('plateau_start_n', maximum['n']))
        if not (valid.any() and valid[i:i+4].all() and
                np.all(p99[i:i+4] <= np.min(p99[valid]) + 2)):
            candidates.remove(maximum)
    result = dict(min(candidates, key=lambda r: r['n']) if candidates else percentile)
    result['primary_metric'] = 'p99_change_db'
    result['reference_candidates'] = {'maximum':maximum['n'], 'percentile':percentile['n']}
    return result


def _select_spl_reference_curve(orders, ratios, diagnostic, metric, plateau_only=False):
    """Provisional reference: one order before the pre-rise order (M=i-2).

    Prefer three consecutive, supported increments >3 dB above the minimum.
    Otherwise accept the leading edge of a confirmed four-order low plateau.
    Gaps and transitions with no added degree cannot establish a minimum/rise.
    """
    unresolved = dict(n=None, ratio=None, fallback=True, tail_only=True,
                      method='SPL reference unresolved', provisional=True)
    if not diagnostic:
        return dict(unresolved, reason='Incremental SPL testing is disabled. Enable it and rerun Stage 3 to obtain an automatic reference.')
    ns = np.asarray(diagnostic['orders'], dtype=int)
    changes = np.asarray(diagnostic[metric], dtype=float)
    unresolved['selection_metric'] = metric
    support = np.asarray(diagnostic['added_degree_frequency_counts'])
    valid = np.isfinite(changes) & (support > 0)
    if not valid.any():
        return dict(unresolved, reason='No valid SPL comparison added a harmonic degree. Check the tested orders and frequency-dependent order limits.')
    minimum_index = int(np.argmin(np.where(valid, changes, np.inf)))
    minimum = changes[minimum_index]
    for i in ([] if plateau_only else range(minimum_index + 1, len(ns) - 2)):
        if (valid[i:i+3].all() and np.all(np.diff(ns[i:i+3]) == 1)
                and np.all(changes[i:i+3] > minimum + 3)):
            reference_n = int(ns[i] - 2)
            if reference_n not in orders:
                return dict(unresolved, reason='The conservative SPL reference falls below the tested order range. Test lower orders or select a reference manually.')
            return dict(n=reference_n, ratio=float(ratios[list(orders).index(reference_n)]),
                        fallback=True, tail_only=True, provisional=True,
                        selection_metric=metric,
                        label='provisional SPL reference, backed off one order',
                        method='SPL sustained-rise reference minus one order',
                        rise_start_n=int(ns[i]), pre_rise_n=int(ns[i]-1))
    # A shallow basin can provide a reference without a subsequent sharp rise.
    # Require an observed descent and four real increments, never capped zeros.
    for i in range(3, len(ns)-3):
        block = changes[i:i+4]
        required_drop = max(.5, .5*np.median(block)) if plateau_only else 2
        if (valid[i-3:i+4].all() and np.all(np.diff(ns[i-3:i+4]) == 1)
                and np.ptp(block) <= 3
                and block[-1] >= block[0] - 1
                and np.median(block) <= minimum + 3
                and np.median(changes[i-3:i]) >= np.median(block) + required_drop):
            # The first transition can still be entering the plateau. Use its
            # second order, confirmed by two more genuine increments.
            reference_n = int(ns[i+1])
            if reference_n in orders:
                return dict(n=reference_n, ratio=float(ratios[list(orders).index(reference_n)]),
                            fallback=True, tail_only=True, provisional=True,
                            selection_metric=metric,
                            label='provisional SPL early-plateau reference',
                            method='SPL low-change plateau onset',
                            plateau_start_n=int(ns[i]),
                            plateau_end_n=int(ns[i+3]))
    return dict(unresolved, reason='No sustained SPL rise or confirmed low-change plateau was found. The plateau fallback needs four supported orders with little further decline after a clear drop. Review or extend the tested order range, or select a reference manually.')


def select_spl_order_choices(orders, ratios, residuals, diagnostic):
    """First valid increment within 1 dB of the minimum, backed off one order."""
    if not diagnostic or 'p99_change_db' not in diagnostic:
        return {}, 'No valid SPL-change data. Rerun Stage 3.'
    ns = np.asarray(diagnostic['orders'], dtype=int)
    values = np.asarray(diagnostic['p99_change_db'], dtype=float)
    valid = (np.isfinite(values)
             & (np.asarray(diagnostic['added_degree_frequency_counts']) > 0)
             & np.isin(ns, orders))
    if not valid.any():
        return {}, 'No valid SPL comparisons added a degree; capped zeros cannot select an order.'
    minimum = float(values[valid].min())
    entry = int(np.min(ns[valid & (values <= minimum + 1.0)]))
    earlier = ns[valid & (ns <= entry - 1)]
    n = int(earlier.max()) if earlier.size else int(ns[valid].min())
    i = list(orders).index(n)
    j = list(ns).index(n)
    convergence_note = ''
    option = dict(n=n, ratio=ratios[i], err=residuals[i], tail_db=np.nan,
                  spl_db=float(values[j]), label='Directivity change: within 1 dB of minimum, then one order back',
                  near_minimum_n=entry, minimum_change_db=minimum, tolerance_db=1.0,
                  backoff_orders=entry-n, convergence_note=convergence_note)
    reason = (f'First order within 1 dB of the minimum SPL change: N={entry}; '
              f'recommend N={n} after backing off one order within the valid tested range.')
    return {'spl': option}, reason + (' ' + convergence_note if convergence_note else '')


def format_tail_power(db):
    if np.isnan(db):
        return 'discarded power unavailable'
    return f"{100 * 10**(db / 10):.3g}% discarded ({db:.2f} dB)"


TAIL_EXPLAINER = 'Sound Power Discarded'
RATIO_EXPLAINER = ('Ratio of source to room field energy in the reflection-free windowed range.\n'
                   'Larger is better for solve stability.')


def format_stage3_order_axis(ax):
    ax.set_xlabel('Stopping Order N\nLower \u2190 directivity detail \u2192 Higher', fontsize=8)


def format_stage3_ratio_axis(ax, inspection_only=False):
    subtitle = ('Outside the usable separation range: shown for inspection only.\n'
                'Not used to choose the solve order.' if inspection_only else RATIO_EXPLAINER)
    ax.set_title('Source to Room Field Ratio', fontsize=12, fontweight='bold', pad=32)
    ax.text(.5, 1.02, subtitle, transform=ax.transAxes, ha='center', va='bottom', fontsize=8)
    format_stage3_order_axis(ax)
    ax.tick_params(axis='x', labelbottom=True)



def highlight_stage3_choices(ax, options, tail=False):
    artists = []
    for key, (color, marker, label) in STAGE3_CHOICE_STYLES.items():
        if key != ('tail' if tail else 'knee'):
            continue
        opt = options.get(key)
        if opt:
            if tail:
                artists.append(ax.scatter([opt['n']], [max(opt['tail_db'], -80.0)],
                                          s=150, color=color, marker=marker,
                                          edgecolors='white', linewidths=1.2, zorder=5,
                                          label=f"-25 dB rule of thumb: N={opt['n']}"))
            else:
                artists.append(ax.scatter([opt['n']], [opt['ratio']], s=150 - 30*len(artists),
                                           color=color, marker=marker, edgecolors='white',
                                           zorder=5+len(artists), label=f"{label}: N={opt['n']}"))
    return artists

def find_rolloff_knee(orders, ratios):
    orders = np.asarray(orders, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    finite = np.isfinite(orders) & np.isfinite(ratios)
    orders = orders[finite]
    ratios = ratios[finite]
    if len(orders) < 4:
        return None

    sort_idx = np.argsort(orders)
    orders = orders[sort_idx]
    ratios = ratios[sort_idx]

    peak_idx = int(np.argmax(ratios))
    if peak_idx >= len(orders) - 2:
        return None

    seg_orders = orders[peak_idx:]
    seg_ratios = ratios[peak_idx:]
    total_drop = float(seg_ratios[0] - seg_ratios[-1])
    if total_drop <= max(1.0, 0.05 * float(np.ptp(ratios))):
        return None

    step_widths = np.diff(seg_orders)
    if np.any(step_widths <= 0):
        return None

    drops = -np.diff(seg_ratios) / step_widths
    positive_drops = drops[drops > 0]
    if len(positive_drops) < 2:
        return None

    ratio_span = abs(float(np.nanmax(ratios) - np.nanmin(ratios)))
    min_extra_drop = max(0.08, 0.006 * ratio_span)
    knee_local_idx = None
    knee_strength = 0.0

    for j in range(1, len(drops)):
        previous_declines = drops[:j]
        previous_declines = previous_declines[previous_declines > 0]
        if len(previous_declines) == 0:
            continue

        baseline = float(np.median(previous_declines))
        baseline = max(baseline, 0.05)
        current = float(drops[j])
        next_drop = float(drops[j + 1]) if j + 1 < len(drops) else current
        sustained = min(current, next_drop)
        acceleration = current - baseline

        # Choose the first order where the decline rate leaves the earlier
        # gentle tail, with the next segment confirming it is not a one-bin dip.
        if (
            current >= baseline * 1.15
            and acceleration >= min_extra_drop
            and sustained >= baseline + (min_extra_drop * 0.25)
        ):
            knee_local_idx = j
            knee_strength = acceleration
            break

    if knee_local_idx is None:
        x_span = float(seg_orders[-1] - seg_orders[0])
        if x_span <= 0:
            return None

        x_norm = (seg_orders - seg_orders[0]) / x_span
        y_norm = (seg_ratios - seg_ratios[-1]) / total_drop
        distances = np.abs(x_norm + y_norm - 1.0) / np.sqrt(2.0)
        if len(distances) <= 2:
            return None

        knee_local_idx = int(np.argmax(distances[1:-1]) + 1)
        knee_strength = float(distances[knee_local_idx])
        if knee_strength <= 0.03:
            return None

    knee_idx = peak_idx + knee_local_idx
    return {
        'n': int(round(float(orders[knee_idx]))),
        'ratio': float(ratios[knee_idx]),
        'distance': float(knee_strength),
        'peak_n': int(round(float(orders[peak_idx]))),
        'peak_ratio': float(ratios[peak_idx]),
        'method': 'post-peak first sustained decline acceleration',
    }

def find_eligible_rolloff_knee(orders, ratios):
    """Preserve the normal knee; retry a rejected knee once with gentler slopes."""
    first = find_rolloff_knee(orders, ratios)
    if first is None or first['ratio'] > SFS_ACCEPTABLE_RATIO_DB:
        return first
    pairs = sorted((float(n), float(r)) for n, r in zip(orders, ratios)
                   if np.isfinite(n) and np.isfinite(r))
    ns, rs = np.asarray(pairs).T
    peak = int(np.argmax(rs))
    drops = -np.diff(rs[peak:]) / np.diff(ns[peak:])
    # One bounded sensitivity reduction: 5% rather than 15%, and a quarter
    # of the original absolute acceleration floor. Require two real segments.
    extra = max(0.02, 0.0015 * float(np.ptp(rs)))
    candidates = []
    for j in range(1, len(drops) - 1):
        idx = peak + j
        if ns[idx + 2] >= first['n'] or rs[idx] <= SFS_ACCEPTABLE_RATIO_DB:
            continue
        previous = drops[:j][drops[:j] > 0]
        if not len(previous):
            continue
        baseline = max(float(np.median(previous)), 0.05)
        current, following = drops[j:j+2]
        if (current >= baseline * 1.05 and current - baseline >= extra
                and min(current, following) >= baseline + extra * 0.25):
            candidates.append(dict(
                n=int(ns[idx]), ratio=float(rs[idx]), distance=float(current-baseline),
                peak_n=int(ns[peak]), peak_ratio=float(rs[peak]),
                method='latest earlier sustained decline, gentler second pass',
                initial_knee_n=first['n'], initial_knee_ratio=first['ratio']))
    return candidates[-1] if candidates else None


def _stage3_recommendation_markers(options=None):
    if options and any(key in options for key in STAGE3_CHOICE_STYLES):
        return [(float(options[key]['n']), float(options[key]['ratio']), color, marker, label)
                for key, (color, marker, label) in STAGE3_CHOICE_STYLES.items() if key in options]
    marker_styles = {
        "recommended": ("#e45756", "o", "Recommended"),
    }
    markers = []
    if not options:
        return markers
    for key, (color, marker, label) in marker_styles.items():
        opt = options.get(key)
        if opt and opt.get("n") is not None and opt.get("ratio") is not None:
            markers.append((float(opt["n"]), float(opt["ratio"]), color, marker, label))
    return markers

def plot_internal_degree_power(ax, orders, power_db):
    """Shared diagnostic for the saved plot and interactive Stage 3 dialog."""
    values = np.asarray(power_db, dtype=float)
    # Keep exact zero power visible, with an explicit display floor.
    displayed = np.maximum(values, -80.0)
    ax.plot(orders, displayed, marker="o", color="#f28e2b", label="Degree N / total internal power")
    ax.axhline(-20.0, color="#777777", linestyle="--", label="-20 dB = 1% reference")
    ax.set_xlabel("Order N")
    ax.set_ylabel("Internal power share (dB)")
    ax.set_title("Mean per-frequency power share; display floor -80 dB; gaps = unavailable", fontsize=9)
    ax.set_xticks(orders)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)


def calc_internal_degree_fraction(coeffs, degree):
    """Power share of degree n in this fit, using only outgoing C coefficients.

    Orthonormal Y_nm and outgoing h_n(kr) give radiated power proportional
    to sum(abs(C_nm)**2) / k**2. At one frequency the common factors cancel.
    This is not the difference in total power between independently refit orders.
    Missing/truncated degrees and zero-total fits are undefined, not zero power.
    """
    internal = np.asarray(coeffs)[0::2]
    if degree < 0 or internal.size < (degree + 1)**2:
        return np.nan
    scale = np.max(np.abs(internal))
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    powers = np.abs(internal / scale)**2
    return float(np.sum(powers[degree**2:(degree + 1)**2]) / np.sum(powers))


def aggregate_internal_degree_power(fractions):
    """Equal frequency weighting in linear power share, then convert to dB."""
    valid = np.asarray(fractions, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not valid.size:
        return np.nan, 0
    mean = float(np.mean(valid))
    return (10.0 * np.log10(mean) if mean > 0 else -np.inf), int(valid.size)


def select_tail_reference(orders, ratios):
    finite = [i for i, ratio in enumerate(ratios) if np.isfinite(ratio)]
    if not finite:
        raise ValueError("No finite Int/Ext ratio available for a tail-power reference.")
    qualified = [i for i in finite if ratios[i] > SFS_ACCEPTABLE_RATIO_DB]
    idx = max(qualified, key=lambda i: orders[i]) if qualified else max(finite, key=lambda i: ratios[i])
    return {'n': int(orders[idx]), 'ratio': float(ratios[idx]), 'fallback': not bool(qualified)}


def calc_internal_degree_shares(coeffs, order):
    internal = np.asarray(coeffs)[0::2]
    if internal.size != (order + 1)**2:
        return None
    scale = np.max(np.abs(internal))
    if not np.isfinite(scale) or scale == 0:
        return None
    powers = np.abs(internal / scale)**2
    return np.array([powers[n*n:(n+1)**2].sum() for n in range(order+1)]) / powers.sum()


def calc_cumulative_tail(orders, reference_n, sample_shares):
    valid = [s for s in sample_shares if s is not None]
    values = []
    for n in orders:
        if n > reference_n or not valid:
            values.append(np.nan)
        else:
            db, _ = aggregate_internal_degree_power([np.sum(s[n+1:]) for s in valid])
            values.append(db)
    return values, len(valid)


def plot_internal_tail_power(ax, orders, power_db, reference):
    reference_n = reference['n']
    displayed = (np.where(np.asarray(orders) < reference_n, np.maximum(power_db, -80.0), np.nan)
                 if reference_n is not None else np.full(len(orders), np.nan))
    ax.plot(orders, displayed, marker="o", color="#f28e2b",
            label="_nolegend_")
    ax.axhline(-20, color="#777777", linestyle="--", label="-20 dB = 1%")
    ax.axhline(-30, color="#777777", linestyle="--", alpha=.65, label="-30 dB = 0.1%")
    ax.set_title(TAIL_EXPLAINER, fontsize=12, fontweight='bold', pad=32)
    ax.text(.5, 1.02,
            'Total sound power discarded by stopping at order N, compared with the selected reference order.',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8)
    format_stage3_order_axis(ax)
    ax.set_ylabel("Discarded internal power (dB)")
    ax.set_xticks(orders)
    if len(orders):
        ax.set_xlim(min(orders) - .5, max(orders) + .5)
    ax.grid(True, linestyle="--", alpha=.35)
    ax.legend(loc="best", fontsize=8)


def save_stage3_order_sweep_plot(orders, ratios, residuals=None, options=None, knee=None, save_path=None, internal_degree_power_db=None, internal_tail_power_db=None, tail_reference=None, spl_change=None):
    if not save_path:
        return None

    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        orders_arr = np.asarray(orders, dtype=float)
        ratios_arr = np.asarray(ratios, dtype=float)

        has_power = internal_tail_power_db is not None or internal_degree_power_db is not None
        fig = Figure(figsize=(14, 9) if spl_change is not None else (10, 8 if has_power else 5))
        FigureCanvasAgg(fig)
        ax_ratio = fig.add_subplot(221 if spl_change is not None else (211 if has_power else 111))
        if internal_tail_power_db is not None:
            ax_tail = fig.add_subplot(223 if spl_change is not None else 212, sharex=ax_ratio)
            plot_internal_tail_power(ax_tail, orders_arr, internal_tail_power_db, tail_reference)
            highlight_stage3_choices(ax_tail, options or {}, tail=True)
            ax_tail.legend(loc='best', fontsize=8)
        elif internal_degree_power_db is not None:
            plot_internal_degree_power(fig.add_subplot(223 if spl_change is not None else 212, sharex=ax_ratio), orders_arr, internal_degree_power_db)
        elif spl_change is not None:
            ax_tail = fig.add_subplot(223, sharex=ax_ratio)
            ax_tail.set_title('Sound Power Discarded', fontweight='bold')
            format_stage3_order_axis(ax_tail)
            ax_tail.set_facecolor('#eeeeee')
            ax_tail.text(.5, .5, 'Unavailable for order selection\nNo usable sound power reference',
                         transform=ax_tail.transAxes, ha='center', va='center', color='#666666')
        ax_ratio.plot(orders_arr, ratios_arr, marker="o", linewidth=1.5, color="#4c78a8", label="_nolegend_")
        ax_ratio.axhline(
            SFS_ACCEPTABLE_RATIO_DB,
            color="#59a14f",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="20 dB quality threshold",
        )
        ax_ratio.set_xlabel("Order N")
        ax_ratio.set_ylabel("Int/Ext ratio (dB)")
        ax_ratio.grid(True, linestyle="--", alpha=0.35)
        ax_ratio.set_xticks(orders_arr)
        ax_ratio.set_xlim(min(orders_arr) - .5, max(orders_arr) + .5)

        highlight_stage3_choices(ax_ratio, options or {})

        lines = ax_ratio.get_lines() + ax_ratio.collections
        labels = [line.get_label() for line in lines if not line.get_label().startswith("_")]
        visible_lines = [line for line in lines if not line.get_label().startswith("_")]
        ax_ratio.legend(visible_lines, labels, loc="best")
        format_stage3_ratio_axis(ax_ratio, inspection_only=bool(tail_reference and tail_reference.get('tail_only')))
        if spl_change is not None:
            from stage3_spl_change import plot_spl_changes
            plot_spl_changes(fig.add_subplot(222), spl_change)
            notes = fig.add_subplot(224)
            notes.axis('off')
            recommendation = recommended_stage3_choice(options or {})
            lines = ['Recommended orders']
            for key in ('knee', 'tail', 'spl'):
                option = (options or {}).get(key)
                if option:
                    prefix = 'Recommended' if key == recommendation else 'Back-up recommendation'
                    lines.append(f"\n{prefix}: {STAGE3_CHOICE_STYLES[key][2]}\n"
                                 f"Order {option['n']} - Int / Ext Ratio: {option['ratio']:.2f} dB\n"
                                 f"Sound Power Discarded: {format_tail_power(option.get('tail_db', np.nan))}")
            notes.text(0, 1, '\n'.join(lines), va='top', fontsize=10)
            band = spl_change.get('test_band_hz')
            title = 'Stage 3 - Recommended Max Order for Stage 4 Solve'
            if band:
                title += f'\nTest range: {band[0]:g} - {band[1]:g} Hz'
            fig.suptitle(title, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, .93) if spl_change is not None else (0, 0, 1, 1))
        fig.savefig(save_path, dpi=150)
        return save_path
    except Exception as e:
        print(f"Warning: Failed to save Stage 3 order sweep plot: {e}")
        return None

def calc_internal_external_ratio(coeffs):
    eps = np.finfo(float).eps
    C_coeffs = coeffs[0::2]  
    D_coeffs = coeffs[1::2]  
    pwr_C = np.sum(np.abs(C_coeffs)**2)
    pwr_D = np.sum(np.abs(D_coeffs)**2)
    return 10 * np.log10((pwr_C + eps) / (pwr_D + eps))

def _worker(args):
    f, Pk, r_k, th_k, ph_k, N, st_db, mx_db, lam, base_norm, c_sound, kr_offset = args
    N_kr = get_kr_limit(f, r_k, c_sound, kr_offset)
    safe_N = min(N, N_kr)
    coeffs, metrics = _solve_one_frequency(
        f_hz=f, P_complex=Pk, coords_sph=(r_k, th_k, ph_k), order_N=safe_N,
        k_val=2.0 * np.pi * f / c_sound,
        CONDITION_METRICS=True, noise_floor_start_db=st_db,
        noise_floor_max_db=mx_db, max_lambda=lam
    )
    ratio_db = calc_internal_external_ratio(coeffs)
    err = (metrics['residual_norm'] / max(base_norm, 1e-20)) * 100.0
    fraction = calc_internal_degree_fraction(coeffs, N)
    return {'N': N, 'st_db': st_db, 'mx_db': mx_db, 'lam': lam, 'ratio_db': ratio_db, 'err': err,
            'internal_degree_fraction': fraction, 'internal_degree_shares': calc_internal_degree_shares(coeffs, N),
            'limited_degree_shares': calc_internal_degree_shares(coeffs, safe_N),
            'internal_coeffs': coeffs[0::2],
            'usable': coeffs.size == 2*(safe_N+1)**2 and bool(np.all(np.isfinite(coeffs[0::2])))}

def _stage3_thread_workers(task_count):
    cpu_count = os.cpu_count() or 1
    return max(1, min(task_count, cpu_count))


def _collect_with_progress(results, total, label, item_name='tasks', weights=None):
    """Collect an ordered map iterator with an in-place progress counter."""
    collected = []
    completed = 0
    report_step = max(1, int(np.ceil(total / 10)))
    next_report = report_step
    sys.stdout.write(f"\r{label}: 0 of {total} {item_name}")
    sys.stdout.flush()
    for index, result in enumerate(results):
        collected.append(result)
        completed += weights[index] if weights is not None else 1
        if completed >= next_report or completed >= total:
            sys.stdout.write(f"\r{label}: {completed} of {total} {item_name}")
            sys.stdout.flush()
            while next_report <= completed:
                next_report += report_step
    sys.stdout.write("\n")
    sys.stdout.flush()
    return collected


def select_stage3_frequency_indices(freqs, start_hz, end_hz, octave_resolution=12):
    """Nearest available bins to 1/x-octave targets; 0 selects every bin.

    Include both available range endpoints and deduplicate mapped FFT bins.
    """
    if not np.isfinite(octave_resolution) or octave_resolution < 0 or int(octave_resolution) != octave_resolution:
        raise ValueError("Stage 3 octave resolution must be a nonnegative integer (0 = all bins).")
    if not np.isfinite(start_hz) or not np.isfinite(end_hz):
        raise ValueError("Stage 3 frequency boundaries must be finite.")
    freqs = np.asarray(freqs, dtype=float)
    low, high = sorted((start_hz, end_hz))
    indices = np.flatnonzero((freqs >= low) & (freqs <= high) & (freqs > 0))
    if not indices.size:
        raise ValueError(f"Stage 3 frequency range {low:g}-{high:g} Hz contains no positive frequency bins.")
    if octave_resolution == 0 or indices.size == 1:
        return indices
    available = freqs[indices]
    # Stream targets to avoid a large temporary array at fine resolutions.
    selected = {0, len(available) - 1}
    steps = int(np.floor(np.log2(available[-1] / available[0]) * octave_resolution))
    for step in range(1, steps + 1):
        target = available[0] * 2.0 ** (step / octave_resolution)
        right = min(int(np.searchsorted(available, target)), len(available) - 1)
        left = max(0, right - 1)
        selected.add(left if target - available[left] <= available[right] - target else right)
        if len(selected) == len(available):
            break
    return indices[sorted(selected)]

def run_open_branch_optimizer(
    input_dir_opti: str,
    input_filename_opti: str,
    test_order_range: tuple,
    freq_start_hz: float = 10000.0,
    freq_end_hz: float = 20000.0,
    use_optimized_origins: bool = False,
    kr_offset: float = 2.0,
    speed_of_sound: float = 343.0,
    save_plot: bool = True,
    plot_save_path: str = None,
    use_process_pool: bool = True,
    octave_resolution: int = 12,
    spl_change_enabled: bool = True,
    spl_sphere_points: int = 1000,
    spl_radius_m: float = 1.0,
    spl_floor_db: float = -40.0,
    process_pool=None,
):
    start_time = time.time()
    # Legacy keyword arguments remain accepted for callers loading old settings,
    # but Stage 3 now always evaluates SPL at the fixed assessment floor.
    spl_change_enabled = True
    spl_floor_db = -40.0
    if spl_change_enabled:
        from stage3_spl_change import evaluation_sphere, evaluate_frequency_changes, summarize_changes, save_spl_changes
        evaluation_sphere(spl_sphere_points, spl_radius_m)
        if not np.isfinite(spl_floor_db) or spl_floor_db >= 0:
            raise ValueError('SPL-change floor must be finite and negative.')

    input_path = os.path.join(input_dir_opti, input_filename_opti)
    parsed_data = load_and_parse_npz(input_path)
    saved_speed = parsed_data.get('speed_of_sound_mps')
    if saved_speed is not None:
        speed_of_sound = float(saved_speed)
        print(f"Using Stage 2 speed of sound from NPZ: {speed_of_sound:g} m/s")
    f_all = parsed_data['freqs']
    data_dict = parsed_data['complex_data']
    filenames = parsed_data['filenames']
    r_static = parsed_data['r_arr']
    th_static = parsed_data['th_arr']
    ph_static = parsed_data['ph_arr']
    
    N_grid, M_unique = get_grid_limit(th_static, ph_static)
    print(f"Data Loaded. Unique observation points: {M_unique} (Maximum N_grid limit: {N_grid})")

    use_opt = use_optimized_origins
    origins_mm = parsed_data['origins_mm'] if (use_opt and parsed_data['origins_mm'] is not None) else np.zeros((len(f_all), 3))

    freq_start_hz, freq_end_hz, tail_only = stage3_test_band(
        freq_start_hz, freq_end_hz, parsed_data.get('stage3_rft_lower_hz'))
    below_rft = tail_only
    mode_note = ''
    if tail_only:
        mode_note = (f'Upper limit is below the RFT boundary. Int/Ext is shown for inspection, not used for selection; '
                     f'test band set to {freq_start_hz:g}-{freq_end_hz:g} Hz (two octaves). '
                     'Solve order is selected directly from the simple SPL-change rule. '
                     'Tail power is not used for selection; this is not a stability guarantee.')
        print(mode_note)

    idx_target = np.where((f_all >= freq_start_hz) & (f_all <= freq_end_hz))[0]
    if len(idx_target) == 0:
        raise ValueError(
            f"Stage 3 frequency range {freq_start_hz:g}-{freq_end_hz:g} Hz "
            f"does not contain any frequency bins from {f_all[0]:g}-{f_all[-1]:g} Hz."
        )
    test_indices = select_stage3_frequency_indices(f_all, freq_start_hz, freq_end_hz, octave_resolution)
    resolution_label = "all bins" if octave_resolution == 0 else f"1/{octave_resolution}-octave"
    print(f"Stage 3 order test frequency range: {freq_start_hz:g}-{freq_end_hz:g} Hz ({len(test_indices)} sample frequencies, {resolution_label})")

    fit_records = {}

    def run_batch(configs):
        tasks = []
        for k in test_indices:
            f = f_all[k]
            Pk = np.array([data_dict[fn][k] for fn in filenames])
            origin_m = origins_mm[k] / 1000.0
            r_k, th_k, ph_k = translate_coordinates(r_static, th_static, ph_static, origin_m)
            base_norm = np.linalg.norm(Pk)
            for N, st_db, mx_db, lam in configs:
                tasks.append((f, Pk, r_k, th_k, ph_k, N, st_db, mx_db, lam, base_norm, speed_of_sound, kr_offset))
        backend = "process" if use_process_pool else "thread"
        max_workers = None if use_process_pool else _stage3_thread_workers(len(tasks))
        worker_msg = "" if max_workers is None else f", {max_workers} workers"
        print(f"Stage 3 parallel backend: {backend} pool ({len(tasks)} tasks{worker_msg})")

        if use_process_pool and process_pool is not None:
            raw_results = _collect_with_progress(
                process_pool.map(_worker, tasks), len(tasks), 'Stage 3 solve progress')
        elif use_process_pool:
            ctx = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(max_workers=6, mp_context=ctx) as ex:
                raw_results = _collect_with_progress(
                    ex.map(_worker, tasks), len(tasks), 'Stage 3 solve progress')
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                raw_results = _collect_with_progress(
                    ex.map(_worker, tasks), len(tasks), 'Stage 3 solve progress')
            
        agg = {}
        for task, r in zip(tasks, raw_results):
            if spl_change_enabled:
                fit_records[(r['N'], float(task[0]))] = r if 'internal_coeffs' in r else None
            key = (r['N'], r['st_db'], r['mx_db'], r['lam'])
            if key not in agg: agg[key] = {'ratio': [], 'err': [], 'degree_fraction': [], 'shares': [], 'limited_shares': []}
            agg[key]['ratio'].append(r['ratio_db'])
            agg[key]['err'].append(r['err'])
            agg[key]['degree_fraction'].append(r['internal_degree_fraction'])
            shares = r['internal_degree_shares']
            limited_shares = r.get('limited_degree_shares', shares)
            if limited_shares is not None:
                limited_shares = np.pad(limited_shares, (0, r['N'] + 1 - len(limited_shares)))
            agg[key]['limited_shares'].append(limited_shares)
            agg[key]['shares'].append(shares)
            
        results = {}
        for key, values in agg.items():
            power_db, count = aggregate_internal_degree_power(values['degree_fraction'])
            results[key] = {'ratio_db': np.mean(values['ratio']), 'err': np.mean(values['err']),
                            'internal_degree_power_db': power_db, 'internal_degree_sample_count': count,
                            'internal_degree_shares': values['shares'],
                            'limited_degree_shares': values['limited_shares']}
        return results

    # =========================================================================
    # STEP 1: FINDING THE STABLE ORDER (Maximum Ratio Detection)
    # =========================================================================
    print("\n" + "="*65)
    print(" STEP 1: FINDING THE STABLE ORDER (UNREGULARIZED)")
    print("="*65)
    min_n, max_n = test_order_range
    
    # Hardcap to N_grid
    max_n = min(max_n, N_grid)
    min_n = min(max(2, min_n), max_n)
    
    orders = list(range(min_n, max_n + 1))
    # Solve the N-1 baseline in the same pool, without adding it to sweep candidates.
    baseline_n = orders[0] - 1
    solve_orders = [baseline_n] + orders if spl_change_enabled and baseline_n >= 0 else orders
    configs_s1 = [(n, -50.0, -70.0, 0.0) for n in solve_orders]
    res_s1 = run_batch(configs_s1)
    print(f"{'Order N':<10} | {'Int/Ext Ratio (dB)':<20} | {'Residual %':<12} | {'Delta Ratio'}")
    print("-" * 65)
    
    # Store data for stable N calculation
    n_vals = []
    ratio_vals = []
    err_vals = []
    delta_vals = []
    degree_power_vals = []
    degree_sample_counts = []
    prev_ratio = None
    
    for n in orders:
        data = res_s1[(n, -50.0, -70.0, 0.0)]
        ratio = data['ratio_db']
        delta = (ratio - prev_ratio) if prev_ratio is not None else 0.0
        
        n_vals.append(n)
        ratio_vals.append(ratio)
        err_vals.append(data['err'])
        delta_vals.append(delta)
        degree_power_vals.append(data['internal_degree_power_db'])
        degree_sample_counts.append(data['internal_degree_sample_count'])
        prev_ratio = ratio
        
        print(f"{n:<10} | {ratio:<20.2f} | {data['err']:<12.2f} | {delta:<10.2f}")
        print(f"           Degree N internal power share: {data['internal_degree_power_db']:.2f} dB "
              f"({data['internal_degree_sample_count']}/{len(test_indices)} frequencies)")

    best_sfs_idx = int(np.argmax(ratio_vals))
    best_sfs_N = n_vals[best_sfs_idx]
    best_sfs_ratio = ratio_vals[best_sfs_idx]
    best_sfs_err = err_vals[best_sfs_idx]
    separation_available = any(np.isfinite(r) and r > SFS_ACCEPTABLE_RATIO_DB for r in ratio_vals)
    tail_only = below_rft or not separation_available
    ratio_note = ''
    if tail_only:
        ratio_note = ('Int/Ext shown for inspection; below RFT, not used for selection.' if below_rft
                      else 'Int/Ext shown for inspection; no order exceeds 20 dB, not used for selection.')
        if not below_rft:
            mode_note = 'Using the simple SPL-change rule to choose the solve order directly. The test band is unchanged; separation is not established.'
        if not separation_available:
            mode_note = NO_SEPARATION_NOTE + ' ' + mode_note
        for data in res_s1.values():
            data['internal_degree_shares'] = data['limited_degree_shares']
    rolloff_knee = None if tail_only else find_eligible_rolloff_knee(n_vals, ratio_vals)
    if rolloff_knee and 'initial_knee_n' in rolloff_knee:
        print(f"=> Initial knee N={rolloff_knee['initial_knee_n']} was at or below 20 dB; "
              f"gentler earlier knee selected at N={rolloff_knee['n']} ({rolloff_knee['ratio']:.2f} dB)")
    if not tail_only and best_sfs_ratio <= 0:
        print("WARNING: No positive Int/Ext ratio was found. Sound field separation may not be ideal; re-consider measurement settings.")
    elif not tail_only and best_sfs_ratio <= SFS_ACCEPTABLE_RATIO_DB:
        print('WARNING: ' + NO_SEPARATION_NOTE)

    print("-" * 65)
    print(f"=> Order N with best SFS and solve stability: N={best_sfs_N} (Ratio: {best_sfs_ratio:.2f} dB)")
    if rolloff_knee:
        print(f"=> Order N at roll-off start: N={rolloff_knee['n']} (Ratio: {rolloff_knee['ratio']:.2f} dB)")
    else:
        print("=> Order N at roll-off start: not detected")

    spl_change = None
    if spl_change_enabled:
        print(f"Evaluating internal SPL changes on a {spl_radius_m:g} m sphere ({spl_sphere_points} points)...")
        evaluation_tasks = []
        for k in test_indices:
            f = float(f_all[k])
            fits = {n: fit_records.get((n, f)) for n in [baseline_n] + orders}
            evaluation_tasks.append((f, orders, fits, speed_of_sound, origins_mm[k]/1000,
                                     spl_sphere_points, spl_radius_m, spl_floor_db))
        executor = concurrent.futures.ProcessPoolExecutor if use_process_pool else concurrent.futures.ThreadPoolExecutor
        # Limit evaluation concurrency: each worker builds its own spherical basis.
        kwargs = {'max_workers': min(4, len(evaluation_tasks), os.cpu_count() or 1)}
        if use_process_pool:
            kwargs['mp_context'] = multiprocessing.get_context('spawn')
        if use_process_pool and process_pool is not None:
            from stage3_spl_change import evaluate_frequency_chunk
            chunks = [evaluation_tasks[i::kwargs['max_workers']] for i in range(kwargs['max_workers'])]
            completed_chunks = _collect_with_progress(
                process_pool.map(evaluate_frequency_chunk, chunks), len(evaluation_tasks),
                'Stage 3 directivity progress', item_name='frequencies',
                weights=[len(chunk) for chunk in chunks])
            frequency_results = [result for chunk in completed_chunks for result in chunk]
        else:
            with executor(**kwargs) as ex:
                frequency_results = _collect_with_progress(
                    ex.map(evaluate_frequency_changes, evaluation_tasks), len(evaluation_tasks),
                    'Stage 3 directivity progress', item_name='frequencies')
        spl_change = summarize_changes(orders, frequency_results, spl_sphere_points, spl_radius_m, spl_floor_db, len(test_indices))
        spl_change['sample_frequencies_hz'] = f_all[test_indices].tolist()
        spl_change['test_band_hz'] = [freq_start_hz, freq_end_hz]
        fit_records.clear()
        print("Sphere directivity-change evaluation complete (initial order sweep).")

    tail_reference = (dict(n=None, ratio=None, fallback=True, tail_only=True, method='Tail power not used in SPL selection mode')
                      if tail_only else select_tail_reference(n_vals, ratio_vals))
    tail_reference['ratio_note'] = ratio_note
    if tail_reference['n'] is not None:
        reference_data = res_s1[(tail_reference['n'], -50.0, -70.0, 0.0)]
        tail_power_vals, tail_count = calc_cumulative_tail(n_vals, tail_reference['n'], reference_data['internal_degree_shares'])
    else:
        tail_power_vals, tail_count = [np.nan] * len(n_vals), 0
    tail_reference['sample_count'] = tail_count
    tail_by_reference = {}
    for n, ratio in zip(n_vals, ratio_vals):
        sample_shares = res_s1[(n, -50.0, -70.0, 0.0)]['internal_degree_shares']
        values, count = calc_cumulative_tail(n_vals, n, sample_shares)
        tail_by_reference[str(n)] = {'n': n, 'ratio': ratio, 'sample_count': count, 'power_db': values}
    print(f"Cumulative tail reference: N={tail_reference['n']} ({tail_count}/{len(test_indices)} frequencies); "
          + tail_reference.get('method', 'best-ratio fallback' if tail_reference['fallback'] else 'highest order >20 dB'))
    for n, db in zip(n_vals, tail_power_vals):
        print(f"  Stop at N={n}: discarded internal tail {db:.2f} dB")

    options = stage3_order_choices(n_vals, ratio_vals, err_vals, rolloff_knee, tail_power_vals, tail_reference['n'], tail_only=tail_only)
    spl_options, spl_reason = select_spl_order_choices(n_vals, ratio_vals, err_vals, spl_change)
    spl_option = spl_options.get('spl')
    spl_candidate_note = (spl_reason
                          if not spl_option else
                          f"Directivity-change candidate N={spl_option['n']} does not exceed 20 dB Int/Ext separation.")
    if spl_option and (tail_only or spl_option['ratio'] > SFS_ACCEPTABLE_RATIO_DB):
        if tail_reference['n'] is not None and spl_option['n'] < tail_reference['n']:
            spl_option['tail_db'] = tail_power_vals[n_vals.index(spl_option['n'])]
        options['spl'] = spl_option
    if spl_change is not None:
        spl_change['order_choices'] = {
            key: dict(n=v['n'], spl_db=v['spl_db'], label=v['label'],
                      near_minimum_n=v['near_minimum_n'], backoff_orders=v['backoff_orders'],
                      convergence_note=v['convergence_note'])
            for key, v in options.items() if key == 'spl'}
    chosen_key = recommended_stage3_choice(options)

    recommended = options[chosen_key] if chosen_key else None
    recommended_N = recommended['n'] if recommended else None
    recommendation_reason = ('The highest eligible order among the Int/Ext, sound-power and directivity-change suggestions is recommended; each exceeds 20 dB separation.'
                             if recommended else 'No candidate exceeds 20 dB separation. Adjust the test frequency range or check windowing, measurement quality and grid coverage; re-measure if needed.')
    if tail_only:
        recommendation_reason = spl_reason
    print("\n" + "="*65)
    print(" FINAL ORDER N RECOMMENDATION")
    print("="*65)
    print(f"Recommended Order N: {recommended_N}")
    print(recommendation_reason)
    print(f"Rule of thumb: Int/Ext SFS ratio greater than {SFS_ACCEPTABLE_RATIO_DB:.0f} dB has been found to produce acceptable results.")
    print("="*65)

    warning = ""
    if not recommended:
        warning = recommendation_reason
    if not tail_only and best_sfs_ratio <= SFS_ACCEPTABLE_RATIO_DB:
        warning = NO_SEPARATION_NOTE
    qualifying_orders = [n for n, r in zip(n_vals, ratio_vals) if np.isfinite(r) and r > SFS_ACCEPTABLE_RATIO_DB]
    if not tail_only and qualifying_orders and max(qualifying_orders) < 4:
        warning += ' Only very low orders achieved acceptable separation. This may be reasonable for a subwoofer. For other drivers, check measurement-point count, spatial coverage, positioning accuracy and the test band.'

    if save_plot and plot_save_path is None:
        stem = os.path.splitext(input_filename_opti)[0]
        plot_save_path = os.path.join(input_dir_opti, f"{stem}_stage3_order_sweep.png")
    saved_plot = save_stage3_order_sweep_plot(
        n_vals,
        ratio_vals,
        err_vals,
        options=options,
        knee=rolloff_knee,
        save_path=plot_save_path if save_plot else None,
        internal_tail_power_db=None if tail_only else tail_power_vals,
        tail_reference=tail_reference,
        spl_change=spl_change,
    )
    if saved_plot:
        print(f"Stage 3 order sweep plot saved to: {saved_plot}")
    if spl_change is not None and saved_plot:
        spl_change['plot_path'] = saved_plot

    elapsed = time.time() - start_time
    print(f"\nStage 3 processing completed in {elapsed:.2f} seconds.")

    return {
        'options': {
            **options,
        },
        'recommended_key': chosen_key,
        'spl_candidate_note': '' if 'spl' in options else spl_candidate_note,
        'recommendation_note': recommendation_reason,
        'sfs_ratio_rule_db': SFS_ACCEPTABLE_RATIO_DB,
        'step1': {
            'orders': n_vals,
            'ratios': ratio_vals,
            'residuals': err_vals,
            'delta_ratios': delta_vals,
            'internal_degree_power_db': degree_power_vals,
            'internal_tail_power_db': tail_power_vals,
            'tail_reference': tail_reference,
            'tail_by_reference': tail_by_reference,
            'internal_degree_sample_counts': degree_sample_counts,
            'sample_frequency_count': len(test_indices),
            'sample_frequencies_hz': f_all[test_indices].tolist(),
            'octave_resolution': octave_resolution,
            'rolloff_knee': rolloff_knee,
        },
        'warning': warning,
        'tail_only': tail_only,
        'below_rft': below_rft,
        'ratio_note': ratio_note,
        'mode_note': mode_note,
        'test_band_hz': [freq_start_hz, freq_end_hz],
        'plot_path': saved_plot,
        'spl_change': spl_change,
    }

def main():
    try:
        import config_process
    except ImportError:
        sys.exit("Error: Could not import config_process.py")

    try:
        from config_process import KR_OFFSET
    except ImportError:
        KR_OFFSET = 2.0
        
    run_open_branch_optimizer(
        input_dir_opti=config_process.INPUT_DIR_OPTI,
        input_filename_opti=config_process.INPUT_FILENAME_OPTI,
        test_order_range=config_process.TEST_ORDER_RANGE,
        use_optimized_origins=getattr(config_process, 'USE_OPTIMIZED_ORIGINS', False),
        speed_of_sound=getattr(config_process, 'SPEED_OF_SOUND', 343.0),
        kr_offset=KR_OFFSET,
        octave_resolution=getattr(config_process, 'TEST_OCTAVE_RESOLUTION', 12),
        spl_sphere_points=getattr(config_process, 'TEST_SPL_SPHERE_POINTS', 1000),
        spl_radius_m=getattr(config_process, 'TEST_SPL_RADIUS_M', 1.0),
    )

if __name__ == "__main__":
    main()
