#!/usr/bin/env python3
"""
Open-Branch Spherical Energy Optimizer
======================================
Step 1: Finds the Tipping Point N. Keeps (N) and (N-1) alive.
Step 2: Sweeps heavy damping to find the plateau/noise floor for both.
Step 3: Sweeps lambdas AND threshold brackets for both orders to find 
        the absolute best global configuration.

Update: Steps 2 and 3 were found ineffective in practice and are retained
below only as legacy reference code. The active Stage 3 optimizer now runs
only the Step 1 order-N test. Stage 4 owns the regularization defaults.
"""

import os
import sys
import time

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
# DEBUG/TESTING SETTINGS
# =============================================================================
# Set to an integer to force Step 2 to use a specific order_N, e.g., 13.
MANUAL_STEP2_ORDER = None

FIXED_NOISE_FLOOR_START_DB = -30.0
FIXED_NOISE_FLOOR_MAX_DB = -40.0
FIXED_MAX_LAMBDA = 0.000001
SFS_ACCEPTABLE_RATIO_DB = 20.0
STAGE3_CHOICE_STYLES = {
    'knee': ('#e45756', 'o', 'Roll-off knee'),
    'highest': ('#9467bd', 's', 'Highest order >20 dB'),
    'tail': ('#f28e2b', '^', 'First tail < -20 dB'),
}
STAGE3_CHOICE_HELP = """Choosing an order for Stage 4

Order N controls how much spatial detail the model can describe. More detail is
useful only while the separation remains reliable.

Top graph: internal/external (Int/Ext) ratio
The internal field represents the source; the external field represents sound
arriving from outside. In the tested frequency region, time windowing is
responsible for removing reflections. With effective windowing, the ratio should
therefore be large, with little energy assigned to the external field. A falling
ratio suggests poorer separation. 20 dB is a practical rule of thumb, not a
guarantee of accuracy or a direct measurement of numerical conditioning.

Bottom graph: cumulative tail power
This shows how much modeled internal sound power you throw away by stopping at
each order. It combines ALL degrees above that order in the selected reference
fit. -20 dB means 1% discarded; -30 dB means 0.1%. More negative means less lost.
Adding a tail below -20 dB may not be worthwhile if it degrades the Int/Ext ratio.
The reference defaults to the highest order above 20 dB. The dropdown changes
the reference and the tail-based choice. Beyond the reference, power is unknown;
the zero tail at the reference itself is omitted and is not a threshold crossing.
These are frequency-averaged modeled powers: small contributions may still matter
to directivity, and narrow-band features can be diluted by averaging.

The three choices
Roll-off knee: where the post-peak ratio starts declining more sharply. It aims
to keep detail before separation worsens. Knee detection seems sensitive to the
specific rate and shape of roll-off, so it may not be robust or even find a knee.

Highest order above 20 dB: keeps the most detail while meeting the separation
rule of thumb. If no tested order exceeds 20 dB, this choice is unavailable;
the best-ratio order remains available as a fallback, with a warning.

First tail below -20 dB: the lowest tested order below the reference that discards
less than 1% of its modeled internal power. Check its Int/Ext ratio too: this
choice does not guarantee good separation. It is unavailable if no such order
is found before the reference.

Select a choice, then click Use in Stage 4. The reference dropdown changes the
tail diagnostic; the radio buttons choose the actual order sent to Stage 4.
"""


def stage3_order_choices(orders, ratios, residuals, knee, tail_db, reference_n):
    selected = {'knee': knee['n'] if knee else None, 'highest': None, 'tail': None}
    qualified = [n for n, r in zip(orders, ratios) if np.isfinite(r) and r > SFS_ACCEPTABLE_RATIO_DB]
    selected['highest'] = max(qualified) if qualified else None
    tails = [n for n, db in zip(orders, tail_db) if n < reference_n and not np.isnan(db) and db < -20]
    selected['tail'] = min(tails) if tails else None
    result = {}
    for key, n in selected.items():
        if n is None:
            continue
        idx = list(orders).index(n)
        result[key] = {'n': n, 'ratio': ratios[idx], 'err': residuals[idx],
                       'label': STAGE3_CHOICE_STYLES[key][2]}
    return result


def highlight_stage3_choices(ax, options, tail=False):
    artists = []
    for key, (color, marker, label) in STAGE3_CHOICE_STYLES.items():
        opt = options.get(key)
        if opt:
            if tail:
                artists.append(ax.axvline(opt['n'], color=color, linestyle='--', alpha=.7,
                                          label=f"{label}: N={opt['n']}"))
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
    displayed = np.where(np.asarray(orders) < reference['n'], np.maximum(power_db, -80.0), np.nan)
    ax.plot(orders, displayed, marker="o", color="#f28e2b",
            label="Power in degrees above N / reference internal power")
    ax.axhline(-20, color="#777777", linestyle="--", label="-20 dB = 1%")
    ax.axvline(reference['n'], color="#9467bd", linestyle=":", label=f"Reference N={reference['n']}")
    if reference.get('manual'):
        qualifier = "manual selection" + ("; below >20 dB threshold" if reference['ratio'] <= SFS_ACCEPTABLE_RATIO_DB else "")
    else:
        qualifier = "best-ratio fallback; no fit >20 dB" if reference['fallback'] else "highest order >20 dB"
    ax.set_title(f"Reference N={reference['n']}: {reference['ratio']:.2f} dB Int/Ext ({qualifier})\n"
                 "Mean tail fraction; tail is zero at reference (omitted); beyond reference unknown", fontsize=8)
    ax.set_xlabel("Stopping order N")
    ax.set_ylabel("Discarded internal power (dB)")
    ax.set_xticks(orders)
    ax.grid(True, linestyle="--", alpha=.35)
    ax.legend(loc="best", fontsize=8)


def save_stage3_order_sweep_plot(orders, ratios, residuals=None, options=None, knee=None, save_path=None, internal_degree_power_db=None, internal_tail_power_db=None, tail_reference=None):
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
        fig = Figure(figsize=(10, 8 if has_power else 5))
        FigureCanvasAgg(fig)
        ax_ratio = fig.add_subplot(211 if has_power else 111)
        if internal_tail_power_db is not None:
            ax_tail = fig.add_subplot(212, sharex=ax_ratio)
            plot_internal_tail_power(ax_tail, orders_arr, internal_tail_power_db, tail_reference)
            highlight_stage3_choices(ax_tail, options or {}, tail=True)
            ax_tail.legend(loc='best', fontsize=8)
        elif internal_degree_power_db is not None:
            plot_internal_degree_power(fig.add_subplot(212, sharex=ax_ratio), orders_arr, internal_degree_power_db)
        ax_ratio.plot(orders_arr, ratios_arr, marker="o", linewidth=1.5, color="#4c78a8", label="Int/Ext ratio")
        ax_ratio.axhline(
            SFS_ACCEPTABLE_RATIO_DB,
            color="#59a14f",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="20 dB acceptable SFS rule of thumb",
        )
        ax_ratio.set_xlabel("Order N")
        ax_ratio.set_ylabel("Int/Ext ratio (dB)")
        ax_ratio.grid(True, linestyle="--", alpha=0.35)
        ax_ratio.set_xticks(orders_arr)

        for x, y, color, marker, label in _stage3_recommendation_markers(options):
            ax_ratio.scatter([x], [y], s=95, color=color, marker=marker, edgecolors="white", linewidths=1.1, zorder=5, label=label)

        lines = ax_ratio.get_lines() + ax_ratio.collections
        labels = [line.get_label() for line in lines if not line.get_label().startswith("_")]
        visible_lines = [line for line in lines if not line.get_label().startswith("_")]
        ax_ratio.legend(visible_lines, labels, loc="best")
        ax_ratio.set_title("Stage 3 Order Sweep: Int/Ext Ratio vs Order N")
        fig.tight_layout()
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
            'internal_degree_fraction': fraction, 'internal_degree_shares': calc_internal_degree_shares(coeffs, N)}

def _stage3_thread_workers(task_count):
    cpu_count = os.cpu_count() or 1
    return max(1, min(task_count, cpu_count))


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
    test_start_db_range: tuple,
    test_lambda_range: tuple,
    test_db_transition_span: float,
    freq_start_hz: float = 10000.0,
    freq_end_hz: float = 20000.0,
    use_optimized_origins: bool = False,
    kr_offset: float = 2.0,
    speed_of_sound: float = 343.0,
    save_plot: bool = True,
    plot_save_path: str = None,
    use_process_pool: bool = True,
    octave_resolution: int = 12,
):
    start_time = time.time()

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

    if freq_start_hz > freq_end_hz:
        freq_start_hz, freq_end_hz = freq_end_hz, freq_start_hz

    idx_target = np.where((f_all >= freq_start_hz) & (f_all <= freq_end_hz))[0]
    if len(idx_target) == 0:
        raise ValueError(
            f"Stage 3 frequency range {freq_start_hz:g}-{freq_end_hz:g} Hz "
            f"does not contain any frequency bins from {f_all[0]:g}-{f_all[-1]:g} Hz."
        )
    test_indices = select_stage3_frequency_indices(f_all, freq_start_hz, freq_end_hz, octave_resolution)
    resolution_label = "all bins" if octave_resolution == 0 else f"1/{octave_resolution}-octave"
    print(f"Stage 3 order test frequency range: {freq_start_hz:g}-{freq_end_hz:g} Hz ({len(test_indices)} sample frequencies, {resolution_label})")

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

        if use_process_pool:
            ctx = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(mp_context=ctx) as ex:
                raw_results = list(ex.map(_worker, tasks))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                raw_results = list(ex.map(_worker, tasks))
            
        agg = {}
        for r in raw_results:
            key = (r['N'], r['st_db'], r['mx_db'], r['lam'])
            if key not in agg: agg[key] = {'ratio': [], 'err': [], 'degree_fraction': [], 'shares': []}
            agg[key]['ratio'].append(r['ratio_db'])
            agg[key]['err'].append(r['err'])
            agg[key]['degree_fraction'].append(r['internal_degree_fraction'])
            agg[key]['shares'].append(r['internal_degree_shares'])
            
        results = {}
        for key, values in agg.items():
            power_db, count = aggregate_internal_degree_power(values['degree_fraction'])
            results[key] = {'ratio_db': np.mean(values['ratio']), 'err': np.mean(values['err']),
                            'internal_degree_power_db': power_db, 'internal_degree_sample_count': count,
                            'internal_degree_shares': values['shares']}
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
    configs_s1 = [(n, -50.0, -50.0 - test_db_transition_span, 1e-10) for n in orders]
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
        data = res_s1[(n, -50.0, -50.0 - test_db_transition_span, 1e-10)]
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
    rolloff_knee = find_rolloff_knee(n_vals, ratio_vals)
    if best_sfs_ratio <= 0:
        print("WARNING: No positive Int/Ext ratio was found. Sound field separation may not be ideal; re-consider measurement settings.")
    elif best_sfs_ratio <= SFS_ACCEPTABLE_RATIO_DB:
        print(f"WARNING: No Order N exceeded {SFS_ACCEPTABLE_RATIO_DB:.0f} dB Int/Ext ratio. Sound field separation may not be ideal. Re-consider measurement settings.")

    print("-" * 65)
    print(f"=> Order N with best SFS and solve stability: N={best_sfs_N} (Ratio: {best_sfs_ratio:.2f} dB)")
    if rolloff_knee:
        print(f"=> Order N at roll-off start: N={rolloff_knee['n']} (Ratio: {rolloff_knee['ratio']:.2f} dB)")
    else:
        print("=> Order N at roll-off start: not detected")

    def step1_option(label, order_n, ratio, err):
        option_warning = ""
        if ratio <= SFS_ACCEPTABLE_RATIO_DB:
            option_warning = f"Below the >{SFS_ACCEPTABLE_RATIO_DB:.0f} dB Int/Ext rule of thumb; sound field separation may not be ideal."
        return {
            'label': label,
            'n': order_n,
            'st': FIXED_NOISE_FLOOR_START_DB,
            'mx': FIXED_NOISE_FLOOR_MAX_DB,
            'lam': FIXED_MAX_LAMBDA,
            'ratio': ratio,
            'err': err,
            'warning': option_warning,
        }

    above_threshold_indices = [
        i for i, ratio in enumerate(ratio_vals)
        if np.isfinite(ratio) and ratio > SFS_ACCEPTABLE_RATIO_DB
    ]
    if rolloff_knee and rolloff_knee['ratio'] > SFS_ACCEPTABLE_RATIO_DB:
        recommended_N = rolloff_knee['n']
        recommended_idx = n_vals.index(recommended_N)
        recommendation_reason = (
            f"Selected the roll-off knee because it is above the >{SFS_ACCEPTABLE_RATIO_DB:.0f} dB "
            "Int/Ext rule of thumb and marks the start of the post-peak decline."
        )
    elif above_threshold_indices:
        recommended_idx = max(above_threshold_indices, key=lambda i: n_vals[i])
        recommended_N = n_vals[recommended_idx]
        recommendation_reason = (
            f"Selected the highest Order N that still exceeds the >{SFS_ACCEPTABLE_RATIO_DB:.0f} dB "
            "Int/Ext rule of thumb because no qualifying roll-off knee was detected."
        )
    else:
        recommended_idx = best_sfs_idx
        recommended_N = best_sfs_N
        recommendation_reason = (
            f"No Order N exceeded the >{SFS_ACCEPTABLE_RATIO_DB:.0f} dB Int/Ext rule of thumb, "
            "so the highest Int/Ext ratio was selected as the fallback."
        )

    recommended = step1_option(
        "Recommended Order N",
        recommended_N,
        ratio_vals[recommended_idx],
        err_vals[recommended_idx],
    )
    recommended['reason'] = recommendation_reason
    if rolloff_knee and recommended_N == rolloff_knee['n']:
        recommended['knee_distance'] = rolloff_knee['distance']
        recommended['method'] = rolloff_knee['method']

    options = {'recommended': recommended}

    tail_reference = select_tail_reference(n_vals, ratio_vals)
    reference_data = res_s1[(tail_reference['n'], -50.0, -50.0 - test_db_transition_span, 1e-10)]
    tail_power_vals, tail_count = calc_cumulative_tail(n_vals, tail_reference['n'], reference_data['internal_degree_shares'])
    tail_reference['sample_count'] = tail_count
    options.update(stage3_order_choices(n_vals, ratio_vals, err_vals, rolloff_knee, tail_power_vals, tail_reference['n']))
    tail_by_reference = {}
    for n, ratio in zip(n_vals, ratio_vals):
        sample_shares = res_s1[(n, -50.0, -50.0 - test_db_transition_span, 1e-10)]['internal_degree_shares']
        values, count = calc_cumulative_tail(n_vals, n, sample_shares)
        tail_by_reference[str(n)] = {'n': n, 'ratio': ratio, 'sample_count': count, 'power_db': values}
    print(f"Cumulative tail reference: N={tail_reference['n']}, Int/Ext={tail_reference['ratio']:.2f} dB "
          f"({tail_count}/{len(test_indices)} frequencies)" + ("; best-ratio fallback" if tail_reference['fallback'] else ""))
    for n, db in zip(n_vals, tail_power_vals):
        print(f"  Stop at N={n}: discarded internal tail {db:.2f} dB")

    print("\n" + "="*65)
    print(" FINAL ORDER N RECOMMENDATION")
    print("="*65)
    print(f"Recommended Order N: N={recommended_N}, Ratio={recommended['ratio']:.2f} dB, Resid={recommended['err']:.2f}%")
    print(recommendation_reason)
    print(f"Rule of thumb: Int/Ext SFS ratio greater than {SFS_ACCEPTABLE_RATIO_DB:.0f} dB has been found to produce acceptable results.")
    print("="*65)

    elapsed = time.time() - start_time
    print(f"\nStage 3 processing completed in {elapsed:.2f} seconds.")

    warning = ""
    if best_sfs_ratio <= SFS_ACCEPTABLE_RATIO_DB:
        warning = f"No Order N exceeded {SFS_ACCEPTABLE_RATIO_DB:.0f} dB Int/Ext ratio. Sound field separation may not be ideal; re-consider measurement settings."

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
        internal_tail_power_db=tail_power_vals,
        tail_reference=tail_reference,
    )
    if saved_plot:
        print(f"Stage 3 order sweep plot saved to: {saved_plot}")

    return {
        'options': {
            **options,
        },
        'recommended_key': 'recommended',
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
        'plot_path': saved_plot,
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
        test_start_db_range=config_process.TEST_START_DB_RANGE,
        test_lambda_range=config_process.TEST_LAMBDA_RANGE,
        test_db_transition_span=config_process.TEST_DB_TRANSITION_SPAN,
        use_optimized_origins=getattr(config_process, 'USE_OPTIMIZED_ORIGINS', False),
        speed_of_sound=getattr(config_process, 'SPEED_OF_SOUND', 343.0),
        kr_offset=KR_OFFSET,
        octave_resolution=getattr(config_process, 'TEST_OCTAVE_RESOLUTION', 12),
    )

if __name__ == "__main__":
    main()
