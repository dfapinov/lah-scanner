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
    marker_styles = {
        "rolloff_knee": ("#e45756", "o", "Roll-off knee"),
        "balanced": ("#54a24b", "D", "Balanced"),
        "best_sfs": ("#b279a2", "s", "Best SFS"),
    }
    markers = []
    if not options:
        return markers
    for key, (color, marker, label) in marker_styles.items():
        opt = options.get(key)
        if opt and opt.get("n") is not None and opt.get("ratio") is not None:
            markers.append((float(opt["n"]), float(opt["ratio"]), color, marker, label))
    return markers

def save_stage3_order_sweep_plot(orders, ratios, residuals=None, options=None, knee=None, save_path=None):
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

        fig = Figure(figsize=(10, 5))
        FigureCanvasAgg(fig)
        ax_ratio = fig.add_subplot(111)
        ax_ratio.plot(orders_arr, ratios_arr, marker="o", linewidth=1.5, color="#4c78a8", label="Int/Ext ratio")
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
        CONDITION_METRICS=True, noise_floor_start_db=st_db,
        noise_floor_max_db=mx_db, max_lambda=lam
    )
    ratio_db = calc_internal_external_ratio(coeffs)
    err = (metrics['residual_norm'] / max(base_norm, 1e-20)) * 100.0
    return {'N': N, 'st_db': st_db, 'mx_db': mx_db, 'lam': lam, 'ratio_db': ratio_db, 'err': err}

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
    use_process_pool: bool = True
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
    sample_count = min(12, len(idx_target))
    test_indices = np.unique(np.linspace(idx_target[0], idx_target[-1], sample_count, dtype=int))
    print(f"Stage 3 order test frequency range: {freq_start_hz:g}-{freq_end_hz:g} Hz ({len(test_indices)} sample frequencies)")

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
        print(f"Stage 3 parallel backend: {backend} pool ({len(tasks)} tasks)")

        if use_process_pool:
            ctx = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(mp_context=ctx) as ex:
                raw_results = list(ex.map(_worker, tasks))
        else:
            with concurrent.futures.ThreadPoolExecutor() as ex:
                raw_results = list(ex.map(_worker, tasks))
            
        agg = {}
        for r in raw_results:
            key = (r['N'], r['st_db'], r['mx_db'], r['lam'])
            if key not in agg: agg[key] = {'ratio': [], 'err': []}
            agg[key]['ratio'].append(r['ratio_db'])
            agg[key]['err'].append(r['err'])
            
        return {k: {'ratio_db': np.mean(v['ratio']), 'err': np.mean(v['err'])} for k, v in agg.items()}

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
    prev_ratio = None
    
    for n in orders:
        data = res_s1[(n, -50.0, -50.0 - test_db_transition_span, 1e-10)]
        ratio = data['ratio_db']
        delta = (ratio - prev_ratio) if prev_ratio is not None else 0.0
        
        n_vals.append(n)
        ratio_vals.append(ratio)
        err_vals.append(data['err'])
        delta_vals.append(delta)
        prev_ratio = ratio
        
        print(f"{n:<10} | {ratio:<20.2f} | {data['err']:<12.2f} | {delta:<10.2f}")

    positive_candidates = [(n, r) for n, r in zip(n_vals, ratio_vals) if r > 0]
    usable_candidates = [(n, r) for n, r in positive_candidates if r >= 15.0]
    if usable_candidates:
        balance_pool = [(n, r) for n, r in usable_candidates if r >= 20.0]
        if balance_pool:
            stable_N, stable_ratio = min(balance_pool, key=lambda item: (item[1] - 20.0, item[0]))
        else:
            stable_N, stable_ratio = max(usable_candidates, key=lambda item: item[1])
    elif positive_candidates:
        stable_N, stable_ratio = max(positive_candidates, key=lambda item: item[1])
    else:
        stable_N, stable_ratio = max(zip(n_vals, ratio_vals), key=lambda item: item[1])

    best_sfs_idx = int(np.argmax(ratio_vals))
    best_sfs_N = n_vals[best_sfs_idx]
    best_sfs_ratio = ratio_vals[best_sfs_idx]
    best_sfs_err = err_vals[best_sfs_idx]
    rolloff_knee = find_rolloff_knee(n_vals, ratio_vals)
    if best_sfs_ratio <= 0:
        print("WARNING: No positive Int/Ext ratio was found. Sound field separation may not be ideal; re-consider measurement settings.")
    elif best_sfs_ratio < 15.0:
        print("WARNING: No Order N reached 15 dB Int/Ext ratio. Using the highest available ratio; sound field separation may not be ideal. Re-consider measurement settings.")

    print("-" * 65)
    print(f"=> Order N with balanced SFS and angular detail: N={stable_N} (Ratio: {stable_ratio:.2f} dB)")
    print(f"=> Order N with best SFS and solve stability: N={best_sfs_N} (Ratio: {best_sfs_ratio:.2f} dB)")
    if rolloff_knee:
        print(f"=> Order N at roll-off start: N={rolloff_knee['n']} (Ratio: {rolloff_knee['ratio']:.2f} dB)")
    else:
        print("=> Order N at roll-off start: not detected")

    print("\n" + "="*65)
    print(" FINAL ORDER N OPTIONS")
    print("="*65)
    print(f"Order N with balanced SFS and angular detail: N={stable_N}, Ratio={stable_ratio:.2f} dB, Resid={err_vals[n_vals.index(stable_N)]:.2f}%")
    print(f"Order N with best SFS and solve stability: N={best_sfs_N}, Ratio={best_sfs_ratio:.2f} dB, Resid={best_sfs_err:.2f}%")
    if rolloff_knee:
        print(f"Order N at roll-off start: N={rolloff_knee['n']}, Ratio={rolloff_knee['ratio']:.2f} dB, Resid={err_vals[n_vals.index(rolloff_knee['n'])]:.2f}%")
    print("="*65)

    elapsed = time.time() - start_time
    print(f"\nStage 3 processing completed in {elapsed:.2f} seconds.")

    warning = ""
    if best_sfs_ratio < 15.0:
        warning = "No Order N reached 15 dB Int/Ext ratio. Sound field separation may not be ideal; re-consider measurement settings."

    def step1_option(label, order_n, ratio, err):
        option_warning = ""
        if ratio < 15.0:
            option_warning = "Below 15 dB Int/Ext ratio; sound field separation may not be ideal."
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

    options = {
        'balanced': step1_option("Order N with balanced SFS and angular detail", stable_N, stable_ratio, err_vals[n_vals.index(stable_N)]),
        'best_sfs': step1_option("Order N with best SFS and solve stability", best_sfs_N, best_sfs_ratio, best_sfs_err),
    }
    if rolloff_knee:
        options['rolloff_knee'] = step1_option(
            "Order N at roll-off start",
            rolloff_knee['n'],
            rolloff_knee['ratio'],
            err_vals[n_vals.index(rolloff_knee['n'])]
        )
        options['rolloff_knee']['warning'] = "Detected from the post-peak Int/Ext curve knee; inspect the saved plot before treating it as a hard limit."
        options['rolloff_knee']['knee_distance'] = rolloff_knee['distance']
        options['rolloff_knee']['method'] = rolloff_knee['method']

    if save_plot and plot_save_path is None:
        stem = os.path.splitext(input_filename_opti)[0]
        plot_save_path = os.path.join(input_dir_opti, f"{stem}_stage3_order_sweep.png")
    saved_plot = save_stage3_order_sweep_plot(
        n_vals,
        ratio_vals,
        err_vals,
        options=options,
        knee=rolloff_knee,
        save_path=plot_save_path if save_plot else None
    )
    if saved_plot:
        print(f"Stage 3 order sweep plot saved to: {saved_plot}")

    return {
        'options': {
            **options,
        },
        'step1': {
            'orders': n_vals,
            'ratios': ratio_vals,
            'residuals': err_vals,
            'delta_ratios': delta_vals,
            'rolloff_knee': rolloff_knee,
        },
        'warning': warning,
        'plot_path': saved_plot,
    }

    # =========================================================================
    # LEGACY REFERENCE ONLY: STEPS 2 AND 3
    # =========================================================================
    # The noise-floor and lambda sweep below was found ineffective in practice.
    # It is intentionally unreachable so Stage 3 only runs the Step 1 order-N
    # test. Retaining the code makes it easy to revisit the experiment later.
    
    noise_floor_N = min(max_n, max(stable_N, best_sfs_N) + 1)
    active_orders = [stable_N]


    # =========================================================================
    # STEP 2: MAPPING THE NOISE FLOOR 
    # =========================================================================
    if MANUAL_STEP2_ORDER is not None:
        step2_orders = [min(max(MANUAL_STEP2_ORDER, min_n), max_n)]
        print("\n" + "="*65)
        print(f" STEP 2: MAPPING NOISE FLOOR (Manual Order N={step2_orders[0]}, Lam=0.01)")
        print("="*65)
    else:
        step2_orders = [noise_floor_N]
        print("\n" + "="*65)
        print(f" STEP 2: MAPPING NOISE FLOOR (Higher Order N={noise_floor_N}, Lam=0.01)")
        print("="*65)
        
    db_start, db_end = test_start_db_range
    step = -5.0 if db_end < db_start else 5.0
    start_dbs = np.arange(db_start, db_end + (step/2), step).tolist()
    
    configs_s2 = []
    for n in step2_orders:
        for st in start_dbs:
            configs_s2.append((n, st, st - test_db_transition_span, 0.01))
            
    res_s2 = run_batch(configs_s2)
    
    noise_eval_N = step2_orders[0]
    print(f"{'Start dB':<10} | {'Ratio N=' + str(noise_eval_N):<15}")
    print("-" * 30)
        
    peak_ratio_eval = -float('inf')
    best_st_db = -30.0
    
    for st in start_dbs:
        r_eval = res_s2[(noise_eval_N, st, st - test_db_transition_span, 0.01)]['ratio_db']
            
        flag = ""
        if r_eval > peak_ratio_eval:
            peak_ratio_eval = r_eval
            best_st_db = st
            flag = " * PEAK"
            
        print(f"{st:<10.1f} | {r_eval:<15.2f} {flag}")

    print(f"\n=> Order {noise_eval_N} Peak Rejection found at {best_st_db} dB")
    
    best_st_db += 5.0
    print(f"=> Adjusting NOISE_FLOOR_START_DB to {best_st_db} dB (+5dB) to begin gradual damping before the knee.")


    # =========================================================================
    # STEP 3: OPTIMIZING THE OPEN BRANCH (Grid Search)
    # =========================================================================
    print("\n" + "="*65)
    print(" STEP 3: OPTIMIZING LAMBDA & TRANSITION SPAN")
    print("="*65)
    
    # Test different max dB floors (transition spans)
    span_start = 10.0
    span_end = 30.0
    span_step = 5.0
    spans = np.arange(span_start, span_end + (span_step/2), span_step).tolist()
    
    lam_min, lam_max = test_lambda_range
    decades = int(np.log10(lam_max) - np.log10(lam_min))
    num_steps = max(2, decades * 2 + 1) # Generally 2 checks per log decade (e.g., 1e-6, 3.16e-6, 1e-5...)
    lambdas = np.logspace(np.log10(lam_min), np.log10(lam_max), num=num_steps).tolist()
    
    configs_s3 = []
    for n in active_orders:
        configs_s3.append((n, best_st_db, best_st_db - test_db_transition_span, 1e-10)) # Baselines
        for span in spans:
            mx_db = best_st_db - span
            for lam in lambdas:
                configs_s3.append((n, best_st_db, mx_db, lam))
                
    res_s3 = run_batch(configs_s3)
    
    # Store Baselines
    base_data = {n: res_s3[(n, best_st_db, best_st_db - test_db_transition_span, 1e-10)] for n in active_orders}
    print("Unregularized Baselines:")
    for n in active_orders:
        print(f"  N={n} -> Ratio: {base_data[n]['ratio_db']:.2f} dB, Resid: {base_data[n]['err']:.2f}%")
    print("-" * 65)
    
    print(f"{'Order':<6} | {'Max dB':<9} | {'Max Lam':<10} | {'Ratio':<8} | {'Resid %':<8} | {'Score'}")
    print("-" * 65)
    
    best_score = -float('inf')
    winner = None
    results_list = []
    
    for n in active_orders:
        for span in spans:
            mx_db = best_st_db - span
            for lam in lambdas:
                data = res_s3[(n, best_st_db, mx_db, lam)]
                
                # Score is Absolute Energy Ratio minus a heavy penalty for hurting the residual
                err_loss = data['err'] - base_data[n]['err']
                penalty = (err_loss * 50) if err_loss > 0 else 0
                
                # Bonus for lowering absolute error (rewards higher N if it stays stable)
                score = data['ratio_db'] - penalty - (data['err'] * 0.5)
                
                results_list.append({'n': n, 'st': best_st_db, 'mx': mx_db, 'lam': lam, 'ratio': data['ratio_db'], 'err': data['err'], 'score': score})

    results_list.sort(key=lambda x: x['score'], reverse=True)
    for row in results_list[:5]:
        flag = " * TOP" if row == results_list[0] else ""
        print(f"{row['n']:<6} | {row['mx']:<9.1f} | {row['lam']:<10.8f} | {row['ratio']:<8.2f} | {row['err']:<8.2f} | {row['score']:<8.2f}{flag}")

    def best_config_for_order(order_n):
        order_results = [r for r in results_list if r['n'] == order_n and r['ratio'] > 0]
        if not order_results:
            baseline = base_data.get(order_n)
            if baseline and baseline['ratio_db'] > 0:
                return {
                    'n': order_n,
                    'st': best_st_db,
                    'mx': best_st_db - test_db_transition_span,
                    'lam': 1e-10,
                    'ratio': baseline['ratio_db'],
                    'err': baseline['err'],
                    'score': baseline['ratio_db'] - (baseline['err'] * 0.5),
                }
            order_results = [r for r in results_list if r['n'] == order_n]
        return order_results[0] if order_results else None

    balanced_cfg = best_config_for_order(stable_N)
    best_sfs_cfg = best_config_for_order(best_sfs_N)
    if best_sfs_cfg is None:
        best_sfs_cfg = {
            'n': best_sfs_N,
            'st': balanced_cfg['st'],
            'mx': balanced_cfg['mx'],
            'lam': balanced_cfg['lam'],
            'ratio': best_sfs_ratio,
            'err': best_sfs_err,
            'score': balanced_cfg['score'],
        }

    def option_from_config(label, cfg, baseline_ratio):
        warning = ""
        if baseline_ratio < 15.0:
            warning = "Below 15 dB Int/Ext ratio; sound field separation may not be ideal."
        if cfg['n'] != stable_N:
            warning = (warning + " " if warning else "") + "Regularization thresholds were tuned with the balanced order N."
        return {
            'label': label,
            'n': cfg['n'],
            'st': cfg['st'],
            'mx': cfg['mx'],
            'lam': cfg['lam'],
            'ratio': cfg['ratio'],
            'err': cfg['err'],
            'warning': warning,
        }

    print("\n" + "="*65)
    print(" FINAL ORDER N OPTIONS")
    print("="*65)
    print(f"Order N with balanced SFS and angular detail: N={balanced_cfg['n']}, Ratio={balanced_cfg['ratio']:.2f} dB, Resid={balanced_cfg['err']:.2f}%")
    print(f"Order N with best SFS and solve stability: N={best_sfs_cfg['n']}, Ratio={best_sfs_cfg['ratio']:.2f} dB, Resid={best_sfs_cfg['err']:.2f}%")
    print("="*65)

    elapsed = time.time() - start_time
    print(f"\nStage 3 processing completed in {elapsed:.2f} seconds.")
    
    warning = ""
    if best_sfs_ratio < 15.0:
        warning = "No Order N reached 15 dB Int/Ext ratio. Sound field separation may not be ideal; re-consider measurement settings."

    return {
        'options': {
            'balanced': option_from_config("Order N with balanced SFS and angular detail", balanced_cfg, stable_ratio),
            'best_sfs': option_from_config("Order N with best SFS and solve stability", best_sfs_cfg, best_sfs_ratio),
        },
        'warning': warning,
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
        kr_offset=KR_OFFSET
    )

if __name__ == "__main__":
    main()
