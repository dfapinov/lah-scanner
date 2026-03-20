"""
fdw_smoothing_core.py
=====================
Core mathematical operations for Frequency Dependent Windowing (FDW)
and Complex Smoothing of Impulse Responses.
"""

import numpy as np
from scipy.signal import find_peaks

def calculate_cycles_from_oct_res(oct_res):
    if oct_res <= 0: return 1.0
    return 1.0 / (2**(1.0/oct_res) - 1)

def calculate_fdw_durations(freqs, oct_res, rft_ms, max_cap_ms):
    t_rft_sec = rft_ms / 1000.0
    t_max_sec = max_cap_ms / 1000.0 if max_cap_ms else None
    target_cycles = calculate_cycles_from_oct_res(oct_res)
    
    durations = target_cycles / freqs
    durations = np.maximum(durations, t_rft_sec) 
    
    if t_max_sec: 
        durations = np.minimum(durations, t_max_sec) 
        
    return durations

def get_alpha_schedule(t_windows, rft_ms, alpha_hf, alpha_lf):
    t_rft_sec = rft_ms / 1000.0
    min_len = t_rft_sec
    max_len = np.max(t_windows)
    
    if max_len > min_len + 1e-9:
        log_min, log_max = np.log(min_len), np.log(max_len)
        log_curr = np.log(np.maximum(t_windows, 1e-9))
        ratio = np.clip((log_curr - log_min) / (log_max - log_min), 0.0, 1.0)
        return (alpha_hf * (1.0 - ratio)) + (alpha_lf * ratio)
    else:
        return np.full_like(t_windows, alpha_hf)

def get_earliest_significant_peak(ir_data, fs, threshold_db):
    abs_data = np.abs(ir_data)
    global_max_val = np.max(abs_data)
    
    if global_max_val == 0: return 0

    threshold_linear = global_max_val * (10**(threshold_db/20.0))
    min_dist_samples = int(fs * 0.0005) 
    peaks, _ = find_peaks(abs_data, height=threshold_linear, distance=min_dist_samples)
    
    if len(peaks) > 0:
        return peaks[0]
    else:
        return np.argmax(abs_data)

def apply_peak_causal_window(ir_data, Fs, T_peak_sec, T_FDW_sec, alpha):
    N_ir = len(ir_data)
    T_peak_idx = int(T_peak_sec * Fs)
    
    T_max_avail_sec = (N_ir / Fs) - T_peak_sec
    T_actual_fdw_sec = min(T_FDW_sec, T_max_avail_sec)
    
    T_FDW_idx = int(T_actual_fdw_sec * Fs)
    T_end_idx = T_peak_idx + T_FDW_idx
    T_end_idx = min(N_ir, T_end_idx)
    
    window_func = np.ones(N_ir)
    len_taper_region = T_end_idx - T_peak_idx
    
    if len_taper_region > 1:
        N_taper = int(len_taper_region * alpha)
        taper_start_global = T_end_idx - N_taper
        taper_start_global = max(T_peak_idx, taper_start_global)
        
        actual_slice_len = T_end_idx - taper_start_global
        if actual_slice_len > 1:
            n_taper = np.arange(actual_slice_len)
            norm_taper = n_taper / (actual_slice_len - 1)
            decay_curve = 0.5 * (1 + np.cos(np.pi * norm_taper))
            window_func[taper_start_global:T_end_idx] = decay_curve
        elif actual_slice_len == 1:
            window_func[taper_start_global] = 0.0
            
    window_func[T_end_idx:] = 0.0
    return ir_data * window_func

def create_triangular_weight(freqs, center_freq, band_width_factor):
    weights = np.zeros_like(freqs, dtype=float)
    f_min_band, f_max_band = center_freq / band_width_factor, center_freq * band_width_factor
    
    active = np.where((freqs >= f_min_band) & (freqs <= f_max_band))
    if len(active[0]) == 0: return weights
    
    ramp_up = (freqs[active] - f_min_band) / (center_freq - f_min_band)
    ramp_down = (f_max_band - freqs[active]) / (f_max_band - center_freq)
    
    weights[active] = np.minimum(ramp_up, ramp_down)
    return np.clip(weights, 0, 1)

def process_single_ir(ir_data, Fs, n_fft_common, freqs_common, T_peak_sec,
                      fdw_rft_ms, fdw_oct_res, fdw_max_cap_ms,
                      fdw_windows_per_oct, fdw_alpha_hf, fdw_alpha_lf):
    IR_MAX_AVAIL_SEC = (len(ir_data) / Fs) - T_peak_sec
    
    f_limit = Fs / 2.0 
    t_rft_sec = fdw_rft_ms / 1000.0
    target_cycles = calculate_cycles_from_oct_res(fdw_oct_res)
    f_transition = target_cycles / t_rft_sec
    
    if IR_MAX_AVAIL_SEC > 0:
        f_floor_dynamic = target_cycles / IR_MAX_AVAIL_SEC
    else:
        f_floor_dynamic = f_limit 

    F_CENTERS_LIST = []
    f_curr = f_limit
    while f_curr > f_floor_dynamic:
        F_CENTERS_LIST.append(f_curr)
        f_curr /= (2**(1.0/fdw_windows_per_oct))
    
    if not F_CENTERS_LIST or F_CENTERS_LIST[-1] > f_floor_dynamic:
        F_CENTERS_LIST.append(f_floor_dynamic)
        
    F_CENTERS = np.array(F_CENTERS_LIST)
    
    T_FDW_WINDOWS = calculate_fdw_durations(F_CENTERS, fdw_oct_res, fdw_rft_ms, fdw_max_cap_ms)
    T_FDW_WINDOWS = np.clip(T_FDW_WINDOWS, 0.0, IR_MAX_AVAIL_SEC) 
    
    ALPHA_SCHEDULE = get_alpha_schedule(T_FDW_WINDOWS, fdw_rft_ms, fdw_alpha_hf, fdw_alpha_lf)
    
    H_Accum = np.zeros(len(freqs_common), dtype=complex)
    W_Accum = np.zeros(len(freqs_common), dtype=float)
    Band_Width = 2**(1.0 / fdw_windows_per_oct)
    
    for i, fc in enumerate(F_CENTERS):
        w_ir = apply_peak_causal_window(ir_data, Fs, T_peak_sec, T_FDW_WINDOWS[i], ALPHA_SCHEDULE[i])
        fft_res = np.fft.rfft(w_ir, n=n_fft_common)
        
        if i == len(F_CENTERS) - 1:
            W_n = np.zeros_like(freqs_common)
            f_max_band = fc * Band_Width
            W_n[freqs_common <= fc] = 1.0
            ramp_idx = (freqs_common > fc) & (freqs_common <= f_max_band)
            W_n[ramp_idx] = (f_max_band - freqs_common[ramp_idx]) / (f_max_band - fc)
        else:
            W_n = create_triangular_weight(freqs_common, fc, Band_Width)
        
        H_Accum += W_n * fft_res
        W_Accum += W_n
        
    with np.errstate(divide='ignore', invalid='ignore'):
        H_Final = np.where(W_Accum > 1e-10, H_Accum / W_Accum, 0)
    
    f_nyq = Fs / 2
    f_taper_start = f_nyq * 0.98
    taper_indices = freqs_common > f_taper_start
    if np.any(taper_indices):
        norm_f = (freqs_common[taper_indices] - f_taper_start) / (f_nyq - f_taper_start)
        H_Final[taper_indices] *= 0.5 * (1 + np.cos(np.pi * norm_f))
    
    meta = {
        't_peak': T_peak_sec, 'f_trans': f_transition, 'f_centers': F_CENTERS, 
        't_windows': T_FDW_WINDOWS, 'alpha_sched': ALPHA_SCHEDULE, 'f_lf_anchor': F_CENTERS[-1]
    }
    return H_Final, meta
    
def apply_complex_smoothing(freqs, complex_data, oct_res, t_peak_sec):
    k_factor = (2**(0.5/oct_res) - 2**(-0.5/oct_res))
    rotation_phasor = np.exp(1j * 2 * np.pi * freqs * t_peak_sec)
    aligned_data = complex_data * rotation_phasor
    
    real_part = aligned_data.real
    imag_part = aligned_data.imag
    
    smoothed_real = np.zeros_like(real_part)
    smoothed_imag = np.zeros_like(imag_part)
    
    n_bins = len(freqs)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    
    for i in range(n_bins):
        f_target = freqs[i]
        
        if f_target <= 1e-6:
            smoothed_real[i] = real_part[i]
            smoothed_imag[i] = imag_part[i]
            continue
            
        bw_hz = f_target * k_factor
        sigma_hz = bw_hz / 2.355
        sigma_bins = sigma_hz / df
        
        if sigma_bins < 0.5:
            smoothed_real[i] = real_part[i]
            smoothed_imag[i] = imag_part[i]
            continue
            
        radius = int(3 * sigma_bins)
        start_idx = max(0, i - radius)
        end_idx = min(n_bins, i + radius + 1)
        
        indices = np.arange(start_idx, end_idx)
        weights = np.exp(-0.5 * ((indices - i) / sigma_bins)**2)
        w_sum = weights.sum()
        
        if w_sum > 0:
            weights /= w_sum
            smoothed_real[i] = np.sum(real_part[start_idx:end_idx] * weights)
            smoothed_imag[i] = np.sum(imag_part[start_idx:end_idx] * weights)
        else:
            smoothed_real[i] = real_part[i]
            smoothed_imag[i] = imag_part[i]

    smoothed_aligned = smoothed_real + 1j * smoothed_imag
    restored_data = smoothed_aligned * np.conj(rotation_phasor)
    
    return restored_data