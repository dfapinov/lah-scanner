#!/usr/bin/env python3  # shebang so you can run this file directly from the terminal

"""
Sweep Measurement Script – acoustic system characterization
============================================================

Quick usage guide:
   • Run "python sweep_function.py" to list audio devices and channels.

   • As a library (recommended for measurements):
      from sweep_function import run_measurement     # import the function
      res = run_measurement()                        # returns dict with arrays + metadata


Inputs/outputs (when imported and run):
  • Inputs come entirely from config_capture.py (devices, channels, sample rate, sweep params).
  • Output is a Python dict with transmit/receive waveforms, aligned/conditioned signals,
    driver info, and peak levels — you decide what to do with it (e.g., save WAV/NPZ later).
"""

from __future__ import annotations  # allows forward-referenced type hints (Python ≥3.7)
import os                           # standard library: environment variables, etc.
import threading                    # for Event used to signal capture completion

# Enable ASIO build of PortAudio in python-sounddevice (Windows).
# This must be set before importing sounddevice.
os.environ["SD_ENABLE_ASIO"] = "1"   # any value works; presence enables ASIO driver discovery

import numpy as np                   # numerical arrays and math
import sounddevice as sd             # real-time audio I/O via PortAudio

# ────────────────────────────── Config import (all parameters come from here) ──────────────────────────────
from config_capture import (        # import all user-configured parameters from your capture config
    # Devices & channels
    OUT_DEV,                        # integer device index for output (speaker+reference)
    IN_DEV,                         # integer device index for input  (mic+loopback)
    OUT_CH_SPKR,                    # output channel index that drives the loudspeaker
    OUT_CH_REF,                     # output channel index that carries the reference sweep
    IN_CH_MIC,                      # input  channel index for the measurement microphone
    IN_CH_LOOP,                     # input  channel index for the loopback/reference

    # Signal / timing
    FS,                             # sample rate in Hz for playback/record
    SWEEP_DUR_S,                    # sweep duration in seconds (signal length before padding)
    F1_HZ,                          # start frequency of exponential sweep (Hz)
    F2_HZ,                          # end frequency of sweep (Hz); None → auto (≈0.48*FS)
    SWEEP_LEVEL_DBFS,               # sweep level in dB relative to full scale
    FADE_MS,                        # Hann fade length at start/end of sweep (ms)
    PRE_SIL_MS,                     # leading silence before sweep (ms)
    POST_SIL_MS,                    # trailing silence after sweep (ms)

    # Conditioning
    MIC_TAIL_TAPER_MS,              # post-sweep fade length on mic channel (ms)

    # Streaming
    BLOCKSIZE,                      # audio block size; None/0 lets host choose
    WASAPI_EXCLUSIVE,               # True for exclusive mode on WASAPI (Windows), else shared
    )

# ─────────────────────────────────── Signal helpers ───────────────────────────────────
def db_to_lin(db): return 10 ** (db / 20.0)  # convert dBFS to linear amplitude gain

def hann_fade(sig: np.ndarray, fade_ms: float, fs: int) -> np.ndarray:
    n = len(sig)                                                    # total samples
    f = max(1, int(round(fade_ms / 1000.0 * fs)))                   # fade length in samples
    if 2 * f >= n:                                                  # if fades would overlap, skip
        return sig
    w = np.ones(n, dtype=sig.dtype)                                 # start with all-ones window
    up = 0.5 * (1 - np.cos(np.pi * np.arange(f) / (f - 1)))         # rising half-Hann
    w[:f] *= up                                                      # apply fade-in
    w[-f:] *= up[::-1]                                              # apply fade-out
    return sig * w                                                  # return faded signal

def make_exp_sweep(fs, T, f1, f2, fade_ms=10.0, level_dbfs=-12.0):
    if f2 is None:                                                  # if end freq unspecified
        f2 = fs * 0.48                                              # default ≈ 0.48 * FS
    f2 = min(float(f2), fs * 0.499)                                 # clamp below Nyquist
    n = int(round(T * fs))                                          # number of samples
    t = np.arange(n) / fs                                           # time vector [0..T)
    w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2                         # angular freqs
    L = T / np.log(w2 / w1)                                         # log-sweep constant
    phase = w1 * L * (np.exp(t / L) - 1.0)                          # exponential phase law
    s = np.sin(phase).astype(np.float32)                             # raw sweep (float32)
    s = hann_fade(s, fade_ms, fs)                                    # apply short Hann fades
    s *= db_to_lin(level_dbfs) / (np.max(np.abs(s)) + 1e-12)         # set level to target dBFS
    inv = s[::-1].astype(np.float64) * np.exp(np.arange(n) / fs / L) # simple inverse kernel
    inv = (inv / (np.max(np.abs(inv)) + 1e-12)).astype(np.float32)   # normalize inverse
    return s.astype(np.float32), inv                                 # return sweep and inverse

def rfft_xcorr(a, b):
    n = int(2 ** np.ceil(np.log2(len(a) + len(b) - 1)))             # power-of-two FFT size
    A = np.fft.rfft(a, n=n)                                         # FFT of first signal
    B = np.fft.rfft(b, n=n)                                         # FFT of second signal
    x = np.fft.irfft(A * np.conj(B), n=n)                           # circular cross-corr via FFT
    x = np.roll(x, len(b) - 1)                                      # shift so lag=0 aligns
    lags = np.arange(-len(b) + 1, len(a))                           # lag indices
    return lags, x[:len(lags)]                                      # return lags and correlation

def matched_filter_detect(x, ref, search_start=None, search_end=None):
    # Use the full sweep as the reference; do not reverse here.
    lags, corr = rfft_xcorr(x, ref)                                 # cross-correlate input vs ref
    if search_start is None:                                        # default search window start
        search_start = 0
    if search_end is None:                                          # default search window end
        search_end = len(x) - 1
    m = (lags >= search_start) & (lags <= search_end)               # mask for search window
    lags_sel, corr_sel = lags[m], corr[m]                           # select windowed region
    i = int(np.argmax(corr_sel))                                    # index of max correlation
    return int(lags_sel[i]), float(corr_sel[i])                     # return lag at best match

def fade_in_until(idx_end, sig, fs, pad_ms):
    y = sig.copy()                                                  # work on a copy
    if idx_end <= 0:                                                # nothing to fade-in before 0
        return y
    N = min(idx_end, max(1, int(round(pad_ms / 1000.0 * fs))))      # fade length (samples)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(N) / (N - 1))) if N > 1 else np.ones(1, y.dtype)  # Hann up
    start = idx_end - N                                             # fade-in start index
    if start < 0:                                                   # clamp if negative
        ramp = ramp[-start:]
        start = 0
    y[:start] = 0.0                                                 # hard-zero before ramp
    y[start:idx_end] *= ramp.astype(y.dtype)                        # apply fade-in ramp
    return y                                                        # return conditioned signal

def fade_to_zero_after(idx_start, sig, fs, pad_ms):
    y = sig.copy()                                                  # copy input
    N = max(1, int(round(pad_ms / 1000.0 * fs)))                    # fade length (samples)
    if idx_start >= len(y):                                         # nothing to do if start beyond end
        return y
    end_fade = min(len(y), idx_start + N)                           # end index for fade-out
    if end_fade > idx_start:                                        # if fade region exists
        n = end_fade - idx_start
        ramp = 0.5 * (1 + np.cos(np.pi * np.arange(n) / (n - 1)))   # Hann down from 1→0
        y[idx_start:end_fade] *= ramp.astype(y.dtype)               # apply fade-out
    if end_fade < len(y):                                           # zero everything after fade
        y[end_fade:] = 0.0
    return y                                                        # return tapered signal

# ───────────────────────────────────── Device helpers ─────────────────────────────────────
def api_name_for_device(dev_index: int) -> str:
    d = sd.query_devices(dev_index)                                 # metadata for a device index
    return sd.query_hostapis()[d['hostapi']]['name'].upper()        # look up host API name (upper)

def list_all_hostapis_and_devices():
    """Print host APIs and all devices with channel counts (grouped by API)."""
    # Re-init to get a fresh view (avoids stale state on some hosts)
    sd._terminate(); sd._initialize()                               # hard refresh PortAudio state
    apis = sd.query_hostapis()                                      # list of host APIs
    devs = sd.query_devices()                                       # list of all devices

    print("\n=== Host APIs ===")                                    # header
    for i, a in enumerate(apis):                                    # iterate host APIs
        print(f"[{i}] {a['name']}")                                 # show index and API name

    print("\n=== Devices (grouped by Host API) ===")                # header
    for api_idx, a in enumerate(apis):                              # for each API…
        print(f"\n-- {api_idx}: {a['name']} --")                    # group header
        for i, d in enumerate(devs):                                # iterate devices
            if d['hostapi'] == api_idx:                             # only those in this API
                sr = d.get('default_samplerate', 'n/a')             # show default sample rate
                print(f"  [{i}] {d['name']} | In:{d['max_input_channels']} Out:{d['max_output_channels']} | default_sr:{sr}")  # device info

    # Also echo the current config selections for quick reference
    try:
        out_name = sd.query_devices(OUT_DEV)['name']                # configured output device name
        in_name  = sd.query_devices(IN_DEV)['name']                 # configured input  device name
    except Exception:
        out_name = "(invalid index)"; in_name = "(invalid index)"   # fallback labels if indices bad
    print("\n=== Current config_capture selections ===")            # section header
    print(f"OUT_DEV={OUT_DEV} ({out_name})  | OUT_CH_SPKR={OUT_CH_SPKR}, OUT_CH_REF={OUT_CH_REF}")  # echo outputs
    print(f"IN_DEV={IN_DEV}   ({in_name})   | IN_CH_MIC={IN_CH_MIC}, IN_CH_LOOP={IN_CH_LOOP}")      # echo inputs
    print(f"FS={FS} Hz, BLOCKSIZE={BLOCKSIZE}, WASAPI_EXCLUSIVE={WASAPI_EXCLUSIVE}")                # streaming params
    print(f"SWEEP: {F1_HZ} Hz → {('0.48*FS' if F2_HZ is None else F2_HZ)} Hz, {SWEEP_DUR_S}s, level {SWEEP_LEVEL_DBFS} dBFS")  # sweep summary
    print(f"Timing: pre {PRE_SIL_MS} ms, post {POST_SIL_MS} ms")    # padding summary

# ───────────────────────────────────── Core measurement (callback engine) ─────────────────────────────────────
def run_measurement() -> dict:
    """Perform one sweep measurement and return a dict with arrays and metadata."""
    sweep, _inv = make_exp_sweep(FS, SWEEP_DUR_S, F1_HZ, F2_HZ, FADE_MS, SWEEP_LEVEL_DBFS)  # build sweep/inverse

    pre  = int(round(PRE_SIL_MS  / 1000.0 * FS))                     # leading silence (samples)
    post = int(round(POST_SIL_MS / 1000.0 * FS))                     # trailing silence (samples)
    slen = len(sweep)                                                # sweep length (samples)

    # Single transmit buffer for both outputs
    tx_signal = np.concatenate([np.zeros(pre, np.float32), sweep, np.zeros(post, np.float32)])  # playback program
    n_frames = len(tx_signal)                                        # total frames to stream

    k_tx_sweep_start = pre                                          # index where sweep starts in tx buffer
    sweep_end_idx    = k_tx_sweep_start + slen                      # index where sweep ends in tx buffer

    out_ch_count = max(OUT_CH_SPKR, OUT_CH_REF) + 1                 # number of output channels to allocate
    in_ch_count  = max(IN_CH_MIC,  IN_CH_LOOP)  + 1                 # number of input  channels to allocate

    out_frames = np.zeros((n_frames, out_ch_count), dtype=np.float32)  # output frame buffer
    out_frames[:, OUT_CH_REF]  = tx_signal                           # place reference on its channel
    out_frames[:, OUT_CH_SPKR] = tx_signal                           # place speaker drive on its channel

    out_api = api_name_for_device(OUT_DEV)                           # name of output host API
    in_api  = api_name_for_device(IN_DEV)                            # name of input  host API

    # Host settings: ASIO device opens only the channels we need via selectors.
    use_asio_in  = ("ASIO" in in_api)                                # True if input host is ASIO
    use_asio_out = ("ASIO" in out_api)                               # True if output host is ASIO

    if use_asio_in:
        in_channels = 2                                              # we will open exactly 2 ASIO inputs
        in_extra = sd.AsioSettings(channel_selectors=[IN_CH_LOOP, IN_CH_MIC])  # map: 0→loop, 1→mic
    else:
        in_channels = max(IN_CH_MIC, IN_CH_LOOP) + 1                 # open up to the highest index
        in_extra = sd.WasapiSettings(exclusive=WASAPI_EXCLUSIVE) if "WASAPI" in in_api else None  # WASAPI opts

    if use_asio_out:
        out_channels = 2                                             # we will open exactly 2 ASIO outputs
        out_extra = sd.AsioSettings(channel_selectors=[OUT_CH_SPKR, OUT_CH_REF])  # map: 0→spkr, 1→ref
    else:
        out_channels = max(OUT_CH_SPKR, OUT_CH_REF) + 1              # open up to the highest index
        out_extra = sd.WasapiSettings(exclusive=WASAPI_EXCLUSIVE) if "WASAPI" in out_api else None  # WASAPI opts

    # If BLOCKSIZE is None/0, pass 0 to let host choose its native block size.
    native_block = 0 if (BLOCKSIZE is None or BLOCKSIZE == 0) else BLOCKSIZE  # 0 means “host default”

    # Capture buffers
    rec_loop = np.zeros(n_frames, dtype=np.float32)                  # loopback capture buffer
    rec_mic  = np.zeros(n_frames, dtype=np.float32)                  # microphone capture buffer

    # Playback/capture indices shared with callback
    idx_play = 0                                                     # playback frame cursor
    idx_rec  = 0                                                     # record   frame cursor
    done_evt = threading.Event()                                     # signals when both streams complete
    status_flag = {"overflow": False, "underflow": False}            # record xruns for diagnostics

    def callback(indata, outdata, frames, time_info, status):
        nonlocal idx_play, idx_rec                                  # allow updates to outer scope vars
        # Record driver status (optional, useful for logging)
        if status.input_overflow:
            status_flag["overflow"] = True                           # flag input overflow (missed samples)
        if status.output_underflow:
            status_flag["underflow"] = True                          # flag output underflow (late writes)

        # How many frames remain to play/capture
        n_remain_out = out_frames.shape[0] - idx_play                # frames left in tx buffer
        n_out = frames if frames <= n_remain_out else n_remain_out   # frames we can output this block

        # Fill output
        if use_asio_out:
            # [SPKR, REF] packed into 2 channels
            outdata[:n_out, 0] = out_frames[idx_play:idx_play+n_out, OUT_CH_SPKR]  # write speaker channel
            outdata[:n_out, 1] = out_frames[idx_play:idx_play+n_out, OUT_CH_REF]   # write reference channel
            if frames > n_out:
                outdata[n_out:, :] = 0.0                             # pad with zeros if final short block
        else:
            outdata[:n_out, :out_channels] = out_frames[idx_play:idx_play+n_out, :out_channels]  # write block
            if frames > n_out:
                outdata[n_out:, :out_channels] = 0.0                 # pad remainder with zeros

        # Capture input (may be partial on last buffer)
        n_remain_in = rec_loop.shape[0] - idx_rec                    # frames left in rx buffers
        n_in = frames if frames <= n_remain_in else n_remain_in      # frames we can read this block
        if n_in > 0:                                                 # only if room remains
            if use_asio_in:
                # indata[:, 0] is LOOP, indata[:, 1] is MIC
                rec_loop[idx_rec:idx_rec+n_in] = indata[:n_in, 0]    # copy loopback to buffer
                rec_mic[idx_rec:idx_rec+n_in]  = indata[:n_in, 1]    # copy mic to buffer
            else:
                if IN_CH_LOOP < indata.shape[1]:
                    rec_loop[idx_rec:idx_rec+n_in] = indata[:n_in, IN_CH_LOOP]  # pick configured loop channel
                if IN_CH_MIC < indata.shape[1]:
                    rec_mic[idx_rec:idx_rec+n_in]  = indata[:n_in, IN_CH_MIC]   # pick configured mic channel

        idx_play += n_out                                            # advance playback cursor
        idx_rec  += n_in                                             # advance record cursor

        # If we finished both playback and capture, signal done
        if idx_play >= out_frames.shape[0] and idx_rec >= rec_loop.shape[0]:
            done_evt.set()                                           # notify main thread to stop waiting

    # Duplex callback stream (single stream for all host APIs)
    with sd.Stream(
        device=(IN_DEV, OUT_DEV),                                    # (input_device, output_device) indices
        samplerate=FS,                                               # sample rate (Hz)
        blocksize=native_block,                                      # requested block size (0 → host default)
        dtype=("float32", "float32"),                                # 32-bit float I/O
        channels=(in_channels, out_channels),                        # (num_in_ch, num_out_ch)
        dither_off=True,                                             # keep bit-transparent float path
        extra_settings=(in_extra, out_extra),                        # API-specific extras (ASIO/WASAPI)
        callback=callback,                                           # our real-time processing function
    ) as s:
        # Small settle is harmless; many drivers like a brief spin-up
        sd.sleep(150)                                                # let backend settle briefly
        # Let the callback drive everything; wait until it has filled the arrays
        done_evt.wait(timeout=max(2.0, (n_frames / FS) + 2.0))       # safety timeout (seconds)

    if status_flag["overflow"]:
        print("Warning: input overflow was reported by the driver.") # notify if input xrun occurred
    if status_flag["underflow"]:
        print("Warning: output underflow was reported by the driver.")  # notify if output xrun occurred

    # ───────────── Post-processing / alignment ─────────────
    k_rec_sweep, _ = matched_filter_detect(rec_loop, sweep)          # find sweep start in loopback
    L = k_rec_sweep - k_tx_sweep_start                               # sample delay vs transmit index

    def advance(sig, lag):
        if lag <= 0:
            return np.concatenate([np.zeros(-lag, sig.dtype), sig])  # pad front if negative lag
        if lag >= len(sig):
            return np.zeros(1, dtype=sig.dtype)                      # empty if lag beyond end
        return sig[lag:]                                             # otherwise drop leading samples

    mic_aligned  = advance(rec_mic,  L)                              # align mic to transmit timeline
    loop_aligned = advance(rec_loop, L)                              # align loopback likewise
    mic_edge_in  = fade_in_until(k_tx_sweep_start, mic_aligned, FS, pad_ms=PRE_SIL_MS)   # soft fade-in
    mic_cond     = fade_to_zero_after(sweep_end_idx, mic_edge_in, FS, pad_ms=MIC_TAIL_TAPER_MS)  # tail taper

    def pk(x): return 20 * np.log10(float(np.max(np.abs(x))) + 1e-12)  # helper: peak level in dBFS

    return {
        "fs": FS,                                                    # sample rate (Hz)
        "latency_samples": int(L),                                   # measured latency in samples
        "latency_seconds": float(L / FS),                            # measured latency in seconds
        "tx_signal": tx_signal.astype(np.float32),                   # transmitted buffer
        "rx_loop_raw": rec_loop.astype(np.float32),                  # raw loopback capture
        "rx_mic_raw":  rec_mic.astype(np.float32),                   # raw microphone capture
        "rx_loop_aligned": loop_aligned.astype(np.float32),          # loopback aligned to tx
        "rx_mic_aligned":  mic_aligned.astype(np.float32),           # mic aligned to tx
        "rx_mic_conditioned": mic_cond.astype(np.float32),           # mic after edge-in and tail taper
        "driver": {
            "hostapi_out": out_api,                                  # output API name
            "hostapi_in":  in_api,                                   # input  API name
            # We keep this minimal; host decides blocksize/latency.
            "requested_blocksize": int(BLOCKSIZE or 0),              # block size we asked for (0=auto)
        },
        "peaks_dbfs": {
            "tx_signal": float(pk(tx_signal)),                       # peak dBFS of transmit buffer
            "rx_loop_raw": float(pk(rec_loop)),                      # peak dBFS of raw loopback
            "rx_mic_raw":  float(pk(rec_mic)),                       # peak dBFS of raw mic
        },
    }                                                                # caller can save/analyse as needed

# ───────────────────────────────────── CLI device lister ─────────────────────────────────────
if __name__ == "__main__":            # only when run as a script (not imported)
    list_all_hostapis_and_devices()   # print available APIs/devices/channels and exit

r"""
       ____  __  __ ___ _____ ______   __   
      |  _ \|  \/  |_ _|_   _|  _ \ \ / /   
      | | | | |\/| || |  | | | |_) \ V /    
      | |_| | |  | || |  | | |  _ < | |     
     _|____/|_| _|_|___| |_| |_|_\_\|_|   __
    |  ___/ \  |  _ \_ _| \ | |/ _ \ \   / /
    | |_ / _ \ | |_) | ||  \| | | | \ \ / / 
    |  _/ ___ \|  __/| || |\  | |_| |\ V /  
    |_|/_/   \_\_|  |___|_| \_|\___/  \_/   
                       
                     ███                 
                   █████         ███     
                 ███████         ████    
               █████████    ██    ████   
     ███████████████████    ████   ████  
    ████████████████████     ███   ████  
    ████████████████████      ██    ████ 
    ████████████████████      ███    ████ 
    ████████████████████      ███    ████ 
    ████████████████████     ███    ████  
     ███████████████████    ████   ████  
              ██████████    ███   ████   
                ████████    █    ████    
                  ██████         ███     
                     ███    
"""