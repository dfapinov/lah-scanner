import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import concurrent.futures
from scipy.special import spherical_jn, sph_harm, spherical_yn

# =================================================================
# USER CONFIGURATION
# =================================================================
CSV_FILE_PATH = 'jan_cylinder_test.csv'  # Path to your grid file
ORDER_N       = 8                         # Expansion Order (N=8 results in 162 modes)
FREQ_START    = 4000                      # Start frequency in Hz
FREQ_END      = 20000                     # End frequency in Hz
FREQ_STEP     = 100                       # Frequency resolution
SPEED_SOUND   = 343.0                     # m/s
# =================================================================

def hankel1(n, z):
    """Spherical Hankel function of the first kind."""
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)

def _compute_grid_metrics(args):
    """Worker function for multiprocessing."""
    f, c, r_sph, theta, phi_sph, order_N = args
    k = 2 * math.pi * f / c
    kr = r_sph * k
    
    A_int_cols = []
    A_ext_cols = []
    
    # Build internal and external blocks
    for n in range(order_N + 1):
        jn = spherical_jn(n, kr)
        hn = hankel1(n, kr)
        for m_ord in range(-n, n + 1):
            Y = sph_harm(m_ord, n, phi_sph, theta)
            A_int_cols.append(jn * Y)
            A_ext_cols.append(hn * Y)
            
    A_int = np.column_stack(A_int_cols)
    A_ext = np.column_stack(A_ext_cols)
    A_full = np.column_stack([A_int, A_ext])
    
    # 1. Condition Number (dB)
    s_full = np.linalg.svd(A_full, compute_uv=False)
    cond_db = 20 * np.log10(s_full[0] / s_full[-1]) if s_full[-1] > 0 else 120
    
    # 2. Field Separation Correlation (Principal Angle Cosine)
    q_int, _ = np.linalg.qr(A_int)
    q_ext, _ = np.linalg.qr(A_ext)
    S_overlap = np.linalg.svd(q_int.conj().T @ q_ext, compute_uv=False)
    sep_corr = np.min([1.0, S_overlap[0]])
    
    return cond_db, sep_corr

def run_assessment():
    # 1. Load and Prepare Coordinates
    df = pd.read_csv(CSV_FILE_PATH)
    r_xy, z_mm, phi_deg = df['r_xy_mm'].values/1000, df['z_mm'].values/1000, df['phi_deg'].values
    phi_rad = np.deg2rad(phi_deg)
    
    r_sph = np.sqrt((r_xy*np.cos(phi_rad))**2 + (r_xy*np.sin(phi_rad))**2 + z_mm**2)
    theta = np.arccos(z_mm / r_sph)
    phi_sph = np.arctan2(r_xy*np.sin(phi_rad), r_xy*np.cos(phi_rad))
    
    freqs = np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP)
    
    # Determine matrix size for console feedback
    num_mics = len(r_sph)
    num_columns = 2 * (ORDER_N + 1)**2
    print(f"--- Grid Assessment Initialization ---")
    print(f"Targeting Order: N = {ORDER_N}")
    print(f"Matrix Size:    {num_mics} (Mics) x {num_columns} (Modes)")
    print(f"Frequency Sweep: {FREQ_START}Hz to {FREQ_END}Hz")
    print(f"--------------------------------------")

    # 2. Parallel Processing
    args_list = [(f, SPEED_SOUND, r_sph, theta, phi_sph, ORDER_N) for f in freqs]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_compute_grid_metrics, args_list))
    
    cond_dbs, correlations = zip(*results)

    # 3. Plotting Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # Panel 1: Condition Number
    ax1.plot(freqs, cond_dbs, color='royalblue', linewidth=2)
    ax1.set_ylabel("Condition Number (dB)")
    ax1.set_ylim(10, 20)
    ax1.set_title(f"Grid Stability vs Frequency (Order N={ORDER_N})", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Separation Correlation
    ax2.plot(freqs, correlations, color='darkmagenta', linewidth=2)
    ax2.set_ylabel("Field Overlap (Correlation)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylim(0.8, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(freqs, 0.9, correlations, where=(np.array(correlations) >= 0.9), 
                     color='red', alpha=0.2, label='Separation Failure')
    ax2.set_title("Internal vs External Separation Sensitivity", fontsize=12)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_assessment()