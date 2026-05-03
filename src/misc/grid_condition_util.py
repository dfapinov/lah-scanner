import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import concurrent.futures
from scipy.special import spherical_jn, sph_harm, spherical_yn
import tkinter as tk
import threading
import platform
import ctypes

# =================================================================
# USER CONFIGURATION
# =================================================================
CSV_FILE_PATH = 'jan_cylinder_grid4.csv'  # Path to your grid file
ORDER_N       = 8                         # Expansion Order (N=8 results in 162 modes)
FREQ_START    = 100                      # Start frequency in Hz
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
    
    # 3. Gram Matrix Aliasing Metric
    G = A_full.conj().T @ A_full
    diag_G = np.abs(np.diag(G))
    diag_G[diag_G == 0] = 1e-12  # Prevent division by zero
    norm_factor = np.sqrt(np.outer(diag_G, diag_G))
    G_norm = np.abs(G) / norm_factor
    np.fill_diagonal(G_norm, 0.0)  # Ignore the diagonal
    
    N_cols = G.shape[0]
    if N_cols > 1:
        aliasing_metric = np.sum(G_norm) / (N_cols * (N_cols - 1))
    else:
        aliasing_metric = 0.0
        
    # 4. White Noise Amplification (WNA)
    G_inv = np.linalg.pinv(G)
    wna = 10 * np.log10(np.abs(np.trace(G_inv)))
    
    # 5. Maximum Coherence (Worst-Case Cross-Talk)
    max_coherence = np.max(np.abs(G_norm))
    
    # 6. Effective Degrees of Freedom (EDOF)
    eigvals = np.linalg.eigvalsh(G)
    p = eigvals / np.sum(eigvals)
    entropy = -np.sum(p * np.log(p + 1e-12))
    edof = np.exp(entropy)
        
    return cond_db, sep_corr, aliasing_metric, wna, max_coherence, edof

def launch_limit_gui(fig, axes_dict):
    """Launches a floating Tkinter window to manually control Y-axis limits."""
    def run_gui():
        # Crisp font rendering on Windows
        if platform.system() == "Windows":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass
                
        root = tk.Tk()
        root.title("Set Y-Axis Limits")
        root.attributes("-topmost", True)
        
        entries = {}
        for name, ax in axes_dict.items():
            frame = tk.Frame(root)
            frame.pack(fill="x", padx=10, pady=5)
            tk.Label(frame, text=name, width=15, anchor="w").pack(side="left")
            
            ymin, ymax = ax.get_ylim()
            
            tk.Label(frame, text="Min:").pack(side="left")
            e_min = tk.Entry(frame, width=7)
            e_min.insert(0, f"{ymin:.2f}")
            e_min.pack(side="left", padx=2)
            
            tk.Label(frame, text="Max:").pack(side="left")
            e_max = tk.Entry(frame, width=7)
            e_max.insert(0, f"{ymax:.2f}")
            e_max.pack(side="left", padx=2)
            
            entries[name] = (ax, e_min, e_max)
            
        def apply_limits():
            for name, (ax, e_min, e_max) in entries.items():
                try:
                    ax.set_ylim(float(e_min.get()), float(e_max.get()))
                except ValueError:
                    pass  # Ignore empty or invalid inputs
            fig.canvas.draw_idle()
            
        tk.Button(root, text="Apply Limits", command=apply_limits).pack(pady=10)
        root.mainloop()
        
    thread = threading.Thread(target=run_gui)
    thread.daemon = True
    thread.start()

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
    
    cond_dbs, correlations, aliasing_metrics, wnas, max_coherences, edofs = zip(*results)

    # 3. Plotting Results
    fig, axs = plt.subplots(3, 2, figsize=(12, 14), sharex=True)
    (ax1, ax2), (ax3, ax4), (ax5, ax6) = axs

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
    ax2.set_ylim(0.8, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(freqs, 0.9, correlations, where=(np.array(correlations) >= 0.9), 
                     color='red', alpha=0.2, label='Separation Failure')
    ax2.set_title("Internal vs External Separation Sensitivity", fontsize=12)
    ax2.legend()
    
    # Panel 3: Aliasing Metric
    ax3.plot(freqs, aliasing_metrics, color='crimson', linewidth=2, label='Aliasing Metric')
    ax3.set_ylabel("Mean Off-Diag Magnitude")
    ax3.set_ylim(bottom=0)
    ax3.set_title("Gram Matrix Orthogonality (Spatial Aliasing)", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Panel 4: White Noise Amplification (WNA)
    ax4.plot(freqs, wnas, color='darkorange', linewidth=2, label='WNA')
    ax4.set_ylabel("WNA (dB)")
    ax4.set_title("White Noise Amplification", fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Panel 5: Maximum Coherence (Worst-Case Cross-Talk)
    ax5.plot(freqs, max_coherences, color='forestgreen', linewidth=2, label='Max Coherence')
    ax5.set_ylabel("Max Coherence Magnitude")
    ax5.set_xlabel("Frequency (Hz)")
    ax5.set_ylim(bottom=0)
    ax5.set_title("Worst-Case Cross-Talk", fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Panel 6: Effective Degrees of Freedom (EDOF)
    ax6.plot(freqs, edofs, color='purple', linewidth=2, label='EDOF')
    ax6.set_ylabel("Effective DOF")
    ax6.set_xlabel("Frequency (Hz)")
    ax6.set_title("Effective Degrees of Freedom", fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()
    
    # 4. Launch Limit Control Panel
    axes_dict = {
        "Condition": ax1,
        "Separation": ax2,
        "Aliasing": ax3,
        "WNA": ax4,
        "Max Coherence": ax5,
        "EDOF": ax6
    }
    launch_limit_gui(fig, axes_dict)
    
    plt.show()

if __name__ == '__main__':
    run_assessment()