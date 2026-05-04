#!/usr/bin/env python3
"""
spatial_error_viewer.py
=======================
An interactive 3D viewer to visualize the spatial distribution of the
Spherical Harmonic Expansion (SHE) fit error for each measurement point.

This tool helps diagnose where the mathematical model is most dissimilar
from the measured data in physical space.
"""

import os
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Local Project Imports ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'process')))
try:
    import schema
    from utils import load_and_parse_npz, spherical_to_cartesian
    from extract_pressures_core import evaluate_she_field
except ImportError as e:
    sys.exit(f"Error: Could not import required project modules. {e}")

try:
    from config_process import SPEED_OF_SOUND
except ImportError:
    SPEED_OF_SOUND = 343.0

def compute_spatial_error_data(she_dict, p_measured_all, coords_sph, c_sound=343.0):
    """
    Core computation engine for the Spatial Error Viewer. Subsamples frequencies to 1/96 octave
    and calculates 3D reconstruction field errors. Shared by standalone and embedded views.
    """
    all_freqs = she_dict[schema.FREQS]
    all_coeffs = she_dict[schema.COEFFS]
    all_n_used = she_dict[schema.N_USED]
    all_origins = she_dict[schema.ORIGINS_MM]
    
    target_freqs = []
    f = 20.0
    max_f = all_freqs[-1]
    while f <= max_f:
        target_freqs.append(f)
        f *= (2 ** (1.0 / 96.0))
        
    freq_indices = []
    for tf in target_freqs:
        idx = (np.abs(all_freqs - tf)).argmin()
        if idx not in freq_indices:
            freq_indices.append(idx)
            
    freq_indices = np.array(freq_indices)
    
    sub_freqs = all_freqs[freq_indices]
    sub_coeffs = all_coeffs[freq_indices]
    sub_n_used = all_n_used[freq_indices]
    sub_origins = all_origins[freq_indices]
    
    sub_p_measured = p_measured_all[freq_indices, :]
    
    sub_she_dict = {
        schema.FREQS: sub_freqs,
        schema.COEFFS: sub_coeffs,
        schema.N_USED: sub_n_used,
        schema.ORIGINS_MM: sub_origins
    }
    
    reconstruction_result = evaluate_she_field(
        coords_sph=coords_sph,
        she_input=sub_she_dict,
        obs_mode="Full",
        c_sound=c_sound,
        use_optimized_origins=True
    )
    
    sub_p_reconstructed = reconstruction_result["complex"]
    
    r_static = coords_sph[:, 2]
    th_static = np.radians(coords_sph[:, 0])
    ph_static = np.radians(coords_sph[:, 1])
    x_coords, y_coords, z_coords = spherical_to_cartesian(r_static, th_static, ph_static)
    
    return x_coords, y_coords, z_coords, sub_freqs, sub_n_used, sub_p_measured, sub_p_reconstructed

class SpatialErrorView:
    """
    Pure MVC View class for Spatial Error. Handles Matplotlib rendering.
    """
    def __init__(self, figsize=(10, 8)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(left=0, right=0.85, bottom=0.05, top=0.95)
        self.cbar_ax = self.fig.add_axes([0.85, 0.15, 0.03, 0.7])
        self.cbar = None
        self.scatter = None
        
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

    def update_view(self, x_coords, y_coords, z_coords, freq, n_used, p_measured, p_reconstructed, threshold_db):
        eps = np.finfo(float).eps
        mags_db = 20 * np.log10(np.abs(p_measured) + eps)
        max_mag_db = np.max(mags_db)
        cutoff_db = max_mag_db + threshold_db

        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.abs(p_reconstructed - p_measured)
            norm_p = np.abs(p_measured)
            pct_error = np.where(norm_p > 1e-9, (error / norm_p) * 100.0, 0)

        pct_error_clipped = np.clip(pct_error, 0, 25).astype(float)
        
        mask_below_thresh = mags_db < cutoff_db
        pct_error_clipped[mask_below_thresh] = np.nan

        if self.scatter is not None:
            self.scatter.remove()

        cmap = plt.get_cmap('viridis_r').copy()
        cmap.set_bad(color='lightgray', alpha=0.2)
        self.scatter = self.ax.scatter(x_coords, y_coords, z_coords, c=pct_error_clipped, cmap=cmap, vmin=0, vmax=25)
        
        if self.cbar is None:
            self.cbar = self.fig.colorbar(self.scatter, cax=self.cbar_ax)
            self.cbar.set_label('Fit Error (%)')
            
            max_range = np.array([x_coords.max()-x_coords.min(), y_coords.max()-y_coords.min(), z_coords.max()-z_coords.min()]).max() / 2.0
            mid_x = (x_coords.max()+x_coords.min()) * 0.5
            mid_y = (y_coords.max()+y_coords.min()) * 0.5
            mid_z = (z_coords.max()+z_coords.min()) * 0.5
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
            self.ax.set_box_aspect((1, 1, 1))

        self.ax.set_title(f'Fit Error at {freq:.1f} Hz (Order N = {n_used})')
        self.fig.canvas.draw_idle()


class SpatialErrorViewerApp:
    """
    Standalone Tkinter application for the Spatial Error Viewer.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Spatial Fit Error Viewer")
        self.root.geometry("1000x800")
        
        self.npz_path = tk.StringVar()
        self.h5_path = tk.StringVar()
        
        self.freqs = None
        self.n_used = None
        self.p_measured_all = None
        self.p_reconstructed_all = None
        self.x_coords = None
        self.y_coords = None
        self.z_coords = None
        
        self._build_ui()
        
        try:
            from config_process import OUTPUT_DIR_FDW, OUTPUT_FILENAME_FDW, OUTPUT_DIR_SHE, OUTPUT_FILENAME_SHE
            default_npz = os.path.join(OUTPUT_DIR_FDW, OUTPUT_FILENAME_FDW)
            default_h5 = os.path.join(OUTPUT_DIR_SHE, OUTPUT_FILENAME_SHE)
            if os.path.exists(default_npz): self.npz_path.set(default_npz)
            if os.path.exists(default_h5): self.h5_path.set(default_h5)
        except ImportError:
            pass

    def _build_ui(self):
        ctrl_frame = ttk.Frame(self.root, padding=5)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(ctrl_frame, text="NPZ Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ctrl_frame, textvariable=self.npz_path, width=50).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(ctrl_frame, text="Browse...", command=lambda: self.npz_path.set(filedialog.askopenfilename(filetypes=[("NPZ Files", "*.npz")]))).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(ctrl_frame, text="H5 Path:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ctrl_frame, textvariable=self.h5_path, width=50).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(ctrl_frame, text="Browse...", command=lambda: self.h5_path.set(filedialog.askopenfilename(filetypes=[("H5 Files", "*.h5")]))).grid(row=1, column=2, padx=5, pady=2)
        
        self.btn_load = ttk.Button(ctrl_frame, text="Load & Compute", command=self.load_data)
        self.btn_load.grid(row=0, column=3, rowspan=2, padx=15, pady=2, sticky=tk.NSEW)
        
        self.lbl_status = ttk.Label(ctrl_frame, text="Ready.")
        self.lbl_status.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)

        bottom_frame = ttk.Frame(self.root, padding=5)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        freq_frame = ttk.Frame(bottom_frame)
        freq_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.lbl_freq = ttk.Label(freq_frame, text="Frequency: N/A", width=20, anchor=tk.W)
        self.lbl_freq.pack(side=tk.LEFT, padx=5)
        
        self.slider_freq = ttk.Scale(freq_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_freq_change, state=tk.DISABLED)
        self.slider_freq.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        thresh_frame = ttk.Frame(bottom_frame)
        thresh_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.lbl_thresh = ttk.Label(thresh_frame, text="Threshold: -30 dB", width=20, anchor=tk.W)
        self.lbl_thresh.pack(side=tk.LEFT, padx=5)
        
        self.slider_thresh = ttk.Scale(thresh_frame, from_=-90.0, to=0.0, orient=tk.HORIZONTAL, command=self.on_thresh_change, state=tk.DISABLED)
        self.slider_thresh.set(-30.0)
        self.slider_thresh.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.view = SpatialErrorView(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.view.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.plot_frame)
    def load_data(self):
        npz = self.npz_path.get()
        h5 = self.h5_path.get()
        if not os.path.exists(npz) or not os.path.exists(h5):
            messagebox.showerror("Error", "Please select valid NPZ and H5 files.")
            return
            
        self.btn_load.config(state=tk.DISABLED)
        self.lbl_status.config(text="Processing data, please wait...")
        self.root.update()
        
        threading.Thread(target=self._compute_thread, args=(npz, h5), daemon=True).start()
        
    def _compute_thread(self, npz_path, h5_path):
        try:
            with h5py.File(h5_path, "r") as h:
                she_dict = {
                    schema.FREQS: h[schema.FREQS][()],
                    schema.COEFFS: h[schema.COEFFS][()],
                    schema.N_USED: h[schema.N_USED][()].astype(int),
                    schema.ORIGINS_MM: h[schema.ORIGINS_MM][()] if schema.ORIGINS_MM in h else np.zeros((len(h[schema.FREQS][()]), 3))
                }

            parsed_npz = load_and_parse_npz(npz_path)
            measured_data = parsed_npz['complex_data']
            filenames = parsed_npz['filenames']
            r_static = parsed_npz['r_arr']
            th_static = parsed_npz['th_arr']
            ph_static = parsed_npz['ph_arr']
            f_all = parsed_npz['freqs']

            num_freqs = len(f_all)
            num_pts = len(filenames)
            p_measured_all = np.zeros((num_freqs, num_pts), dtype=np.complex128)
            for i, fn in enumerate(filenames):
                p_measured_all[:, i] = measured_data[fn]
                
            coords_sph = np.column_stack((np.degrees(th_static), np.degrees(ph_static), r_static))
            
            x, y, z, f, n, pm, pr = compute_spatial_error_data(she_dict, p_measured_all, coords_sph, SPEED_OF_SOUND)

            self.root.after(0, self._on_compute_done, x, y, z, f, n, pm, pr)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.lbl_status.config(text="Error occurred."))

    def _on_compute_done(self, x, y, z, f, n, pm, pr):
        self.x_coords, self.y_coords, self.z_coords = x, y, z
        self.freqs, self.n_used = f, n
        self.p_measured_all, self.p_reconstructed_all = pm, pr
        
        self.slider_freq.config(state=tk.NORMAL, to=len(self.freqs)-1)
        self.slider_thresh.config(state=tk.NORMAL)
        self.slider_freq.set(0)
        self.lbl_status.config(text="Computation complete.")
        self.btn_load.config(state=tk.NORMAL)
        self.update_plot()

    def on_freq_change(self, val):
        if self.freqs is not None:
            idx = int(float(val))
            self.lbl_freq.config(text=f"Frequency: {self.freqs[idx]:.1f} Hz")
            self.update_plot()
            
    def on_thresh_change(self, val):
        if self.freqs is not None:
            self.lbl_thresh.config(text=f"Threshold: {float(val):.1f} dB")
            self.update_plot()

    def update_plot(self):
        if self.freqs is None: return
        idx = int(float(self.slider_freq.get()))
        thresh = float(self.slider_thresh.get())
        
        self.view.update_view(
            self.x_coords, self.y_coords, self.z_coords, 
            self.freqs[idx], self.n_used[idx], 
            self.p_measured_all[idx, :], self.p_reconstructed_all[idx, :], 
            thresh
        )

def main():
    root = tk.Tk()
    
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
        
    app = SpatialErrorViewerApp(root)
    
    def on_closing():
        import gc
        plt.close(app.view.fig)
        root.destroy()
        gc.collect()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()