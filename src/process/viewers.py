import warnings
import os
import functools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.ticker as ticker
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from utils import natural_keys
import threading

# =============================================================================
# Helper to silence matplot log axis warnings
# =============================================================================
def silence_log_warnings(func):
    """Decorator to suppress persistent Matplotlib log-scale layout warnings."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*non-positive.*")
            warnings.simplefilter("ignore", category=UserWarning)
            return func(*args, **kwargs)
    return wrapper
    
# =============================================================================
# Helper for Standalone Tkinter Initialization
# =============================================================================
def get_tk_root(title="Viewer"):
    """Helper to create a Tk root or Toplevel if one already exists."""
    is_main_root = False
    if tk._default_root is None:
        root = tk.Tk()
        is_main_root = True
    else:
        root = tk.Toplevel()
    root.title(title)
    return root, is_main_root



# =============================================================================
# FDW Viewer (Stage 1)
# =============================================================================

class FDWView:
    """Pure MVC View for Stage 1 FDW Analysis."""
    def __init__(self, figsize=(12, 10)):
        self.fig = plt.figure(figsize=figsize)
        self.gs = self.fig.add_gridspec(3, 1, hspace=0.35, height_ratios=[3, 2, 1])
        self.ax1 = self.fig.add_subplot(self.gs[0])
        self.ax2 = self.fig.add_subplot(self.gs[1]) 
        self.ax3 = self.fig.add_subplot(self.gs[2], sharex=self.ax1)
        self.fig.subplots_adjust(bottom=0.08, right=0.88, left=0.08, top=0.95)

    @silence_log_warnings
    def update_view(self, fname, index, freqs, H_raw, H_smooth, meta_dict, wav_data, fs, fdw_f_min, fdw_rft_ms):
        m = meta_dict
        
        self.ax1.clear()
            
        mag_db_raw = 20 * np.log10(np.abs(H_raw) + 1e-12)
        
        if H_smooth is not None:
            mag_db_smooth = 20 * np.log10(np.abs(H_smooth) + 1e-12)
            self.ax1.semilogx(freqs, mag_db_raw, color='silver', lw=1.0, alpha=0.6, label="FDW Raw")
            self.ax1.semilogx(freqs, mag_db_smooth, 'r', lw=1.5, zorder=15, label="FDW Smoothed")
            self.ax1.legend(loc="upper right", fontsize=9)
        else:
            self.ax1.semilogx(freqs, mag_db_raw, 'k', lw=2, zorder=10, label="FDW Raw")
        
        f_trans = m['f_trans']
        f_anchor = m.get('f_lf_anchor', 200.0)
        
        self.ax1.axvspan(f_trans, fs/2, color='#E0F7FA', alpha=0.5)
        self.ax1.text(np.sqrt(f_trans*(fs/2)), -35, f"Fixed Window\n({fdw_rft_ms} ms)", 
                     color='#006064', fontsize=9, fontweight='bold', va='center', ha='center', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=20)
        
        cmap = plt.get_cmap('YlOrRd')
        num_octaves = np.log2(f_trans / fdw_f_min) if f_trans > fdw_f_min else 1
        
        self.ax2.clear()
        
        if wav_data is not None:
            t_ms = np.linspace(0, len(wav_data)/fs * 1000, len(wav_data))
            self.ax2.plot(t_ms, wav_data, color='#333333', lw=1, zorder=1, label="Raw IR")
            local_max = np.max(np.abs(wav_data))
            if local_max > 0: self.ax2.set_ylim(-local_max/0.9, local_max/0.9)
            peak_time_ms = m['t_peak'] * 1000
            self.ax2.axvline(peak_time_ms, color='red', linestyle='-', linewidth=1.5, label='Detected Peak')
        else:
            self.ax2.text(0.5, 0.5, "Error loading wav", ha='center', va='center')
            t_ms, local_max, peak_time_ms = np.array([]), 0, 0

        valid_indices = [i for i, fc in enumerate(m['f_centers']) if fdw_f_min <= fc <= f_trans]
        valid_centers = m['f_centers'][valid_indices]
        
        boundaries = [f_trans]
        if len(valid_centers) > 0 and abs(valid_centers[0] - f_trans) > 1.0: boundaries.append(valid_centers[0])
        if len(valid_centers) > 0: boundaries.extend(valid_centers[1:])
        boundaries.append(fdw_f_min)
        
        for k in range(len(boundaries) - 1):
            high_bound, low_bound = boundaries[k], boundaries[k+1]
            f_mid = np.sqrt(high_bound * low_bound)
            norm_pos = np.log2(f_trans / f_mid) / num_octaves
            c_val = 0.1 + (0.5 * np.clip(norm_pos, 0, 1))
            self.ax1.axvspan(low_bound, high_bound, color=cmap(c_val), alpha=0.4)

        valid_band_idx = 0 
        for i, fc in enumerate(m['f_centers']):
            if fc < fdw_f_min or fc > f_trans: continue
            
            norm_pos = np.log2(f_trans / fc) / num_octaves
            line_color = cmap(0.1 + (0.5 * np.clip(norm_pos, 0, 1)))
            self.ax1.axvline(fc, color=line_color, ls='-', alpha=0.8, linewidth=1.2)
            
            t_win_ms = m['t_windows'][i] * 1000
            label_y = -35 + (valid_band_idx % 2) * 4
            self.ax1.text(fc, label_y, f"{fc:.0f}Hz\n{t_win_ms:.1f}ms", ha='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'), zorder=20)
            
            if len(t_ms) > 0:
                win_len_ms = t_win_ms
                alpha = m['alpha_sched'][i]
                t_end_abs = peak_time_ms + win_len_ms
                t_taper_len = win_len_ms * alpha
                t_taper_start_abs = t_end_abs - t_taper_len
                
                t_flat = np.array([0, t_taper_start_abs])
                y_flat = np.array([local_max, local_max])
                
                if t_taper_len > 0:
                    t_curve = np.linspace(t_taper_start_abs, t_end_abs, 50)
                    norm_t = (t_curve - t_taper_start_abs) / t_taper_len
                    y_curve = local_max * 0.5 * (1 + np.cos(np.pi * norm_t))
                else:
                    t_curve, y_curve = np.array([t_end_abs, t_end_abs]), np.array([local_max, 0])

                t_full = np.concatenate((t_flat, t_curve))
                y_full = np.concatenate((y_flat, y_curve))
                valid = t_full <= t_ms[-1]
                
                self.ax2.plot(t_full[valid], y_full[valid], color=line_color, lw=2, alpha=0.8, zorder=10, label=f"{fc:.0f}Hz: {win_len_ms:.1f}ms")
            valid_band_idx += 1

        self.ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.8), fontsize=9)
        self.ax2.set_ylabel("Amplitude")
        self.ax2.set_xlabel("Time (ms)")
        self.ax2.set_title("Impulse Response Waveform & FDW Envelopes")
        self.ax2.grid(True, alpha=0.3)

        self.ax3.clear()
        f_raw, a_raw = m['f_centers'], m['alpha_sched']
        sort_idx = np.argsort(f_raw)
        f_plot, a_plot = f_raw[sort_idx], a_raw[sort_idx]
        
        if f_plot[0] > fdw_f_min:
            f_plot = np.insert(f_plot, 0, fdw_f_min)
            a_plot = np.insert(a_plot, 0, a_plot[0]) 
            
        self.ax3.semilogx(f_plot, a_plot, 'b-', lw=3)
        a_min, a_max = np.min(a_plot), np.max(a_plot)
        y_range = max(1e-3, a_max - a_min)
        
        self.ax3.axvline(f_trans, color='green', linestyle='--', alpha=0.7)
        self.ax3.text(f_trans, a_max - (y_range * 0.05), f" Alpha HF: {f_trans:.0f} Hz", color='green', fontsize=8, ha='left', va='top')
        self.ax3.axvline(f_anchor, color='green', linestyle='--', alpha=0.7)
        self.ax3.text(f_anchor, a_min + (y_range * 0.05), f"Alpha LF: {f_anchor:.0f} Hz ", color='green', fontsize=8, ha='right', va='bottom')

        self.ax3.set_ylabel("Alpha")
        self.ax3.set_xlim(fdw_f_min, fs/2)
        self.ax3.grid(True, which='both', alpha=0.3)
        self.ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())

        self.ax1.set_title(f"File [{index}]: {fname}\nFDW Magnitude")
        self.ax1.set_ylabel("Magnitude (dB)")
        self.ax1.set_xlim(fdw_f_min, fs/2)
        self.ax1.grid(True, which='both', alpha=0.3)
        self.ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
        
        self.fig.canvas.draw_idle()

class FDWViewer:
    """Standalone Tkinter wrapper for FDWView."""
    def __init__(self, freqs, data_dict, meta_dict, fs, ir_dir, crop_samples, data_dict_smooth=None, fdw_f_min=20.0, fdw_rft_ms=5.0):
        self.freqs, self.data_dict, self.data_dict_smooth = freqs, data_dict, data_dict_smooth
        self.meta_dict, self.fs, self.ir_dir = meta_dict, fs, ir_dir
        self.crop_samples, self.fdw_f_min, self.fdw_rft_ms = crop_samples, fdw_f_min, fdw_rft_ms
        self.filenames = sorted(list(data_dict.keys()), key=natural_keys)
        self.index = 0
        
        self.root, self.is_main = get_tk_root("FDW Viewer")
        self.root.geometry("1600x1200")
        
        # Safe destruction intercept
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.view = FDWView(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.view.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.root)
        
        ctrl_frame = ttk.Frame(self.root, padding=5)
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(ctrl_frame, text="< Previous", command=self.prev_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Next >", command=self.next_plot).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(ctrl_frame, text="Jump to ID:").pack(side=tk.LEFT, padx=(20, 5))
        self.var_id = tk.StringVar(value="0")
        ent_id = ttk.Entry(ctrl_frame, textvariable=self.var_id, width=8)
        ent_id.pack(side=tk.LEFT)
        ent_id.bind("<Return>", self.submit_id)
        
        ttk.Button(ctrl_frame, text="Exit Plot", command=self.on_closing).pack(side=tk.RIGHT, padx=5)
        
        self.refresh_plot()
        if self.is_main: self.root.mainloop()

    def refresh_plot(self):
        from stage1_fdwsmooth import load_and_prep_ir
        fname = self.filenames[self.index]
        H_raw, m = self.data_dict[fname], self.meta_dict[fname]
        H_smooth = self.data_dict_smooth[fname] if self.data_dict_smooth and fname in self.data_dict_smooth else None
        
        self.var_id.set(str(self.index))
        file_path = os.path.join(self.ir_dir, fname)
        try:
            wav_data, _ = load_and_prep_ir(file_path, crop_samples=self.crop_samples)
            if len(wav_data) < self.crop_samples:
                wav_data = np.pad(wav_data, (0, self.crop_samples - len(wav_data)), mode='constant')
        except Exception:
            wav_data = None
            
        self.view.update_view(fname, self.index, self.freqs, H_raw, H_smooth, m, wav_data, self.fs, self.fdw_f_min, self.fdw_rft_ms)

    def next_plot(self): self.index = (self.index + 1) % len(self.filenames); self.refresh_plot()
    def prev_plot(self): self.index = (self.index - 1) % len(self.filenames); self.refresh_plot()
    def submit_id(self, event=None): 
        try: 
            val = int(self.var_id.get())
            if 0 <= val < len(self.filenames): self.index = val; self.refresh_plot()
        except: pass

    def on_closing(self):
        import gc
        plt.close(self.view.fig)
        self.root.destroy()
        gc.collect()

# =============================================================================
# Acoustic Origin Viewer (Stage 2)
# =============================================================================

class FrequencyBrowser3DView:
    """Pure MVC View for full 3D grid scan validation."""
    def __init__(self, cfg, figsize=(10, 8)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.85, top=0.95) 
        self.cbar_ax = self.fig.add_axes([0.88, 0.2, 0.03, 0.6])
        self.cbar = None
        
        self.ax.set_xlim(cfg['x_bounds'])
        self.ax.set_ylim(cfg['y_bounds'])
        self.ax.set_zlim(cfg['z_bounds'])
        self.ax.set_box_aspect((cfg['x_bounds'][1]-cfg['x_bounds'][0], cfg['y_bounds'][1]-cfg['y_bounds'][0], cfg['z_bounds'][1]-cfg['z_bounds'][0]))
        self.ax.set_xlabel('X (Depth) mm')
        self.ax.set_ylabel('Y (Width) mm')
        self.ax.set_zlabel('Z (Height) mm')

        self.ax.view_init(elev=30, azim=135)
        self.fig.canvas.mpl_connect('motion_notify_event', self._enforce_turntable)

    def _enforce_turntable(self, event):
        if hasattr(self.ax, 'elev'):
            elev, azim = self.ax.elev, self.ax.azim
            if elev > 89.9: self.ax.view_init(elev=89.9, azim=azim)
            elif elev < -89.9: self.ax.view_init(elev=-89.9, azim=azim)

    def update_view(self, f, grid_3d, xv, yv, zv, final_c, plane, val):
        while self.ax.collections: self.ax.collections[0].remove()
        for line in self.ax.lines: line.remove()
            
        actual_val = val
        if grid_3d is not None:
            vmin, vmax = grid_3d.min(), grid_3d.max()
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            
            if plane == 'YZ (X Depth)':
                idx = (np.abs(xv - val)).argmin()
                actual_val = xv[idx]
                slice_2d = grid_3d[idx, :, :]
                Y_grid, Z_grid = np.meshgrid(yv, zv, indexing='ij')
                X_grid = np.full_like(Y_grid, actual_val)
            elif plane == 'XZ (Y Width)':
                idx = (np.abs(yv - val)).argmin()
                actual_val = yv[idx]
                slice_2d = grid_3d[:, idx, :]
                X_grid, Z_grid = np.meshgrid(xv, zv, indexing='ij')
                Y_grid = np.full_like(X_grid, actual_val)
            else:
                idx = (np.abs(zv - val)).argmin()
                actual_val = zv[idx]
                slice_2d = grid_3d[:, :, idx]
                X_grid, Y_grid = np.meshgrid(xv, yv, indexing='ij')
                Z_grid = np.full_like(X_grid, actual_val)

            colors = plt.cm.viridis(norm(slice_2d))
            self.ax.plot_surface(X_grid, Y_grid, Z_grid, facecolors=colors, rstride=1, cstride=1, shade=False, alpha=0.85)
            
            if self.cbar is None: self.cbar = self.fig.colorbar(sm, cax=self.cbar_ax)
            else: self.cbar.update_normal(sm)
            self.cbar.set_label('Residual Error (%)')
            grid_min_text = f"{np.min(grid_3d):.2f}%"
        else:
            grid_min_text = "N/A (Bypassed)"
            if self.cbar is not None:
                self.cbar.remove()
                self.cbar = None

        if final_c is not None:
            self.ax.scatter(final_c[0], final_c[1], final_c[2], c='yellow', marker='D', s=100, edgecolors='black', label='Simplex Final')

        self.ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=9)
        fc_str = f"({final_c[0]:.1f}, {final_c[1]:.1f}, {final_c[2]:.1f})" if final_c is not None else "None"
        self.ax.set_title(f"Frequency: {f:.1f} Hz | Grid Scan Minima: {grid_min_text}\nSearch Final: {fc_str}\nPlane: {plane} @ {actual_val:.1f} mm", fontweight='bold')
        self.fig.canvas.draw_idle()


class FrequencyBrowser3D:
    """Standalone Tkinter wrapper for FrequencyBrowser3DView."""
    def __init__(self, sweep_data, cfg, show_immediately=True):
        self.data, self.cfg = sweep_data, cfg
        self.freqs = sorted(list(sweep_data.keys()))
        self.index = 0
        
        self.root, self.is_main = get_tk_root("Interactive 3D Vector Search Cache")
        self.root.geometry("1100x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.view = FrequencyBrowser3DView(cfg, figsize=(9, 7))
        self.canvas = FigureCanvasTkAgg(self.view.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.root)
        
        ctrl_frame = ttk.Frame(self.root, padding=5)
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        nav_frame = ttk.Frame(ctrl_frame)
        nav_frame.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="< Prev (Hz)", command=self.prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next (Hz) >", command=self.next).pack(side=tk.LEFT, padx=2)
        
        plane_frame = ttk.LabelFrame(ctrl_frame, text="Plane")
        plane_frame.pack(side=tk.LEFT, padx=10)
        self.var_plane = tk.StringVar(value='YZ (X Depth)')
        for p in ('YZ (X Depth)', 'XZ (Y Width)', 'XY (Z Height)'):
            ttk.Radiobutton(plane_frame, text=p, variable=self.var_plane, value=p, command=self.update_radio).pack(side=tk.LEFT, padx=5)

        slide_frame = ttk.Frame(ctrl_frame)
        slide_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Label(slide_frame, text="Slice Pos:").pack(side=tk.LEFT)
        
        d = self.data[self.freqs[0]]
        self.x_arr = d['X_vals'] if d['X_vals'] is not None else np.array([cfg['x_bounds'][0], cfg['x_bounds'][1]])
        self.y_arr = d['Y_vals'] if d['Y_vals'] is not None else np.array([cfg['y_bounds'][0], cfg['y_bounds'][1]])
        self.z_arr = d['Z_vals'] if d['Z_vals'] is not None else np.array([cfg['z_bounds'][0], cfg['z_bounds'][1]])
        
        self.slider = ttk.Scale(slide_frame, from_=self.x_arr[0], to=self.x_arr[-1], orient=tk.HORIZONTAL, command=self.update_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.slice_val = 0.0

        self.refresh_plot()
        if show_immediately and self.is_main: self.root.mainloop()

    def refresh_plot(self):
        f = self.freqs[self.index]
        d = self.data[f]
        self.view.update_view(f, d['grid'], d['X_vals'], d['Y_vals'], d['Z_vals'], d['final_c'], self.var_plane.get(), self.slice_val)

    def update_slider(self, val): self.slice_val = float(val); self.refresh_plot()
    def prev(self): self.index = (self.index - 1) % len(self.freqs); self.refresh_plot()
    def next(self): self.index = (self.index + 1) % len(self.freqs); self.refresh_plot()
    
    def update_radio(self):
        label = self.var_plane.get()
        if label == 'YZ (X Depth)': vmin, vmax = self.x_arr[0], self.x_arr[-1]
        elif label == 'XZ (Y Width)': vmin, vmax = self.y_arr[0], self.y_arr[-1]
        else: vmin, vmax = self.z_arr[0], self.z_arr[-1]
        self.slider.config(from_=vmin, to=vmax)
        self.slice_val = np.clip(self.slice_val, vmin, vmax)
        self.slider.set(self.slice_val)
        self.refresh_plot()

    def on_closing(self):
        import gc
        plt.close(self.view.fig)
        self.root.destroy()
        gc.collect()

class CloudBrowser3DView:
    """Pure MVC View for 3D Coordinate Cloud."""
    def __init__(self, cfg, figsize=(8, 6)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(bottom=0.1, left=0.05, right=0.95, top=0.9)
        
        self.ax.set_xlim(cfg['x_bounds'])
        self.ax.set_ylim(cfg['y_bounds'])
        self.ax.set_zlim(cfg['z_bounds'])
        self.ax.set_box_aspect((cfg['x_bounds'][1]-cfg['x_bounds'][0], cfg['y_bounds'][1]-cfg['y_bounds'][0], cfg['z_bounds'][1]-cfg['z_bounds'][0]))
        self.ax.set_xlabel('X (Depth) mm')
        self.ax.set_ylabel('Y (Width) mm')
        self.ax.set_zlabel('Z (Height) mm')

        self.ax.view_init(elev=30, azim=135)
        self.fig.canvas.mpl_connect('motion_notify_event', self._enforce_turntable)
        self.scatter_all, self.scatter_hi = None, None

    def _enforce_turntable(self, event):
        if hasattr(self.ax, 'elev') and abs(self.ax.elev - 30.0) > 1e-3:
            self.ax.view_init(elev=30.0, azim=self.ax.azim)

    def update_view(self, freqs, pts_x, pts_y, pts_z, active_idx):
        if not freqs: return
        if self.scatter_all is not None: self.scatter_all.remove()
        if self.scatter_hi is not None: self.scatter_hi.remove()
        
        self.scatter_all = self.ax.scatter(pts_x, pts_y, pts_z, c='blue', s=30, alpha=0.6, label='Nelder Final')
        
        f = freqs[active_idx]
        pt = (pts_x[active_idx], pts_y[active_idx], pts_z[active_idx])
        self.scatter_hi = self.ax.scatter([pt[0]], [pt[1]], [pt[2]], c='red', s=120, alpha=1.0, label='Highlighted')
        
        self.ax.legend(loc='upper right')
        self.ax.set_title(f"Frequency: {f:.1f} Hz\nCoordinate: ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})", fontweight='bold')
        self.fig.canvas.draw_idle()


class CloudBrowser3D:
    """Standalone Tkinter wrapper for CloudBrowser3DView."""
    def __init__(self, sweep_data, cfg, show_immediately=True):
        self.data, self.cfg = sweep_data, cfg
        self.freqs = sorted([f for f, d in sweep_data.items() if d['final_c'] is not None])
        self.index = 0
        
        self.root, self.is_main = get_tk_root("3D Coordinate Cloud Viewer")
        self.root.geometry("900x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.view = CloudBrowser3DView(cfg, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.view.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.root)
        
        ctrl_frame = ttk.Frame(self.root, padding=5)
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.lbl_freq = ttk.Label(ctrl_frame, text="Freq:")
        self.lbl_freq.pack(side=tk.LEFT, padx=5)
        self.slider = ttk.Scale(ctrl_frame, from_=0, to=max(0, len(self.freqs)-1), orient=tk.HORIZONTAL, command=self.update_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.refresh_data()
        if show_immediately and self.is_main: self.root.mainloop()

    def update_slider(self, val):
        self.index = int(float(val))
        if len(self.freqs) > 0: self.lbl_freq.config(text=f"{self.freqs[self.index]:.1f} Hz")
        self.update_plot()

    def refresh_data(self):
        self.pts_x = [self.data[f]['final_c'][0] for f in self.freqs]
        self.pts_y = [self.data[f]['final_c'][1] for f in self.freqs]
        self.pts_z = [self.data[f]['final_c'][2] for f in self.freqs]
        self.update_plot()

    def update_plot(self):
        if len(self.freqs) > 0: self.view.update_view(self.freqs, self.pts_x, self.pts_y, self.pts_z, self.index)

    def on_closing(self):
        import gc
        plt.close(self.view.fig)
        self.root.destroy()
        gc.collect()

class ValidationView:
    """Pure MVC View for Origin Search validation plotting."""
    def __init__(self, figsize=(10, 10)):
        self.fig, (self.ax_x, self.ax_y, self.ax_z) = plt.subplots(3, 1, figsize=figsize, sharex=True)
        self.fig.subplots_adjust(bottom=0.1, top=0.95, right=0.95, left=0.1)

    @silence_log_warnings
    def update_view(self, history_freq, history_search, history_orig, history_err_search, history_err_orig):
        axes = [self.ax_x, self.ax_y, self.ax_z]
        labels = ['X (Depth)', 'Y (Width)', 'Z (Height)']
               
        for ax in axes: 
            ax.clear()
        
        for i, sax in enumerate(axes):
            sax.format_xdata = lambda x: f"{x:.1f} Hz"
            sax.format_ydata = lambda y: f"{y:.2f} mm"

            sax.plot(history_freq, history_search[i], 'b-', alpha=0.5, label='Search Path' if i == 0 else "")
            
            mod_idx = [j for j in range(len(history_freq)) if not np.allclose(
                [history_orig[0][j], history_orig[1][j], history_orig[2][j]], 
                [history_search[0][j], history_search[1][j], history_search[2][j]]
            )]
            unmod_idx = [j for j in range(len(history_freq)) if j not in mod_idx]
            
            if unmod_idx:
                sax.plot([history_freq[j] for j in unmod_idx], [history_search[i][j] for j in unmod_idx], 'bo', alpha=0.6, label='Original Result' if i == 0 else "")
            if mod_idx:
                sax.plot([history_freq[j] for j in mod_idx], [history_orig[i][j] for j in mod_idx], 'bx', alpha=0.4, label='Previous Result' if i == 0 else "")
                sax.plot([history_freq[j] for j in mod_idx], [history_search[i][j] for j in mod_idx], 'ro', alpha=0.9, label='New Optimal Result' if i == 0 else "")
                         
            for j, f in enumerate(history_freq):
                if j in mod_idx:
                    sax.plot([f, f], [history_orig[i][j], history_search[i][j]], 'r--', alpha=0.5)
                    sax.text(f, history_orig[i][j], f"{history_err_orig[j]:.2f}%", color='blue', fontsize=8, ha='left', va='top', alpha=0.5)
                    sax.text(f, history_search[i][j], f"{history_err_search[j]:.2f}%", color='red', fontsize=8, ha='right', va='bottom', alpha=0.9)
                else:
                    sax.text(f, history_search[i][j], f"{history_err_search[j]:.2f}%", color='blue', fontsize=8, ha='right', va='bottom', alpha=0.8)

            sax.set_xscale('log')
            sax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}kHz" if x >= 1000 else f"{int(x)}Hz"))
            sax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
            sax.set_ylabel(f'{labels[i]} (mm)')
            sax.grid(True, which='both', ls='--', alpha=0.3)
            if i == 0:
                sax.set_title('Frequency Sweep Validation: Coordinates & Residual Error Labels', fontweight='bold')
                sax.legend(loc='upper right')

        self.ax_z.set_xlabel('Frequency (Hz)')
        self.fig.canvas.draw_idle()


class ValidationUI:
    """Standalone Tkinter wrapper for ValidationView."""
    def __init__(self, sweep_results, f_all, keys, d_dict, parsed_geom, cfg):
        self.sweep_results, self.f_all, self.keys, self.d_dict, self.cfg = sweep_results, f_all, keys, d_dict, cfg
        self.r_arr, self.th_arr, self.ph_arr = parsed_geom
        
        # Store checkbox references securely to prevent premature garbage collection
        self.vars_dict = {}
        
        for f, d in self.sweep_results.items():
            if 'original_c' not in d and d['final_c'] is not None:
                d['original_c'] = np.copy(d['final_c'])
        
        self.root, self.is_main = get_tk_root("Search Validation Summary")
        self.root.geometry("1100x900")
        self.root.protocol("WM_DELETE_WINDOW", self.accept)
        
        self.view = ValidationView(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.view.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.root)
        
        ctrl_frame = ttk.Frame(self.root, padding=5)
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(ctrl_frame, text="Edit Coordinates", command=self.open_edit_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Rescan Frequencies", command=self.open_rescan_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Accept & Save", command=self.accept).pack(side=tk.RIGHT, padx=5)
        
        self.accepted = False
        self.refresh_plot()
        
        if self.cfg['enable_full_grid_scan']: self.browser3d = FrequencyBrowser3D(self.sweep_results, self.cfg, show_immediately=False)
        else: self.browser3d = CloudBrowser3D(self.sweep_results, self.cfg, show_immediately=False)
            
        if self.is_main: self.root.mainloop()
        else: self.root.wait_window()

    def refresh_plot(self):
        from stage2_centre_origin import get_order_for_frequency, solve_physics_3d
        
        hf, hs, ho, hes, heo = [], [[], [], []], [[], [], []], [], []
        for f_hz in sorted(self.sweep_results.keys()):
            d = self.sweep_results[f_hz]
            if d['final_c'] is not None:
                hf.append(f_hz)
                orig_c = d.get('original_c', d['final_c'])
                for i in range(3): hs[i].append(d['final_c'][i]); ho[i].append(orig_c[i])
                
                idx = (np.abs(self.f_all - f_hz)).argmin()
                P = [self.d_dict[k][idx] for k in self.keys]
                N = get_order_for_frequency(f_hz, self.cfg.get('manual_order_table',{}), self.cfg.get('target_n_max_origins',4), self.cfg.get('N_grid',999))
                pc = ([(float(f_hz), float(2 * np.pi * f_hz / self.cfg['speed_of_sound']), np.conj(np.array(P)))], self.r_arr, self.th_arr, self.ph_arr, N, self.cfg)
                
                e_s = solve_physics_3d(d['final_c'][0], d['final_c'][1], d['final_c'][2], pc)
                hes.append(e_s)
                heo.append(e_s if np.allclose(orig_c, d['final_c']) else solve_physics_3d(orig_c[0], orig_c[1], orig_c[2], pc))
                
        self.view.update_view(hf, hs, ho, hes, heo)

    def open_edit_dialog(self):
        top = tk.Toplevel(self.root)
        top.title("Edit Coordinates")
        top.geometry("400x300")
        
        tree = ttk.Treeview(top, columns=('Freq', 'X', 'Y', 'Z'), show='headings')
        tree.heading('Freq', text='Freq (Hz)')
        tree.heading('X', text='X')
        tree.heading('Y', text='Y')
        tree.heading('Z', text='Z')
        
        # Explicitly configure column widths so they don't clip at spawn
        tree.column('Freq', width=80, anchor='center')
        tree.column('X', width=80, anchor='center')
        tree.column('Y', width=80, anchor='center')
        tree.column('Z', width=80, anchor='center')

        for f in sorted(self.sweep_results.keys()):
            d = self.sweep_results[f]
            if d['final_c'] is not None: tree.insert('', tk.END, values=(f"{f:.1f}", f"{d['final_c'][0]:.2f}", f"{d['final_c'][1]:.2f}", f"{d['final_c'][2]:.2f}"))
        tree.pack(fill=tk.BOTH, expand=True)
        
        frame = tk.Frame(top); frame.pack(pady=5)
        tk.Label(frame, text="X:").pack(side=tk.LEFT)
        ent_x = tk.Entry(frame, width=8); ent_x.pack(side=tk.LEFT)
        tk.Label(frame, text="Y:").pack(side=tk.LEFT)
        ent_y = tk.Entry(frame, width=8); ent_y.pack(side=tk.LEFT)
        tk.Label(frame, text="Z:").pack(side=tk.LEFT)
        ent_z = tk.Entry(frame, width=8); ent_z.pack(side=tk.LEFT)
        
        def on_select(evt):
            sel = tree.selection()
            if not sel: return
            vals = tree.item(sel[0], 'values')
            ent_x.delete(0, tk.END); ent_x.insert(0, vals[1])
            ent_y.delete(0, tk.END); ent_y.insert(0, vals[2])
            ent_z.delete(0, tk.END); ent_z.insert(0, vals[3])
        tree.bind('<<TreeviewSelect>>', on_select)
        
        def apply_edit():
            sel = tree.selection()
            if not sel: return
            try: x, y, z = float(ent_x.get()), float(ent_y.get()), float(ent_z.get())
            except ValueError: return
            vals = list(tree.item(sel[0], 'values'))
            vals[1], vals[2], vals[3] = f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"
            tree.item(sel[0], values=vals)
            exact_f = min(self.sweep_results.keys(), key=lambda k: abs(k - float(vals[0])))
            self.sweep_results[exact_f]['final_c'] = np.array([x, y, z])
            self.refresh_plot()
            if hasattr(self.browser3d, 'refresh_data'): self.browser3d.refresh_data()
            else: self.browser3d.refresh_plot()
            
        tk.Button(frame, text="Apply", command=apply_edit).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Close", command=top.destroy).pack(pady=5)

    def open_rescan_dialog(self):
        from stage2_centre_origin import get_order_for_frequency, generate_3d_landscape_volumetric, run_simplex_descent_3d, solve_physics_3d
        top = tk.Toplevel(self.root)
        top.title("Rescan Frequencies")
        top.geometry("450x400")
        frame = tk.Frame(top); frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        sf = tk.Frame(canvas)
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.vars_dict.clear() # Reset dictionary for fresh list
        for i, f in enumerate(sorted(self.sweep_results.keys())):
            var = tk.BooleanVar()
            self.vars_dict[f] = var
            row, col = divmod(i, 4)
            tk.Checkbutton(sf, text=f"{f:.1f} Hz", variable=var).grid(row=row, column=col, sticky='w', padx=5, pady=2)
            
        def do_rescan():
            freqs_to_rescan = [f for f, var in self.vars_dict.items() if var.get()]
            top.destroy()
            self.root.update_idletasks() # Clear destruction events
            self.root.update()
            
            if not freqs_to_rescan: return
                
            wait_win = tk.Toplevel(self.root)
            wait_win.title("Please Wait")
            tk.Label(wait_win, text="Scanning, this may take some time.\nCheck main app 'Processing CLI' for progress.", font=("Arial", 11)).pack(expand=True, padx=20, pady=20)
            
            # Force the wait window to fully draw before thread locks the GIL
            wait_win.update_idletasks()
            wait_win.update()
            
            def rescan_thread_task():
                for f in freqs_to_rescan:
                    idx = (np.abs(self.f_all - f)).argmin()
                    P = [self.d_dict[k][idx] for k in self.keys]
                    N = get_order_for_frequency(f, self.cfg.get('manual_order_table',{}), self.cfg.get('target_n_max_origins',4), self.cfg.get('N_grid',999))
                    pc = ([(float(f), float(2*np.pi*f/self.cfg['speed_of_sound']), np.conj(np.array(P)))], self.r_arr, self.th_arr, self.ph_arr, N, self.cfg)
                    x_vals, y_vals, z_vals, grid = generate_3d_landscape_volumetric(pc)
                    min_idx = np.unravel_index(np.argmin(grid), grid.shape)
                    tc = (x_vals[min_idx[0]], y_vals[min_idx[1]], z_vals[min_idx[2]])
                    opt_c, path = run_simplex_descent_3d(tc, pc)
                    self.sweep_results[f].update({'grid': grid, 'X_vals': x_vals, 'Y_vals': y_vals, 'Z_vals': z_vals, 'true': tc, 'final_c': opt_c, 'path': path})
                self.root.after(0, finish_rescan)
                
            def finish_rescan():
                wait_win.destroy()
                self.refresh_plot()
                if hasattr(self.browser3d, 'refresh_data'): self.browser3d.refresh_data()
                else: self.browser3d.refresh_plot()
                if not self.cfg['enable_full_grid_scan']:
                    self.rescan_browser = FrequencyBrowser3D({f: self.sweep_results[f] for f in freqs_to_rescan}, self.cfg, show_immediately=False)
                    self.rescan_browser.root.deiconify()
                    
            import gc
            gc.collect()

            # Execute thread slightly delayed so Tkinter has time to breathe
            self.root.after(100, lambda: threading.Thread(target=rescan_thread_task, daemon=True).start())
            
        tk.Button(top, text="Run Full Grid Scan", command=do_rescan).pack(pady=5)

    def accept(self):
        """Cleanly destroys the UI and forces garbage collection on the main thread."""
        import gc
        
        self.accepted = True
        
        # 1. Clear out variable references to prevent Tkinter GC tracebacks
        if hasattr(self, 'vars_dict'):
            self.vars_dict.clear()
            
        # 2. Explicitly close Matplotlib figures to destroy backend Tkinter canvases safely
        if hasattr(self, 'view') and hasattr(self.view, 'fig'):
            plt.close(self.view.fig)
            
        if hasattr(self, 'browser3d'):
            if hasattr(self.browser3d, 'view') and hasattr(self.browser3d.view, 'fig'):
                plt.close(self.browser3d.view.fig)
            if hasattr(self.browser3d, 'root'):
                self.browser3d.root.destroy()
                
        if hasattr(self, 'rescan_browser'):
            if hasattr(self.rescan_browser, 'view') and hasattr(self.rescan_browser.view, 'fig'):
                plt.close(self.rescan_browser.view.fig)
            if hasattr(self.rescan_browser, 'root'):
                self.rescan_browser.root.destroy()
                
        # 3. Destroy main root
        self.root.destroy()
        
        # 4. Force a garbage collection pass in the main thread before returning to the driving script
        gc.collect()

# =============================================================================
# SHE Results Viewer (Stage 4)
# =============================================================================

class SHEResultsView:
    """Pure MVC View for Stage 4 SHE Results."""
    def __init__(self, figsize=(10, 6)):
        self.fig, self.ax_err = plt.subplots(figsize=figsize)
        self.fig.subplots_adjust(bottom=0.2)
        
        self.line_err, = self.ax_err.plot([], [], 'b-', linewidth=1.5, alpha=0.8, label='Fit Error')
        self.line_thresh = self.ax_err.axhline(y=-20.0, color='red', linestyle='--', alpha=0.7, label='10% or -20dB Error')
        
        self.ax_err.set_xscale('log')
        self.ax_err.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}kHz" if x >= 1000 else f"{int(x)}Hz"))
        self.ax_err.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        self.ax_err.grid(True, which='both', ls='--', alpha=0.5)
        self.ax_err.set_xlabel('Frequency')
        
        self.ax_n = self.ax_err.twiny()
        self.ax_n.set_xscale('log')
        
        self.fig_cond = None
        self.ax_cond = None

    @silence_log_warnings
    def update_view(self, f_sel, pct_error, db_error, boundaries, tick_locs, tick_labels, min_n, max_n, is_db=True):
        self.ax_err.set_xlim(f_sel[0], f_sel[-1])
        self.ax_n.set_xlim(self.ax_err.get_xlim())
        
        if is_db:
            self.line_err.set_data(f_sel, db_error)
            self.line_thresh.set_ydata([-20.0, -20.0])
            self.ax_err.set_ylabel('Fit Error (dB)')
            self.ax_err.format_ydata = lambda y: f"{y:.1f} dB"
            self.ax_err.set_ylim(auto=True)
        else:
            self.line_err.set_data(f_sel, pct_error)
            self.line_thresh.set_ydata([10.0, 10.0])
            self.ax_err.set_ylabel('Fit Error (%)')
            self.ax_err.format_ydata = lambda y: f"{y:.2f}%"
            self.ax_err.set_ylim(bottom=0, top=max(20.0, np.max(pct_error) * 1.1))

        self.ax_err.relim()
        self.ax_err.autoscale_view()
        
        for p in reversed(self.ax_err.patches):
            p.remove()
            
        cmap = plt.get_cmap('YlOrRd')
        for (f_start, f_end, n) in boundaries:
            norm_n = (n - min_n) / max(1, max_n - min_n)
            c_val = 0.1 + (0.5 * np.clip(norm_n, 0, 1))
            self.ax_err.axvspan(f_start, f_end, facecolor=cmap(c_val), alpha=0.4, edgecolor='none', linewidth=0)

        self.ax_n.set_xticks(tick_locs)
        self.ax_n.set_xticklabels(tick_labels)
        self.ax_n.set_xlabel('Order N')
        self.ax_n.minorticks_off()
        
        self.ax_err.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        self.fig.canvas.draw_idle()

    @silence_log_warnings
    def update_cond_view(self, f_sel, res_cond):
        if self.fig_cond is None:
            self.fig_cond, self.ax_cond = plt.subplots(figsize=(10, 6))
            self.fig_cond.subplots_adjust(bottom=0.2)
            self.line_cond, = self.ax_cond.plot([], [], 'g-', linewidth=1.5, alpha=0.8, label='Condition Number')
            self.ax_cond.set_xscale('log')
            self.ax_cond.set_yscale('log')
            self.ax_cond.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}kHz" if x >= 1000 else f"{int(x)}Hz"))
            self.ax_cond.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
            self.ax_cond.grid(True, which='both', ls='--', alpha=0.5)
            self.ax_cond.set_xlabel('Frequency')
            self.ax_cond.set_ylabel('Condition Number')
            self.ax_cond.set_title('SHE Solver: Matrix Condition Number vs Frequency')
            self.ax_cond.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
        
        self.ax_cond.set_xlim(f_sel[0], f_sel[-1])
        self.line_cond.set_data(f_sel, res_cond)
        self.ax_cond.relim()
        self.ax_cond.autoscale_view()
        self.fig_cond.canvas.draw_idle()

def plot_she_results(f_sel: np.ndarray, pct_error: np.ndarray, res_cond: np.ndarray, n_used: np.ndarray, condition_metrics: bool, P_measured: np.ndarray = None, resid_vec: np.ndarray = None, save_path_prefix: str = None, she_dict: dict = None, coords_sph: np.ndarray = None, c_sound: float = 343.0) -> None:
    """Standalone Tkinter Wrapper for SHE Results."""
    root, is_main = get_tk_root("SHE Solver Results")
    root.geometry("1000x700")
    
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    tab_err = ttk.Frame(notebook)
    notebook.add(tab_err, text="Fit Error")
    
    view = SHEResultsView(figsize=(10, 5))
    canvas_err = FigureCanvasTkAgg(view.fig, master=tab_err)
    canvas_err.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    NavigationToolbar2Tk(canvas_err, tab_err)
    
    state = {'is_db': True, 'pct_error': pct_error.copy(), 'db_error': 20 * np.log10(np.clip(pct_error / 100.0, 1e-12, None)), 'view_spatial': None}
    
    min_n, max_n = np.min(n_used), np.max(n_used)
    boundaries, tick_locs, tick_labels = [], [], []
    current_n, start_f = n_used[0], f_sel[0]
    
    for i in range(1, len(f_sel)):
        if n_used[i] != current_n:
            boundaries.append((start_f, f_sel[i], current_n))
            current_n = n_used[i]
            start_f = f_sel[i]
    boundaries.append((start_f, f_sel[-1], current_n))
    
    for (f_start, f_end, n) in boundaries:
        tick_locs.append(np.sqrt(f_start * f_end))
        tick_labels.append(str(n))
        
    def refresh():
        view.update_view(f_sel, state['pct_error'], state['db_error'], boundaries, tick_locs, tick_labels, min_n, max_n, state['is_db'])
        
    ctrl_frame = ttk.Frame(tab_err, padding=5)
    ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    def toggle_clicked():
        state['is_db'] = not state['is_db']
        refresh()
        
    ttk.Button(ctrl_frame, text="Toggle % / dB", command=toggle_clicked).pack(side=tk.RIGHT, padx=5)
    
    if P_measured is not None and resid_vec is not None:
        eps = np.finfo(float).eps
        mags_db = 20 * np.log10(np.abs(P_measured) + eps)
        max_mags_db = np.max(mags_db, axis=1, keepdims=True)
        
        slide_frame = ttk.Frame(ctrl_frame)
        slide_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        lbl_thresh = ttk.Label(slide_frame, text="Threshold: -90 dB")
        lbl_thresh.pack(side=tk.LEFT, padx=5)
        
        def update_thresh(val):
            v = float(val)
            lbl_thresh.config(text=f"Threshold: {v:.0f} dB")
            cutoff_db = max_mags_db + v
            mask = mags_db >= cutoff_db
            resid_masked = np.where(mask, resid_vec, 0)
            p_masked = np.where(mask, P_measured, 0)
            bn = np.linalg.norm(p_masked, axis=1)
            rn = np.linalg.norm(resid_masked, axis=1)
            state['pct_error'] = rn / np.maximum(bn, 1e-20) * 100.0
            state['db_error'] = 20 * np.log10(np.clip(state['pct_error'] / 100.0, 1e-12, None))
            refresh()
            
        thresh_slider = ttk.Scale(slide_frame, from_=-90.0, to=0.0, orient=tk.HORIZONTAL, command=update_thresh)
        thresh_slider.set(-90.0)
        thresh_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    if condition_metrics:
        tab_cond = ttk.Frame(notebook)
        notebook.add(tab_cond, text="Condition Number")
        view.update_cond_view(f_sel, res_cond)
        canvas_cond = FigureCanvasTkAgg(view.fig_cond, master=tab_cond)
        canvas_cond.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas_cond, tab_cond)

    if she_dict is not None and coords_sph is not None and P_measured is not None:
        tab_spatial = ttk.Frame(notebook)
        notebook.add(tab_spatial, text="Spatial Error")
        
        lbl_spatial_status = ttk.Label(tab_spatial, text="Click 'Compute Spatial Error' to process data.", font=("Arial", 12))
        lbl_spatial_status.pack(expand=True)
        
        btn_compute = ttk.Button(tab_spatial, text="Compute Spatial Error")
        btn_compute.pack(pady=10)
        
        spatial_frame = ttk.Frame(tab_spatial)
        
        def run_spatial_compute():
            btn_compute.pack_forget()
            lbl_spatial_status.config(text="Processing data, please wait...")
            root.update()
            threading.Thread(target=_spatial_compute_thread, daemon=True).start()
            
        def _spatial_compute_thread():
            import sys, os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'misc')))
            try:
                from spatial_error_viewer import compute_spatial_error_data, SpatialErrorView
                x, y, z, f, n, pm, pr = compute_spatial_error_data(she_dict, P_measured, coords_sph, c_sound)
                root.after(0, lambda: _on_spatial_compute_done(x, y, z, f, n, pm, pr, SpatialErrorView))
            except Exception as e:
                root.after(0, lambda: lbl_spatial_status.config(text=f"Error computing spatial data: {e}"))
                
        def _on_spatial_compute_done(x_coords, y_coords, z_coords, sub_freqs, sub_n_used, p_measured_all, p_reconstructed_all, SpatialErrorView):
            lbl_spatial_status.pack_forget()
            spatial_frame.pack(fill=tk.BOTH, expand=True)
            
            ctrl_spatial = ttk.Frame(spatial_frame, padding=5)
            ctrl_spatial.pack(side=tk.BOTTOM, fill=tk.X)
            
            lbl_freq = ttk.Label(ctrl_spatial, text="Frequency: N/A")
            lbl_freq.pack(side=tk.TOP, anchor=tk.W, padx=5)
            
            slider_freq = ttk.Scale(ctrl_spatial, from_=0, to=max(0, len(sub_freqs)-1), orient=tk.HORIZONTAL)
            slider_freq.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
            
            lbl_thresh = ttk.Label(ctrl_spatial, text="Threshold: -30 dB")
            lbl_thresh.pack(side=tk.TOP, anchor=tk.W, padx=5)
            
            slider_thresh = ttk.Scale(ctrl_spatial, from_=-90.0, to=0.0, orient=tk.HORIZONTAL)
            slider_thresh.set(-30.0)
            slider_thresh.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
            
            state['view_spatial'] = SpatialErrorView(figsize=(10, 5))
            canvas_spatial = FigureCanvasTkAgg(state['view_spatial'].fig, master=spatial_frame)
            canvas_spatial.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            NavigationToolbar2Tk(canvas_spatial, spatial_frame)
            
            def update_spatial(*args):
                idx = int(float(slider_freq.get()))
                thresh = float(slider_thresh.get())
                lbl_freq.config(text=f"Frequency: {sub_freqs[idx]:.1f} Hz")
                lbl_thresh.config(text=f"Threshold: {thresh:.1f} dB")
                state['view_spatial'].update_view(x_coords, y_coords, z_coords, sub_freqs[idx], sub_n_used[idx], p_measured_all[idx, :], p_reconstructed_all[idx, :], thresh)
                
            slider_freq.config(command=update_spatial)
            slider_thresh.config(command=update_spatial)
            slider_freq.set(0)
            update_spatial()
            
        btn_compute.config(command=run_spatial_compute)

    def on_closing():
        import gc
        plt.close(view.fig)
        if condition_metrics and view.fig_cond:
            plt.close(view.fig_cond)
        if state['view_spatial'] is not None:
            plt.close(state['view_spatial'].fig)
        root.destroy()
        gc.collect()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    refresh()
    
    if save_path_prefix:
        try:
            view.fig.savefig(f"{save_path_prefix}_residual_error.png", dpi=150, bbox_inches='tight')
            if condition_metrics and view.fig_cond:
                view.fig_cond.savefig(f"{save_path_prefix}_condition_number.png", dpi=150, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save plots: {e}")
            
    if is_main: root.mainloop()

# =============================================================================
# Origin Stage 5 Viewer
# =============================================================================

class Stage5Viewer:
    """
    Pure 3D coordinate viewer for Stage 5 (Pressure Extraction).
    Visualizes the DUT (Speaker) as a bounding box, the mic coordinates,
    and the reference axis for zero theta/phi.
    
    MVC View Component: Knows nothing about Tkinter, NiceGUI, or external state.
    """
    def __init__(self, figsize=(7, 6), dpi=100):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.fig.patch.set_facecolor('white')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.08)
        
        # Set static axis properties ONCE so camera angles aren't reset during updates
        self.ax.set_xlabel('X (Depth) [m]')
        self.ax.set_ylabel('Y (Width) [m]')
        self.ax.set_zlabel('Z (Height) [m]')
        
        # Set the initial camera angle (elevation and azimuth)
        self.ax.view_init(elev=30, azim=-45)
        
        # Track drawn objects so we can remove them cleanly without clearing the axes
        self._drawn_artists = []

    def update_view(self, box_dims=(0.2, 0.3, 0.4), mic_coords_xyz=None, 
                    ref_origin=(0.0, 0.0, 0.0), zero_theta_deg=90.0, zero_phi_deg=0.0):
        """
        Updates the 3D plot with new parameters without resetting camera rotation.
        """
        # Clean up old plot elements cleanly
        for artist in self._drawn_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._drawn_artists.clear()

        # 1. Draw DUT Box centered at (0,0,0)
        width_y, depth_x, height_z = box_dims
        dx, dy, dz = depth_x / 2.0, width_y / 2.0, height_z / 2.0
        
        # Define the 8 vertices of the box
        r = [-1, 1]
        pts = np.array([[x*dx, y*dy, z*dz] for x in r for y in r for z in r])
        
        # Define the 6 faces of the box
        faces = [
            [pts[0], pts[1], pts[3], pts[2]], # X- (Back)
            [pts[4], pts[5], pts[7], pts[6]], # X+ (Front)
            [pts[0], pts[1], pts[5], pts[4]], # Y- (Right)
            [pts[2], pts[3], pts[7], pts[6]], # Y+ (Left)
            [pts[0], pts[2], pts[6], pts[4]], # Z- (Bottom)
            [pts[1], pts[3], pts[7], pts[5]], # Z+ (Top)
        ]
        
        face_colors = ['cyan', 'blue', 'cyan', 'cyan', 'cyan', 'cyan']
        
        poly3d = Poly3DCollection(faces, alpha=0.15, facecolors=face_colors, edgecolors='gray', linewidths=1)
        self.ax.add_collection3d(poly3d)
        self._drawn_artists.append(poly3d)
        
        # Dummy point to ensure box appears in the legend
        dut_scatter = self.ax.scatter([0], [0], [0], c='cyan', marker='s', s=20, alpha=0.5, label='DUT')
        self._drawn_artists.append(dut_scatter)

        max_radius = max(dx, dy, dz)

        # 2. Draw Mic Coordinates
        if mic_coords_xyz is not None and len(mic_coords_xyz) > 0:
            mic_coords_xyz = np.array(mic_coords_xyz)
            mic_scatter = self.ax.scatter(mic_coords_xyz[:, 0], mic_coords_xyz[:, 1], mic_coords_xyz[:, 2], 
                            c='blue', marker='o', s=15, alpha=0.7, label='Mic Positions')
            self._drawn_artists.append(mic_scatter)
            
            dists = np.linalg.norm(mic_coords_xyz, axis=1)
            if len(dists) > 0:
                max_radius = max(max_radius, np.max(dists))

        # 3. Draw Reference Origin and Axes
        ox, oy, oz = ref_origin
        ref_scatter = self.ax.scatter([ox], [oy], [oz], c='red', marker='P', s=40, label='Ref Origin (Offset)')
        self._drawn_artists.append(ref_scatter)
        
        # Convert angles to radians
        th = np.radians(zero_theta_deg)
        ph = np.radians(zero_phi_deg)
        
        # Primary reference axis (Forward / 0 degrees)
        arrow_len = max_radius * 0.8 if max_radius > 0 else 1.0
        fx = arrow_len * np.sin(th) * np.cos(ph)
        fy = arrow_len * np.sin(th) * np.sin(ph)
        fz = arrow_len * np.cos(th)
        
        quiv = self.ax.quiver(ox, oy, oz, fx, fy, fz, color='red', arrow_length_ratio=0.1, 
                       linewidth=2, label='Zero Axis (Front)')
        self._drawn_artists.append(quiv)

        # Place a multi-column legend at the bottom of the plot area
        leg = self.ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize='small', frameon=False)
        self._drawn_artists.append(leg)
        
        self._set_axes_equal()
        self.fig.canvas.draw_idle()

    def _set_axes_equal(self):
        # Hack to force equal aspect ratio in matplotlib 3D plots
        limits = np.array([self.ax.get_xlim3d(), self.ax.get_ylim3d(), self.ax.get_zlim3d()])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        self.ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        self.ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        self.ax.set_zlim3d([origin[2] - radius, origin[2] + radius])