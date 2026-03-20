import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, Slider, RadioButtons
import matplotlib.ticker as ticker
import tkinter as tk
from tkinter import ttk
from utils import natural_keys

def plot_she_results(f_sel: np.ndarray, pct_error: np.ndarray, res_cond: np.ndarray, condition_metrics: bool) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(f_sel, pct_error, 'b-', linewidth=1.5, alpha=0.8, label='Fit Error')
    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='10% or -20dB Error')
    plt.xscale('log')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}kHz" if x >= 1000 else f"{int(x)}Hz"))
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.format_xdata = lambda x: f"{x:.1f} Hz"
    ax.format_ydata = lambda y: f"{y:.2f}%"
    
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.xlabel('Frequency')
    plt.ylabel('Fit Error (%)')
    plt.title('SHE Solver: Fit Error vs Frequency')
    plt.xlim(f_sel[0], f_sel[-1])
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if condition_metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(f_sel, res_cond, 'g-', linewidth=1.5, alpha=0.8, label='Condition Number')
        plt.xscale('log')
        plt.yscale('log')
        
        ax2 = plt.gca()
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}kHz" if x >= 1000 else f"{int(x)}Hz"))
        ax2.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax2.format_xdata = lambda x: f"{x:.1f} Hz"
        ax2.format_ydata = lambda y: f"{y:.2e}"
        
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.xlabel('Frequency')
        plt.ylabel('Condition Number')
        plt.title('SHE Solver: Matrix Condition Number vs Frequency')
        plt.xlim(f_sel[0], f_sel[-1])
        plt.legend(loc='upper right')
        plt.tight_layout()
        
    plt.show()

# =============================================================================
# FDW Viewer (Stage 1)
# =============================================================================

class FDWViewer:
    def __init__(self, freqs, data_dict, meta_dict, fs, ir_dir, crop_samples, data_dict_smooth=None,
                 fdw_f_min=20.0, fdw_rft_ms=5.0):
        self.freqs = freqs
        self.data_dict = data_dict
        self.data_dict_smooth = data_dict_smooth
        self.meta_dict = meta_dict
        self.fs = fs
        self.ir_dir = ir_dir
        self.crop_samples = crop_samples
        self.filenames = sorted(list(data_dict.keys()), key=natural_keys)
        self.index = 0
        self.fdw_f_min = fdw_f_min
        self.fdw_rft_ms = fdw_rft_ms
        
        self.fig = plt.figure(figsize=(14, 12))
        self.gs = self.fig.add_gridspec(3, 1, hspace=0.35, height_ratios=[3, 2, 1])
        
        self.ax1 = self.fig.add_subplot(self.gs[0])
        self.ax2 = self.fig.add_subplot(self.gs[1]) 
        self.ax3 = self.fig.add_subplot(self.gs[2], sharex=self.ax1)
        
        plt.subplots_adjust(bottom=0.15, right=0.85)
        
        ax_prev = plt.axes([0.15, 0.03, 0.1, 0.05])
        ax_next = plt.axes([0.26, 0.03, 0.1, 0.05])
        ax_input = plt.axes([0.45, 0.03, 0.15, 0.05])
        ax_exit = plt.axes([0.75, 0.03, 0.1, 0.05])
        
        self.b_prev = Button(ax_prev, 'Previous')
        self.b_next = Button(ax_next, 'Next')
        self.b_exit = Button(ax_exit, 'Exit Plot')
        self.text_box = TextBox(ax_input, 'Jump to ID: ', initial="0")
        
        self.b_prev.on_clicked(self.prev_plot)
        self.b_next.on_clicked(self.next_plot)
        self.text_box.on_submit(self.submit_id)
        self.b_exit.on_clicked(lambda e: plt.close())
        
        self.update_plot()
        plt.show()

    def update_plot(self):
        from stage1_fdwsmooth import load_and_prep_ir
        
        fname = self.filenames[self.index]
        H_raw, m = self.data_dict[fname], self.meta_dict[fname]
        
        self.text_box.eventson = False
        self.text_box.set_val(str(self.index))
        self.text_box.eventson = True
        
        # Silence the harmless log-scale warning during clear
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.ax1.clear()
        
        mag_db_raw = 20 * np.log10(np.abs(H_raw) + 1e-12)
        peak_mag = np.max(mag_db_raw)
        
        if self.data_dict_smooth and fname in self.data_dict_smooth:
            H_smooth = self.data_dict_smooth[fname]
            mag_db_smooth = 20 * np.log10(np.abs(H_smooth) + 1e-12)
            
            self.ax1.semilogx(self.freqs, mag_db_raw - peak_mag, color='silver', lw=1.0, alpha=0.6, label="FDW Raw")
            self.ax1.semilogx(self.freqs, mag_db_smooth - peak_mag, 'r', lw=1.5, zorder=15, label="FDW Smoothed")
            self.ax1.legend(loc="upper right", fontsize=9)
        else:
            self.ax1.semilogx(self.freqs, mag_db_raw - peak_mag, 'k', lw=2, zorder=10, label="FDW Raw")
        
        f_trans = m['f_trans']
        f_anchor = m.get('f_lf_anchor', 200.0)
        
        self.ax1.axvspan(f_trans, self.fs/2, color='#E0F7FA', alpha=0.5)
        self.ax1.text(np.sqrt(f_trans*(self.fs/2)), -35, f"Fixed Window\n({self.fdw_rft_ms} ms)", 
                     color='#006064', fontsize=9, fontweight='bold', va='center', ha='center', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=20)
        
        cmap = plt.get_cmap('YlOrRd')
        num_octaves = np.log2(f_trans / self.fdw_f_min) if f_trans > self.fdw_f_min else 1
        
        self.ax2.clear()
        file_path = os.path.join(self.ir_dir, fname)
        try:
            # Only load up to the crop limit, not the entire file
            wav_data, _ = load_and_prep_ir(file_path, crop_samples=self.crop_samples)
            
            # Zero pad if necessary to match FFT length exactly
            if len(wav_data) < self.crop_samples:
                wav_data = np.pad(wav_data, (0, self.crop_samples - len(wav_data)), mode='constant')
                
            t_ms = np.linspace(0, len(wav_data)/self.fs * 1000, len(wav_data))
            
            self.ax2.plot(t_ms, wav_data, color='#333333', lw=1, zorder=1, label="Raw IR")
            
            local_max = np.max(np.abs(wav_data))
            if local_max > 0: self.ax2.set_ylim(-local_max/0.9, local_max/0.9)
            
            peak_time_ms = m['t_peak'] * 1000
            self.ax2.axvline(peak_time_ms, color='red', linestyle='-', linewidth=1.5, label='Detected Peak')
        except Exception as e:
            self.ax2.text(0.5, 0.5, f"Error loading wav: {e}", ha='center', va='center')
            t_ms = np.array([])
            local_max = 0
            peak_time_ms = 0

        valid_indices = [i for i, fc in enumerate(m['f_centers']) if self.fdw_f_min <= fc <= f_trans]
        valid_centers = m['f_centers'][valid_indices]
        
        boundaries = [f_trans]
        if len(valid_centers) > 0 and abs(valid_centers[0] - f_trans) > 1.0:
            boundaries.append(valid_centers[0])
            
        if len(valid_centers) > 0:
            boundaries.extend(valid_centers[1:])
        boundaries.append(self.fdw_f_min)
        
        for k in range(len(boundaries) - 1):
            high_bound = boundaries[k]
            low_bound = boundaries[k+1]
            f_mid = np.sqrt(high_bound * low_bound)
            norm_pos = np.log2(f_trans / f_mid) / num_octaves
            c_val = 0.1 + (0.5 * np.clip(norm_pos, 0, 1))
            self.ax1.axvspan(low_bound, high_bound, color=cmap(c_val), alpha=0.4)

        valid_band_idx = 0 
        
        for i, fc in enumerate(m['f_centers']):
            if fc < self.fdw_f_min: continue
            if fc > f_trans: continue
            
            norm_pos = np.log2(f_trans / fc) / num_octaves
            line_color = cmap(0.1 + (0.5 * np.clip(norm_pos, 0, 1)))
            
            self.ax1.axvline(fc, color=line_color, ls='-', alpha=0.8, linewidth=1.2)
            
            t_win_ms = m['t_windows'][i] * 1000
            label_y = -35 + (valid_band_idx % 2) * 4
            self.ax1.text(fc, label_y, f"{fc:.0f}Hz\n{t_win_ms:.1f}ms", 
                         ha='center', fontsize=8, 
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'), zorder=20)
            
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
                    t_curve = np.array([t_end_abs, t_end_abs])
                    y_curve = np.array([local_max, 0])

                t_full = np.concatenate((t_flat, t_curve))
                y_full = np.concatenate((y_flat, y_curve))
                valid = t_full <= t_ms[-1]
                
                lbl = f"{fc:.0f}Hz: {win_len_ms:.1f}ms"
                self.ax2.plot(t_full[valid], y_full[valid], color=line_color, lw=2, alpha=0.8, zorder=10, label=lbl)
            
            valid_band_idx += 1

        self.ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9)
        self.ax2.set_ylabel("Amplitude")
        self.ax2.set_xlabel("Time (ms)")
        self.ax2.set_title("Impulse Response Waveform & FDW Envelopes")
        self.ax2.grid(True, alpha=0.3)

        self.ax3.clear()
        
        f_raw = m['f_centers']
        a_raw = m['alpha_sched']
        
        sort_idx = np.argsort(f_raw)
        f_plot = f_raw[sort_idx]
        a_plot = a_raw[sort_idx]
        
        if f_plot[0] > self.fdw_f_min:
            f_plot = np.insert(f_plot, 0, self.fdw_f_min)
            a_plot = np.insert(a_plot, 0, a_plot[0]) 
            
        self.ax3.semilogx(f_plot, a_plot, 'b-', lw=3)
        
        a_min, a_max = np.min(a_plot), np.max(a_plot)
        y_range = max(1e-3, a_max - a_min)
        y_pos_hf = a_max - (y_range * 0.05)
        y_pos_lf = a_min + (y_range * 0.05)

        self.ax3.axvline(f_trans, color='green', linestyle='--', alpha=0.7)
        self.ax3.text(f_trans, y_pos_hf, f" Alpha HF: {f_trans:.0f} Hz", color='green', 
                      fontsize=8, ha='left', va='top')

        self.ax3.axvline(f_anchor, color='green', linestyle='--', alpha=0.7)
        self.ax3.text(f_anchor, y_pos_lf, f"Alpha LF: {f_anchor:.0f} Hz ", color='green', 
                      fontsize=8, ha='right', va='bottom')

        self.ax3.set_ylabel("Alpha")
        self.ax3.set_xlim(self.fdw_f_min, self.fs/2)
        self.ax3.grid(True, which='both', alpha=0.3)
        self.ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())

        self.ax1.set_title(f"File [{self.index}]: {fname}\nFDW Magnitude")
        self.ax1.set_ylabel("Magnitude (dB)")
        self.ax1.set_xlim(self.fdw_f_min, self.fs/2)
        self.ax1.grid(True, which='both', alpha=0.3)
        self.ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
        
        self.fig.canvas.draw_idle()

    def next_plot(self, e): self.index = (self.index + 1) % len(self.filenames); self.update_plot()
    def prev_plot(self, e): self.index = (self.index - 1) % len(self.filenames); self.update_plot()
    def submit_id(self, t): 
        try: val = int(t); self.index = val if 0 <= val < len(self.filenames) else self.index; self.update_plot()
        except: pass


# =============================================================================
# Origin Search Viewers (Stage 2)
# =============================================================================

class FrequencyBrowser3D:
    def __init__(self, sweep_data, cfg, show_immediately=True):
        self.data = sweep_data
        self.freqs = sorted(list(sweep_data.keys()))
        self.index = 0
        self.cfg = cfg
        
        self.fig = plt.figure(figsize=(12, 9))
        self.fig.canvas.manager.set_window_title('Interactive 3D Vector Search Cache')
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.1, bottom=0.3, right=0.85) 
        self.cbar_ax = self.fig.add_axes([0.88, 0.3, 0.03, 0.6]) 
        
        self.state = {
            'cbar': None,
            'plane': 'YZ (X Depth)',
            'slice_val': 0.0,
        }
        
        # Static Axis Properties
        self.ax.set_xlim(self.cfg['x_bounds'])
        self.ax.set_ylim(self.cfg['y_bounds'])
        self.ax.set_zlim(self.cfg['z_bounds'])
        self.ax.set_box_aspect((self.cfg['x_bounds'][1]-self.cfg['x_bounds'][0], self.cfg['y_bounds'][1]-self.cfg['y_bounds'][0], self.cfg['z_bounds'][1]-self.cfg['z_bounds'][0]))
        self.ax.set_xlabel('X (Depth) mm')
        self.ax.set_ylabel('Y (Width) mm')
        self.ax.set_zlabel('Z (Height) mm')

        # Set an initial turntable view and connect elevation clamp
        self.ax.view_init(elev=30, azim=-45)
        self.fig.canvas.mpl_connect('motion_notify_event', self._enforce_turntable)

        # GUI Widgets (Layout adapted from reference script)
        axcolor = 'lightgoldenrodyellow'
        rax = plt.axes([0.05, 0.05, 0.25, 0.12], facecolor=axcolor)
        self.radio = RadioButtons(rax, ('YZ (X Depth)', 'XZ (Y Width)', 'XY (Z Height)'))

        ax_slider = plt.axes([0.4, 0.1, 0.4, 0.03], facecolor=axcolor)
        
        d = self.data[self.freqs[0]]
        self.x_arr = d['X_vals'] if d['X_vals'] is not None else np.array([self.cfg['x_bounds'][0], self.cfg['x_bounds'][1]])
        self.y_arr = d['Y_vals'] if d['Y_vals'] is not None else np.array([self.cfg['y_bounds'][0], self.cfg['y_bounds'][1]])
        self.z_arr = d['Z_vals'] if d['Z_vals'] is not None else np.array([self.cfg['z_bounds'][0], self.cfg['z_bounds'][1]])
        
        self.depth_slider = Slider(ax_slider, 'Slice Pos', self.x_arr[0], self.x_arr[-1], valinit=0.0, valstep=self.cfg['grid_res_mm'])

        axprev = plt.axes([0.55, 0.02, 0.1, 0.05])
        axnext = plt.axes([0.66, 0.02, 0.1, 0.05])
        self.btn_prev = Button(axprev, 'Prev (Hz)')
        self.btn_next = Button(axnext, 'Next (Hz)')
        
        self.depth_slider.on_changed(self.update_slider)
        self.radio.on_clicked(self.update_radio)
        self.btn_prev.on_clicked(self.prev)
        self.btn_next.on_clicked(self.next)

        self.update_plot()
        if show_immediately:
            plt.show()

    def _enforce_turntable(self, event):
        if hasattr(self.ax, 'elev'):
            elev = self.ax.elev
            azim = self.ax.azim
            
            needs_update = False
            new_elev = elev
            
            if elev > 89.9:
                new_elev = 89.9
                needs_update = True
            elif elev < -89.9:
                new_elev = -89.9
                needs_update = True
                
            has_roll = hasattr(self.ax, 'roll')
            if has_roll and abs(self.ax.roll) > 1e-3:
                needs_update = True
                
            if needs_update:
                if has_roll:
                    self.ax.view_init(elev=new_elev, azim=azim, roll=0)
                else:
                    self.ax.view_init(elev=new_elev, azim=azim)

    def update_plot(self):
        while self.ax.collections:
            self.ax.collections[0].remove()
        for line in self.ax.lines:
            line.remove()
            
        f = self.freqs[self.index]
        d = self.data[f]
        grid_3d, xv, yv, zv = d['grid'], d['X_vals'], d['Y_vals'], d['Z_vals']
        
        plane = self.state['plane']
        val = self.state['slice_val']
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
            
            if self.state['cbar'] is None:
                self.state['cbar'] = self.fig.colorbar(sm, cax=self.cbar_ax)
            else:
                self.state['cbar'].update_normal(sm)
            self.state['cbar'].set_label('Residual Error (%)')
            grid_min_text = f"{np.min(grid_3d):.2f}%"
        else:
            grid_min_text = "N/A (Bypassed)"
            if self.state['cbar'] is not None:
                self.state['cbar'].remove()
                self.state['cbar'] = None

        # Plot the Key Markers
        if d['final_c'] is not None:
            fc = d['final_c']
            self.ax.scatter(fc[0], fc[1], fc[2], c='yellow', marker='D', s=100, edgecolors='black', label='Simplex Final')

        self.ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=9)
        self.ax.set_title(f"Frequency: {f:.1f} Hz | Grid Scan Minima: {grid_min_text}\n"
                          f"Search Final: ({d['final_c'][0]:.1f}, {d['final_c'][1]:.1f}, {d['final_c'][2]:.1f})\n"
                          f"Plane: {plane} @ {actual_val:.1f} mm", fontweight='bold')

        self.fig.canvas.draw_idle()

    def update_slider(self, val):
        self.state['slice_val'] = self.depth_slider.val
        self.update_plot()

    def update_radio(self, label):
        self.state['plane'] = label
        if label == 'YZ (X Depth)':
            self.depth_slider.valmin, self.depth_slider.valmax = self.x_arr[0], self.x_arr[-1]
        elif label == 'XZ (Y Width)':
            self.depth_slider.valmin, self.depth_slider.valmax = self.y_arr[0], self.y_arr[-1]
        else:
            self.depth_slider.valmin, self.depth_slider.valmax = self.z_arr[0], self.z_arr[-1]
        
        new_val = np.clip(self.state['slice_val'], self.depth_slider.valmin, self.depth_slider.valmax)
        self.depth_slider.set_val(new_val)
        self.depth_slider.ax.set_xlim(self.depth_slider.valmin, self.depth_slider.valmax)
        self.update_plot()

    def next(self, event):
        self.index = (self.index + 1) % len(self.freqs)
        self.update_plot()

    def prev(self, event):
        self.index = (self.index - 1) % len(self.freqs)
        self.update_plot()

class CloudBrowser3D:
    def __init__(self, sweep_data, cfg, show_immediately=True):
        self.data = sweep_data
        self.freqs = sorted([f for f, d in sweep_data.items() if d['final_c'] is not None])
        self.index = 0
        self.cfg = cfg
        
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('3D Coordinate Cloud Viewer')
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.2)
        
        # Static Axis Properties
        self.ax.set_xlim(self.cfg['x_bounds'])
        self.ax.set_ylim(self.cfg['y_bounds'])
        self.ax.set_zlim(self.cfg['z_bounds'])
        self.ax.set_box_aspect((self.cfg['x_bounds'][1]-self.cfg['x_bounds'][0], self.cfg['y_bounds'][1]-self.cfg['y_bounds'][0], self.cfg['z_bounds'][1]-self.cfg['z_bounds'][0]))
        self.ax.set_xlabel('X (Depth) mm')
        self.ax.set_ylabel('Y (Width) mm')
        self.ax.set_zlabel('Z (Height) mm')

        # Set an initial turntable view and connect elevation clamp
        self.ax.view_init(elev=30, azim=-45)
        self.fig.canvas.mpl_connect('motion_notify_event', self._enforce_turntable)

        axcolor = 'lightgoldenrodyellow'
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor=axcolor)
        
        self.freq_slider = Slider(ax_slider, 'Frequency', 0, max(0, len(self.freqs)-1), valinit=0, valstep=1, valfmt='%0.0f')
        if len(self.freqs) > 0:
            self.freq_slider.valtext.set_text(f"{self.freqs[0]:.1f} Hz")
            
        self.freq_slider.on_changed(self.update_slider)
        
        self.scatter_all = None
        self.scatter_hi = None
        
        self.refresh_data()
        
        if show_immediately:
            plt.show()

    def _enforce_turntable(self, event):
        if hasattr(self.ax, 'elev'):
            if abs(self.ax.elev - 30.0) > 1e-3:
                self.ax.view_init(elev=30.0, azim=self.ax.azim)

    def update_slider(self, val):
        self.index = int(self.freq_slider.val)
        if len(self.freqs) > 0:
            self.freq_slider.valtext.set_text(f"{self.freqs[self.index]:.1f} Hz")
        self.update_plot()

    def refresh_data(self):
        self.pts_x = [self.data[f]['final_c'][0] for f in self.freqs]
        self.pts_y = [self.data[f]['final_c'][1] for f in self.freqs]
        self.pts_z = [self.data[f]['final_c'][2] for f in self.freqs]
        
        if self.scatter_all is not None: self.scatter_all.remove()
        if self.scatter_hi is not None: self.scatter_hi.remove()
            
        if len(self.freqs) > 0:
            self.scatter_all = self.ax.scatter(self.pts_x, self.pts_y, self.pts_z, c='blue', s=30, alpha=0.6, label='Nelder Final')
            pt = self.data[self.freqs[self.index]]['final_c']
            self.scatter_hi = self.ax.scatter([pt[0]], [pt[1]], [pt[2]], c='red', s=120, alpha=1.0, label='Highlighted')
            self.ax.legend(loc='upper right')
            
        self.update_plot()

    def update_plot(self):
        if len(self.freqs) == 0: return
        idx = self.index
        f = self.freqs[idx]
        pt = self.data[f]['final_c']
        
        if self.scatter_hi is not None:
            self.scatter_hi._offsets3d = ([pt[0]], [pt[1]], [pt[2]])
            
        self.ax.set_title(f"Frequency: {f:.1f} Hz\nCoordinate: ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})", fontweight='bold')
        self.fig.canvas.draw_idle()


class ValidationUI:
    def __init__(self, sweep_results, f_all, keys, d_dict, parsed_geom, cfg):
        self.sweep_results = sweep_results
        self.f_all = f_all
        self.keys = keys
        self.d_dict = d_dict
        self.r_arr, self.th_arr, self.ph_arr = parsed_geom
        self.cfg = cfg
        
        for f, d in self.sweep_results.items():
            if 'original_c' not in d and d['final_c'] is not None:
                d['original_c'] = np.copy(d['final_c'])
        
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()
        
        self.fig_sum, (self.ax_x, self.ax_y, self.ax_z) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        self.fig_sum.canvas.manager.set_window_title('Search Validation Summary with Error Labels')
        
        plt.subplots_adjust(bottom=0.2)
        
        ax_edit = plt.axes([0.1, 0.05, 0.2, 0.075])
        self.btn_edit = Button(ax_edit, 'Edit Coordinates')
        self.btn_edit.on_clicked(self.open_edit_dialog)
        
        ax_rescan = plt.axes([0.35, 0.05, 0.2, 0.075])
        self.btn_rescan = Button(ax_rescan, 'Rescan Frequencies')
        self.btn_rescan.on_clicked(self.open_rescan_dialog)
        
        ax_accept = plt.axes([0.7, 0.05, 0.2, 0.075])
        self.btn_accept = Button(ax_accept, 'Accept & Save')
        self.btn_accept.on_clicked(self.accept)
        
        self.fig_sum.canvas.mpl_connect('close_event', self.on_close)
        
        self.accepted = False
        self.update_plot()
        
        if self.cfg['enable_full_grid_scan']:
            self.browser3d = FrequencyBrowser3D(self.sweep_results, self.cfg, show_immediately=False)
        else:
            self.browser3d = CloudBrowser3D(self.sweep_results, self.cfg, show_immediately=False)
            
        plt.show()

    def update_plot(self):
        from stage2_centre_origin import get_order_for_frequency, solve_physics_3d
        
        for ax in (self.ax_x, self.ax_y, self.ax_z):
            ax.clear()
            
        history_freq = []
        history_search_x, history_search_y, history_search_z = [], [], []
        history_orig_x, history_orig_y, history_orig_z = [], [], []
        history_err_search, history_err_orig = [], []

        for f_hz in sorted(self.sweep_results.keys()):
            d = self.sweep_results[f_hz]
            if d['final_c'] is not None:
                history_freq.append(f_hz)
                history_search_x.append(d['final_c'][0])
                history_search_y.append(d['final_c'][1])
                history_search_z.append(d['final_c'][2])
                
                orig_c = d.get('original_c', d['final_c'])
                history_orig_x.append(orig_c[0])
                history_orig_y.append(orig_c[1])
                history_orig_z.append(orig_c[2])
                
                current_f_idx = (np.abs(self.f_all - f_hz)).argmin()
                P_list = [self.d_dict[k][current_f_idx] for k in self.keys]
                
                current_order_N = get_order_for_frequency(f_hz)
                
                phys_context = (
                    [(float(f_hz), float(2 * np.pi * f_hz / self.cfg['speed_of_sound']), np.conj(np.array(P_list)))], 
                    self.r_arr, self.th_arr, self.ph_arr, current_order_N, self.cfg
                )
                
                e_search = solve_physics_3d(d['final_c'][0], d['final_c'][1], d['final_c'][2], phys_context)
                
                if not np.allclose(orig_c, d['final_c']):
                    e_orig = solve_physics_3d(orig_c[0], orig_c[1], orig_c[2], phys_context)
                else:
                    e_orig = e_search
                    
                history_err_search.append(e_search)
                history_err_orig.append(e_orig)
                
        axes = [self.ax_x, self.ax_y, self.ax_z]
        coords_search = [history_search_x, history_search_y, history_search_z]
        coords_orig = [history_orig_x, history_orig_y, history_orig_z]
        labels = ['X (Depth)', 'Y (Width)', 'Z (Height)']

        for i, sax in enumerate(axes):
            sax.format_xdata = lambda x: f"{x:.1f} Hz"
            sax.format_ydata = lambda y: f"{y:.2f} mm"

            # Base path using current coords (blue line)
            sax.plot(history_freq, coords_search[i], 'b-', alpha=0.5, label='Search Path' if i == 0 else "")
            
            mod_idx = [j for j in range(len(history_freq)) if not np.allclose(
                [coords_orig[0][j], coords_orig[1][j], coords_orig[2][j]], 
                [coords_search[0][j], coords_search[1][j], coords_search[2][j]]
            )]
            
            # Plot all current points
            unmod_idx = [j for j in range(len(history_freq)) if j not in mod_idx]
            if unmod_idx:
                sax.plot([history_freq[j] for j in unmod_idx], [coords_search[i][j] for j in unmod_idx], 
                         'bo', alpha=0.6, label='Original Result' if i == 0 else "")
            if mod_idx:
                sax.plot([history_freq[j] for j in mod_idx], [coords_orig[i][j] for j in mod_idx], 
                         'bx', alpha=0.4, label='Previous Result' if i == 0 else "")
                sax.plot([history_freq[j] for j in mod_idx], [coords_search[i][j] for j in mod_idx], 
                         'ro', alpha=0.9, label='New Optimal Result' if i == 0 else "")
                         
            for j, f in enumerate(history_freq):
                if j in mod_idx:
                    # Connecting line to show movement
                    sax.plot([f, f], [coords_orig[i][j], coords_search[i][j]], 'r--', alpha=0.5)
                    sax.text(f, coords_orig[i][j], f"{history_err_orig[j]:.2f}%", 
                             color='blue', fontsize=8, ha='left', va='top', alpha=0.5)
                    sax.text(f, coords_search[i][j], f"{history_err_search[j]:.2f}%", 
                             color='red', fontsize=8, ha='right', va='bottom', alpha=0.9)
                else:
                    sax.text(f, coords_search[i][j], f"{history_err_search[j]:.2f}%", 
                             color='blue', fontsize=8, ha='right', va='bottom', alpha=0.8)

            sax.set_xscale('log')
            sax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}kHz" if x >= 1000 else f"{int(x)}Hz"))
            sax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
            sax.set_ylabel(f'{labels[i]} (mm)')
            sax.grid(True, which='both', ls='--', alpha=0.3)
            if i == 0:
                sax.set_title('Frequency Sweep Validation: Coordinates & Residual Error Labels', fontweight='bold')
                sax.legend(loc='upper right')

        self.ax_z.set_xlabel('Frequency (Hz)')
        self.fig_sum.canvas.draw_idle()

    def open_edit_dialog(self, event):
        top = tk.Toplevel(self.tk_root)
        top.title("Edit Coordinates")
        top.geometry("400x300")
        
        tree = ttk.Treeview(top, columns=('Freq', 'X', 'Y', 'Z'), show='headings')
        tree.heading('Freq', text='Freq (Hz)')
        tree.heading('X', text='X')
        tree.heading('Y', text='Y')
        tree.heading('Z', text='Z')
        
        for f in sorted(self.sweep_results.keys()):
            d = self.sweep_results[f]
            if d['final_c'] is not None:
                tree.insert('', tk.END, values=(f"{f:.1f}", f"{d['final_c'][0]:.2f}", f"{d['final_c'][1]:.2f}", f"{d['final_c'][2]:.2f}"))
                
        tree.pack(fill=tk.BOTH, expand=True)
        
        frame = tk.Frame(top)
        frame.pack(pady=5)
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
            try:
                x = float(ent_x.get())
                y = float(ent_y.get())
                z = float(ent_z.get())
            except ValueError:
                return
            vals = list(tree.item(sel[0], 'values'))
            vals[1] = f"{x:.2f}"
            vals[2] = f"{y:.2f}"
            vals[3] = f"{z:.2f}"
            tree.item(sel[0], values=vals)
            f = float(vals[0])
            exact_f = min(self.sweep_results.keys(), key=lambda k: abs(k - f))
            self.sweep_results[exact_f]['final_c'] = np.array([x, y, z])
            self.update_plot()
            if hasattr(self, 'browser3d') and self.browser3d is not None:
                if hasattr(self.browser3d, 'refresh_data'):
                    self.browser3d.refresh_data()
                else:
                    self.browser3d.update_plot()
            
        tk.Button(frame, text="Apply", command=apply_edit).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Close", command=top.destroy).pack(pady=5)

    def open_rescan_dialog(self, event):
        from stage2_centre_origin import get_order_for_frequency, generate_3d_landscape_volumetric, run_simplex_descent_3d, solve_physics_3d
        
        top = tk.Toplevel(self.tk_root)
        top.title("Rescan Frequencies")
        top.geometry("450x400")
        
        frame = tk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        vars = {}
        for i, f in enumerate(sorted(self.sweep_results.keys())):
            var = tk.BooleanVar()
            vars[f] = var
            row, col = divmod(i, 4)
            cb = tk.Checkbutton(scrollable_frame, text=f"{f:.1f} Hz", variable=var)
            cb.grid(row=row, column=col, sticky='w', padx=5, pady=2)
            
        def do_rescan():
            freqs_to_rescan = [f for f, var in vars.items() if var.get()]
            
            top.destroy()
            self.tk_root.update()
            
            if not freqs_to_rescan:
                return
                
            for f in freqs_to_rescan:
                print(f"\nRescanning {f:.1f} Hz...")
                current_f_idx = (np.abs(self.f_all - f)).argmin()
                P_list = [self.d_dict[k][current_f_idx] for k in self.keys]
                current_order_N = get_order_for_frequency(f)
                phys_context = (
                    [(float(f), float(2*np.pi*f/self.cfg['speed_of_sound']), np.conj(np.array(P_list)))],
                    self.r_arr, self.th_arr, self.ph_arr, current_order_N, self.cfg
                )
                
                print(f"   -> Running Volumetric Grid Scan...")
                x_vals, y_vals, z_vals, grid = generate_3d_landscape_volumetric(phys_context)
                min_idx = np.unravel_index(np.argmin(grid), grid.shape)
                true_coords = (x_vals[min_idx[0]], y_vals[min_idx[1]], z_vals[min_idx[2]])
                val_true = np.min(grid)
                
                print(f"   -> 3D Amoeba descending (Grid-Seeded)...")
                opt_c, path = run_simplex_descent_3d(true_coords, phys_context)
                
                val_opt = solve_physics_3d(opt_c[0], opt_c[1], opt_c[2], phys_context)
                print(f"   -> New Grid Scan Minima : ({true_coords[0]:>6.1f}, {true_coords[1]:>6.1f}, {true_coords[2]:>6.1f}) | Error: {val_true:>5.2f}%")
                print(f"   -> New Simplex Final    : ({opt_c[0]:>6.1f}, {opt_c[1]:>6.1f}, {opt_c[2]:>6.1f}) | Error: {val_opt:>5.2f}%")
                
                self.sweep_results[f]['grid'] = grid
                self.sweep_results[f]['X_vals'] = x_vals
                self.sweep_results[f]['Y_vals'] = y_vals
                self.sweep_results[f]['Z_vals'] = z_vals
                self.sweep_results[f]['true'] = true_coords
                self.sweep_results[f]['final_c'] = opt_c
                self.sweep_results[f]['path'] = path
                
            print("\nRescan complete.")
            self.update_plot()
            if hasattr(self, 'browser3d') and self.browser3d is not None:
                if hasattr(self.browser3d, 'refresh_data'):
                    self.browser3d.refresh_data()
                else:
                    self.browser3d.update_plot()
            
            if not self.cfg['enable_full_grid_scan']:
                # Automatically open the volumetric viewer for just the newly scanned grids
                rescanned_dict = {f: self.sweep_results[f] for f in freqs_to_rescan}
                self.rescan_browser = FrequencyBrowser3D(rescanned_dict, self.cfg, show_immediately=False)
                self.rescan_browser.fig.show()
            
        tk.Button(top, text="Run Full Grid Scan", command=do_rescan).pack(pady=5)

    def accept(self, event):
        self.accepted = True
        self.tk_root.destroy()
        plt.close(self.fig_sum)
        if hasattr(self, 'browser3d') and self.browser3d is not None and hasattr(self.browser3d, 'fig'):
            plt.close(self.browser3d.fig)
        if hasattr(self, 'rescan_browser') and self.rescan_browser is not None and hasattr(self.rescan_browser, 'fig'):
            plt.close(self.rescan_browser.fig)

    def on_close(self, event):
        try:
            self.tk_root.destroy()
        except:
            pass
        if hasattr(self, 'browser3d') and self.browser3d is not None and hasattr(self.browser3d, 'fig'):
            plt.close(self.browser3d.fig)
        if hasattr(self, 'rescan_browser') and self.rescan_browser is not None and hasattr(self.rescan_browser, 'fig'):
            plt.close(self.rescan_browser.fig)