#!/usr/bin/env python3
"""
Interactive 3D Replay Viewer for Coordinate Grids - Stage 5 Updated
-------------------------------------------------
Features:
- Bottom-aligned coordinate readout (4 lines max)
- Optimized 3D plot area (wider view)
- Pure GUI-agnostic rendering engine with Tkinter wrapper
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Helper function ---
def _get_xyz_from_wavs(wav_dir):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from utils import parse_coords_from_filename, spherical_to_cartesian
    except ImportError:
        raise ImportError("Could not import utils.py. Ensure it is located in the same or parent directory.")
        
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {wav_dir}")

    r_list, th_list, ph_list = [], [], []
    for f in wav_files:
        fname = os.path.basename(f)
        try:
            r_sph, theta, phi = parse_coords_from_filename(fname)
            r_list.append(r_sph)
            th_list.append(theta)
            ph_list.append(phi)
        except Exception:
            pass
            
    if not r_list:
        raise ValueError(f"No valid coordinate-encoded .wav files found in {wav_dir}")

    x, y, z = spherical_to_cartesian(np.array(r_list), np.array(th_list), np.array(ph_list))
    
    df = pd.DataFrame({
        'r_xy_mm': np.hypot(x, y) * 1000.0,
        'phi_deg': np.degrees(np.arctan2(y, x)),
        'z_mm': z * 1000.0
    })
    return df

# ── 1. The Pure GUI-Agnostic Engine ──────────────────────────────────────────
class CoordViewerEngine:
    """
    Pure rendering engine. Knows nothing about UI widgets.
    """
    def __init__(self, input_data=None):
        # Playback State
        self.curr_idx = 0
        self.exact_idx = 0.0
        self.is_playing = False
        self.ppm = 600.0
        self.tail_length = 50
        self.use_history_fading = False
        self.use_ortho = False
        self.timer_interval_ms = 50
        self.show_readout = True

        # Rotation Animation State
        self.is_rotating = False
        self.rot_full_angle = 45.0
        self.rot_target_angle = 22.5
        self.rot_dir = 1
        self.rot_accumulated = 0.0

        # Setup Figure - Bottom margin increased for the new readout position
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.subplots_adjust(top=1.0, bottom=0.10, left=0.0, right=1.0)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.view_init(elev=30, azim=135)

        # Custom Colormap
        self.custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            'black_hsv', 
            ['black'] + [plt.get_cmap('hsv')(i) for i in np.linspace(0, 1, 256)]
        )

        # Fast line plots
        self.base_pts, = self.ax.plot([], [], [], marker='o', linestyle='none', color=self.custom_cmap(0.0), markersize=2, alpha=0.2)
        self.line_hist, = self.ax.plot([], [], [], c='gray', alpha=0.3, linewidth=1.0)
        self.line_active, = self.ax.plot([], [], [], c='red', linewidth=2.0)
        self.head_pt, = self.ax.plot([], [], [], marker='o', c='blue', markersize=6)

        # Coordinate Readout Text - Positioned at bottom-left
        self.txt_readout = self.fig.text(
            0.02, 0.02, "",
            fontsize=9, family='monospace', verticalalignment='bottom',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        # Initialize default properties
        self.N = 0 
        self.x = self.y = self.z = self.phi_arr = np.array([]) 

        if input_data is not None:
            self.load_data(input_data)

        # Internal animation loops
        self.anim = animation.FuncAnimation(self.fig, self._on_frame, interval=self.timer_interval_ms, cache_frame_data=False)
        self.rot_anim = animation.FuncAnimation(self.fig, self._on_rotate_frame, interval=20, cache_frame_data=False)

    def load_data(self, input_data):
        if isinstance(input_data, str) and input_data.endswith('.csv'):
            self.df = pd.read_csv(input_data)
        elif isinstance(input_data, str):
            self.df = _get_xyz_from_wavs(input_data)
        elif isinstance(input_data, dict):
            self.df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            self.df = input_data.copy()
        else:
            raise TypeError("input_data must be a file path, dict, or DataFrame.")

        self.N = len(self.df)
        self.phi_arr = self.df["phi_deg"].to_numpy()
        self.r_m = self.df["r_xy_mm"].to_numpy() / 1000.0
        self.z_m = self.df["z_mm"].to_numpy() / 1000.0
        self.phi_rad = np.radians(self.phi_arr)
        
        self.x = -self.r_m * np.cos(self.phi_rad)
        self.y = self.r_m * np.sin(self.phi_rad)
        self.z = self.z_m

        self.base_pts.set_data(self.x, self.y)
        self.base_pts.set_3d_properties(self.z)

        if self.curr_idx >= self.N:
            self.curr_idx = max(0, self.N - 1)
            self.exact_idx = float(self.curr_idx)

        self._set_axes_equal()
        self.update_plot()

    def _set_axes_equal(self):
        if self.N == 0: return
        max_range = np.array([
            self.x.max() - self.x.min(),
            self.y.max() - self.y.min(),
            self.z.max() - self.z.min()
        ]).max() / 2.0
        mid_x = (self.x.max() + self.x.min()) * 0.5
        mid_y = (self.y.max() + self.y.min()) * 0.5
        mid_z = (self.z.max() + self.z.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- External Control API ---
    def set_current_index(self, idx):
        if self.is_playing: return
        idx = max(0, min(int(idx), self.N - 1))
        if idx != self.curr_idx:
            self.curr_idx = idx
            self.exact_idx = float(idx)
            self.update_plot()

    def play(self): self.is_playing = True; self.ax.set_title("Playing...")
    def pause(self): self.is_playing = False; self.ax.set_title("Paused")
    def rewind(self): self.pause(); self.curr_idx = 0; self.exact_idx = 0.0; self.update_plot()
    
    def step_fwd(self):
        self.pause()
        if self.curr_idx < self.N - 1: self.curr_idx += 1; self.update_plot()

    def step_back(self):
        self.pause()
        if self.curr_idx > 0: self.curr_idx -= 1; self.update_plot()

    def set_speed(self, ppm):
        if ppm > 0: self.ppm = ppm

    def set_tail_length(self, val):
        self.tail_length = int(val)
        if not self.is_playing: self.update_plot()

    def set_history_mode(self, enabled: bool):
        self.use_history_fading = enabled
        if not self.is_playing: self.update_plot()

    def set_ortho(self, enabled: bool):
        self.use_ortho = enabled
        self.ax.set_proj_type('ortho' if self.use_ortho else 'persp')
        self.fig.canvas.draw_idle()

    def set_view(self, elev, azim):
        self.ax.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def set_alpha(self, val):
        self.base_pts.set_alpha(val)
        self.fig.canvas.draw_idle()

    def set_color(self, val):
        self.base_pts.set_color(self.custom_cmap(val))
        self.fig.canvas.draw_idle()

    def toggle_readout(self, force_state=None):
        self.show_readout = not self.show_readout if force_state is None else force_state
        self.txt_readout.set_visible(self.show_readout)
        
        # Adjust bottom margin to make room for the readout box
        bottom_margin = 0.15 if self.show_readout else 0.05
        self.fig.subplots_adjust(bottom=bottom_margin)
        self.fig.canvas.draw_idle()

    def start_rotation(self, angle=45.0):
        self.is_rotating = True
        self.rot_full_angle = angle
        self.rot_target_angle = angle / 2.0
        self.rot_dir = 1
        self.rot_accumulated = 0.0

    def stop_rotation(self):
        self.is_rotating = False

    # --- Render Logic ---
    def update_plot(self):
        if self.N == 0: return

        i = self.curr_idx
        start_active = max(0, i - self.tail_length) if self.use_history_fading else 0

        self.head_pt.set_data([self.x[i]], [self.y[i]])
        self.head_pt.set_3d_properties([self.z[i]])
        self.line_active.set_data(self.x[start_active:i+1], self.y[start_active:i+1])
        self.line_active.set_3d_properties(self.z[start_active:i+1])
        
        if self.use_history_fading and start_active > 0:
            self.line_hist.set_data(self.x[0:start_active], self.y[0:start_active])
            self.line_hist.set_3d_properties(self.z[0:start_active])
        else:
            self.line_hist.set_data([], [])
            self.line_hist.set_3d_properties([])

        if self.show_readout:
            lines = []
            # Displaying 4 lines: 1 back, current, 2 forward
            for offset in range(-1, 3):
                idx = i + offset
                if 0 <= idx < self.N:
                    prefix = "->" if offset == 0 else "  "
                    lines.append(f"{prefix} {idx:4d} | X{self.x[idx]*1000:6.1f} Y{self.y[idx]*1000:6.1f} Z{self.z[idx]*1000:6.1f} A{self.phi_arr[idx]:6.1f}")
                else:
                    lines.append("        ---")
            self.txt_readout.set_text("\n".join(lines))

        self.fig.canvas.draw_idle()

    def _on_frame(self, frame):
        if self.is_playing and self.N > 0:
            step = (self.ppm / 60.0) * (self.timer_interval_ms / 1000.0)
            self.exact_idx += step
            new_idx = int(self.exact_idx)
            
            if new_idx > self.curr_idx:
                if new_idx >= self.N:
                    self.curr_idx = self.N - 1
                    self.pause()
                    self.ax.set_title("Replay Finished")
                else:
                    self.curr_idx = new_idx
                self.update_plot()
        return self.head_pt,

    def _on_rotate_frame(self, frame):
        if not self.is_rotating: return None
        step = 0.25
        self.ax.azim += (step * self.rot_dir)
        self.rot_accumulated += step
        if self.rot_accumulated >= self.rot_target_angle:
            self.rot_dir *= -1
            self.rot_accumulated = 0.0
            self.rot_target_angle = self.rot_full_angle
        self.fig.canvas.draw_idle()
        return None

# ── 2. The Native Tkinter Wrapper ────────────────────────────────────────────
class TkinterCoordApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Standalone 3D Replay Viewer - Stage 5")
        self.root.geometry("1100x850")
        
        self.engine = CoordViewerEngine()

        self._build_ui()
        self._pack_canvas()
        
        self.root.after(100, self._sync_ui_state)

    def _build_ui(self):
        ui_frame = tk.Frame(self.root, padx=10, pady=5)
        ui_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # -- Row 1: File & Scrubbing --
        row1 = tk.Frame(ui_frame)
        row1.pack(fill=tk.X, pady=2)
        tk.Button(row1, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="Gen Dummy", command=self.gen_dummy).pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="Toggle Coords", command=self.toggle_coords).pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1, text="Timeline:").pack(side=tk.LEFT, padx=5)
        self.scrub_slider = tk.Scale(row1, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_scrub)
        self.scrub_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.lbl_idx = tk.Label(row1, text="Point: 0 / 0", width=15)
        self.lbl_idx.pack(side=tk.LEFT, padx=5)

        # -- Row 2: VCR & Playback Config --
        row2 = tk.Frame(ui_frame)
        row2.pack(fill=tk.X, pady=2)
        tk.Button(row2, text="|<", command=self.engine.rewind, width=4).pack(side=tk.LEFT, padx=2)
        tk.Button(row2, text="<", command=self.engine.step_back, width=4).pack(side=tk.LEFT, padx=2)
        self.btn_play = tk.Button(row2, text="Play", command=self.toggle_play, width=6)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        tk.Button(row2, text=">", command=self.engine.step_fwd, width=4).pack(side=tk.LEFT, padx=2)

        tk.Label(row2, text="Rate (PPM):").pack(side=tk.LEFT, padx=(15,2))
        self.ent_speed = tk.Entry(row2, width=6)
        self.ent_speed.insert(0, str(int(self.engine.ppm)))
        self.ent_speed.bind("<Return>", lambda e: self.engine.set_speed(float(self.ent_speed.get())))
        self.ent_speed.pack(side=tk.LEFT)

        tk.Label(row2, text="Tail Len:").pack(side=tk.LEFT, padx=(15,2))
        self.scl_tail = tk.Scale(row2, from_=1, to=500, orient=tk.HORIZONTAL, command=lambda v: self.engine.set_tail_length(int(v)))
        self.scl_tail.set(self.engine.tail_length)
        self.scl_tail.pack(side=tk.LEFT)

        self.var_hist = tk.BooleanVar(value=False)
        tk.Checkbutton(row2, text="Fade Hist", variable=self.var_hist, command=lambda: self.engine.set_history_mode(self.var_hist.get())).pack(side=tk.LEFT, padx=15)

        # -- Row 3: Visual & Camera Settings --
        row3 = tk.Frame(ui_frame)
        row3.pack(fill=tk.X, pady=2)
        tk.Button(row3, text="Top", command=lambda: self.engine.set_view(90, 0)).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="Front", command=lambda: self.engine.set_view(0, 0)).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="Side", command=lambda: self.engine.set_view(0, -90)).pack(side=tk.LEFT, padx=2)

        tk.Label(row3, text="Color:").pack(side=tk.LEFT, padx=(15,2))
        self.scl_col = tk.Scale(row3, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, command=lambda v: self.engine.set_color(float(v)))
        self.scl_col.pack(side=tk.LEFT)

        tk.Label(row3, text="Opacity:").pack(side=tk.LEFT, padx=(15,2))
        self.scl_alpha = tk.Scale(row3, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, command=lambda v: self.engine.set_alpha(float(v)))
        self.scl_alpha.set(0.2)
        self.scl_alpha.pack(side=tk.LEFT)

        self.var_ortho = tk.BooleanVar(value=False)
        tk.Checkbutton(row3, text="Ortho", variable=self.var_ortho, command=lambda: self.engine.set_ortho(self.var_ortho.get())).pack(side=tk.LEFT, padx=15)

        tk.Label(row3, text="Rot Ang:").pack(side=tk.LEFT)
        self.ent_rot = tk.Entry(row3, width=5)
        self.ent_rot.insert(0, "45")
        self.ent_rot.pack(side=tk.LEFT)
        
        self.btn_rot = tk.Button(row3, text="Rotate", command=self.toggle_rotation)
        self.btn_rot.pack(side=tk.LEFT, padx=5)

    def _pack_canvas(self):
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.engine.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self.engine.load_data(path)
            self._update_slider_range()

    def gen_dummy(self):
        pts = np.random.randint(100, 400)
        t = np.linspace(0, 4 * np.pi, pts)
        df = pd.DataFrame({
            'r_xy_mm': np.abs(np.sin(t)) * 75,
            'phi_deg': np.degrees(t),
            'z_mm': np.linspace(0, 100, pts)
        })
        self.engine.load_data(df)
        self._update_slider_range()

    def _update_slider_range(self):
        max_idx = max(0, self.engine.N - 1)
        self.scrub_slider.config(to=max_idx)

    def on_scrub(self, val):
        self.engine.set_current_index(int(val))

    def toggle_play(self):
        if self.engine.is_playing:
            self.engine.pause()
            self.btn_play.config(text="Play")
        else:
            self.engine.play()
            self.btn_play.config(text="Pause")

    def toggle_coords(self):
        self.engine.toggle_readout()

    def toggle_rotation(self):
        if self.engine.is_rotating:
            self.engine.stop_rotation()
            self.btn_rot.config(text="Rotate")
        else:
            try: angle = float(self.ent_rot.get())
            except ValueError: angle = 45.0
            self.engine.start_rotation(angle)
            self.btn_rot.config(text="Stop Rot")

    def _sync_ui_state(self):
        max_idx = max(0, self.engine.N - 1)
        self.lbl_idx.config(text=f"Point: {self.engine.curr_idx} / {max_idx}")
        if self.engine.is_playing and self.scrub_slider.get() != self.engine.curr_idx:
            self.scrub_slider.set(self.engine.curr_idx)
        if not self.engine.is_playing and self.btn_play['text'] == "Pause":
            self.btn_play.config(text="Play")
        self.root.after(50, self._sync_ui_state)

if __name__ == "__main__":
    def on_closing():
        if hasattr(app.engine, 'anim') and app.engine.anim is not None:
            app.engine.anim.event_source.stop()
        if hasattr(app.engine, 'rot_anim') and app.engine.rot_anim is not None:
            app.engine.rot_anim.event_source.stop()
        root.quit()
        root.destroy()
        sys.exit(0)

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = TkinterCoordApp(root)
    app.gen_dummy()
    root.mainloop()