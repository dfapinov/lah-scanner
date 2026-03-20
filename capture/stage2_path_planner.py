#!/usr/bin/env python3
"""
Path Planning Method – θ-Binned Snake Strategy (-180 to +180)
=============================================================

Reads best_spiral_cylindrical.csv and produces an ordered path output.
Includes an interactive VCR-style replay tool with a scrolling coordinate readout.

Features:
    - Scrolling Coordinate List (Top Left)
    - Progress Slider
    - Speed Control
    - Rim-to-Rim Cap Transitions
    - -180 to +180 Degree Phi Range

Usage:
    python stage2_path_planner2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from config_capture import (
    INPUT_PATH_PLAN,
    OUTPUT_PATH_PLAN,
    DELTA_THETA_DEG,
    CAP_TOL_MM,
    SIDE_SNAKE_START,
    PRINT_SUMMARY
)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH_PLAN)
required = {"r_xy_mm", "phi_deg", "z_mm"}
if not required.issubset(df.columns):
    raise ValueError(f"Input must contain columns: {sorted(required)}")

# Internal units: meters for geometry, degrees for phi
r_xy_mm = df["r_xy_mm"].to_numpy(dtype=float)
# No modulo 360. Trust input is approx -180 to 180.
phi_d   = df["phi_deg"].to_numpy(dtype=float)
z_mm    = df["z_mm"].to_numpy(dtype=float)

r_xy = r_xy_mm / 1000.0
z    = z_mm    / 1000.0
N    = len(df)

# Cartesian for static background plotting
phi_rad = np.radians(phi_d)
x_all = r_xy * np.cos(phi_rad)
y_all = r_xy * np.sin(phi_rad)

# Cap detection
z_min, z_max = float(z.min()), float(z.max())
cap_tol = CAP_TOL_MM / 1000.0
on_top_cap    = np.isclose(z, z_max, atol=cap_tol)
on_bottom_cap = np.isclose(z, z_min, atol=cap_tol)
on_side       = ~(on_top_cap | on_bottom_cap)

# ── Binning Logic (-180 to 180) ──────────────────────────────────────────────
# Shift +180 to create 0-360 scale for binning only
bin_width = float(DELTA_THETA_DEG)
num_bins  = int(np.ceil(360.0 / bin_width))

phi_shifted = phi_d + 180.0
theta_bins = np.floor(phi_shifted / bin_width).astype(int)
theta_bins = np.clip(theta_bins, 0, num_bins - 1)

# ── Sorting Helpers ──────────────────────────────────────────────────────────

def get_binned_indices(mask, theta_bins_arr, num_bins, reverse_bins=False):
    """
    Returns a list of index-arrays, one per bin.
    """
    bin_groups = []
    r = range(num_bins - 1, -1, -1) if reverse_bins else range(num_bins)
    for b in r:
        idxs = np.where((theta_bins_arr == b) & mask)[0]
        if idxs.size > 0:
            bin_groups.append(idxs)
    return bin_groups

def sort_bin_by_radius(indices, r_arr, high_to_low=True):
    """Sorts a set of indices by radius."""
    sorted_idxs = indices[np.argsort(r_arr[indices])]
    if high_to_low:
        return sorted_idxs[::-1]
    return sorted_idxs

# ── Build Order ──────────────────────────────────────────────────────────────
order_indices = []
snake_up = (SIDE_SNAKE_START.lower() == "up")

# 1. Side Walls (Low Phi -> High Phi | -180 -> +180)
side_bins = get_binned_indices(on_side, theta_bins, num_bins, reverse_bins=False)

for idxs in side_bins:
    z_sorted = idxs[np.argsort(z[idxs])]
    if not snake_up:
        z_sorted = z_sorted[::-1]
    order_indices.extend(z_sorted.tolist())
    snake_up = not snake_up

# 2. Top Cap (High Phi -> Low Phi | +180 -> -180)
top_bins = get_binned_indices(on_top_cap, theta_bins, num_bins, reverse_bins=True)

if top_bins:
    previous_end_was_high_r = True 
    for i, idxs in enumerate(top_bins):
        is_first = (i == 0)
        is_last  = (i == len(top_bins) - 1)
        if is_first:
            sorted_idxs = sort_bin_by_radius(idxs, r_xy, high_to_low=True)
            previous_end_was_high_r = False 
        elif is_last:
            sorted_idxs = sort_bin_by_radius(idxs, r_xy, high_to_low=False)
            previous_end_was_high_r = True
        else:
            start_high = previous_end_was_high_r
            sorted_idxs = sort_bin_by_radius(idxs, r_xy, high_to_low=start_high)
            previous_end_was_high_r = not start_high 
        order_indices.extend(sorted_idxs.tolist())

# 3. Bottom Cap (Low Phi -> High Phi | -180 -> +180)
bot_bins = get_binned_indices(on_bottom_cap, theta_bins, num_bins, reverse_bins=False)

if bot_bins:
    previous_end_was_high_r = True
    for i, idxs in enumerate(bot_bins):
        is_first = (i == 0)
        is_last  = (i == len(bot_bins) - 1)
        if is_first:
            sorted_idxs = sort_bin_by_radius(idxs, r_xy, high_to_low=True)
            previous_end_was_high_r = False 
        elif is_last:
            sorted_idxs = sort_bin_by_radius(idxs, r_xy, high_to_low=False)
            previous_end_was_high_r = True
        else:
            start_high = previous_end_was_high_r
            sorted_idxs = sort_bin_by_radius(idxs, r_xy, high_to_low=start_high)
            previous_end_was_high_r = not start_high
        order_indices.extend(sorted_idxs.tolist())

# ── Safety Check ─────────────────────────────────────────────────────────────
seen = set(order_indices)
if len(seen) != N:
    missing = [i for i in range(N) if i not in seen]
    if missing:
        m = np.array(missing)
        m_sorted = m[np.argsort(phi_d[m])]
        order_indices.extend(m_sorted.tolist())
        print(f"Warning: {len(missing)} points were recovered.")

# ── Save Output ──────────────────────────────────────────────────────────────
out = df.copy()
out["order_idx"] = -1
out.loc[order_indices, "order_idx"] = np.arange(N)
out = out.sort_values("order_idx").reset_index(drop=True)

# Ensure 'order_idx' is placed before 'gen_settings'
cols = list(out.columns)
if "gen_settings" in cols:
    cols.remove("order_idx")
    cols.insert(cols.index("gen_settings"), "order_idx")
    out = out[cols]

out.to_csv(OUTPUT_PATH_PLAN, index=False)

if PRINT_SUMMARY:
    print(f"Saved {OUTPUT_PATH_PLAN}")
    print(f"Points: {N}")
    print("Strategy: Side(-180->180) -> Top(180->-180) -> Bottom(-180->180)")

# ── Interactive 3D Replay Class ──────────────────────────────────────────────
class PathReplayPlayer:
    def __init__(self, df_ordered):
        self.df = df_ordered
        self.N = len(df_ordered)
        
        # Prepare geometry
        self.phi_arr = self.df["phi_deg"].to_numpy() # Keep raw phi for display
        self.r_m = self.df["r_xy_mm"].to_numpy() / 1000.0
        self.z_m = self.df["z_mm"].to_numpy() / 1000.0
        self.phi_rad = np.radians(self.phi_arr)
        
        self.x = self.r_m * np.cos(self.phi_rad)
        self.y = self.r_m * np.sin(self.phi_rad)
        self.z = self.z_m

        # State
        self.curr_idx = 0
        self.is_playing = False
        self.ppm = 600.0
        self.tail_length = 50
        self.use_history_fading = False

        # Setup Figure
        self.fig = plt.figure(figsize=(12, 9)) # Slightly larger for readout
        self.fig.subplots_adjust(bottom=0.25, left=0.1) # Room for text on left
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.scatter(self.x, self.y, self.z, c='lightgray', s=5, alpha=0.2, depthshade=False)
        self.line_hist, = self.ax.plot([], [], [], c='gray', alpha=0.3, linewidth=1.0)
        self.line_active, = self.ax.plot([], [], [], c='red', linewidth=2.0)
        self.head_pt, = self.ax.plot([], [], [], marker='o', c='blue', markersize=6)

        # Coordinate Readout Text (Top Left)
        # transform=self.ax.transAxes puts 0,0 at bottom-left and 1,1 at top-right
        self.txt_readout = self.ax.text2D(
            0.02, 0.95, "", transform=self.ax.transAxes,
            fontsize=9, family='monospace', verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        self._set_axes_equal()
        self.ax.set_title("Path Replay: Stopped")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        self._init_widgets()
        
        self.anim = animation.FuncAnimation(
            self.fig, self._on_frame, interval=100, save_count=100, cache_frame_data=False
        )
        
        self.update_plot()
        plt.show()

    def _set_axes_equal(self):
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

    def _init_widgets(self):
        # Layouts
        ax_rewind = plt.axes([0.1, 0.12, 0.06, 0.05])
        ax_back   = plt.axes([0.17, 0.12, 0.06, 0.05])
        ax_play   = plt.axes([0.24, 0.12, 0.06, 0.05])
        ax_fwd    = plt.axes([0.31, 0.12, 0.06, 0.05])
        
        ax_speed_box = plt.axes([0.45, 0.12, 0.1, 0.05])
        ax_tail_sld  = plt.axes([0.65, 0.12, 0.25, 0.03])
        
        ax_hist_tog  = plt.axes([0.65, 0.06, 0.15, 0.05])
        
        # Progress Slider & Text
        ax_idx_text  = plt.axes([0.05, 0.04, 0.15, 0.04]) 
        ax_prog_sld  = plt.axes([0.22, 0.04, 0.68, 0.04])

        self.btn_rewind = Button(ax_rewind, '|<')
        self.btn_back   = Button(ax_back, '<')
        self.btn_play   = Button(ax_play, 'Play')
        self.btn_fwd    = Button(ax_fwd, '>')
        
        self.text_speed = TextBox(ax_speed_box, 'Rate (PPM): ', initial=str(int(self.ppm)))
        self.sld_tail = Slider(ax_tail_sld, 'Tail Len', 1, 500, valinit=self.tail_length, valstep=1)
        self.chk_hist = CheckButtons(ax_hist_tog, ['Fade History'], [False])
        
        self.sld_prog = Slider(ax_prog_sld, '', 0, self.N - 1, valinit=0, valfmt='%0.0f')

        self.ax_idx = ax_idx_text
        self.ax_idx.axis('off')
        self.txt_idx = self.ax_idx.text(0, 0.5, f"Point: 0 / {self.N}")

        # Events
        self.btn_rewind.on_clicked(self.rewind)
        self.btn_back.on_clicked(self.step_back)
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_fwd.on_clicked(self.step_fwd)
        self.text_speed.on_submit(self.set_speed)
        self.sld_tail.on_changed(self.set_tail_length)
        self.chk_hist.on_clicked(self.toggle_history_mode)
        self.sld_prog.on_changed(self.set_progress)

    def _on_frame(self, frame):
        if self.is_playing:
            if self.curr_idx < self.N - 1:
                self.curr_idx += 1
                self.update_plot()
            else:
                self.is_playing = False
                self.btn_play.label.set_text("Play")
                self.ax.set_title("Replay Finished")
        return self.head_pt,

    def update_plot(self):
        i = self.curr_idx
        start_active = max(0, i - self.tail_length) if self.use_history_fading else 0

        # Update Graphics
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

        # Update UI Text
        self.txt_idx.set_text(f"Point: {i} / {self.N-1}")

        # Update Readout (Coordinate List)
        self._update_readout(i)

        if self.sld_prog.val != i:
            self.sld_prog.eventson = False
            self.sld_prog.set_val(i)
            self.sld_prog.eventson = True

        self.fig.canvas.draw_idle()

    def _update_readout(self, center_idx):
        lines = []
        # We want 11 lines total: current ± 5
        start_offset = -5
        end_offset = 5
        
        for offset in range(start_offset, end_offset + 1):
            idx = center_idx + offset
            
            if 0 <= idx < self.N:
                # Convert back to mm for display readability
                x_mm = self.x[idx] * 1000.0
                y_mm = self.y[idx] * 1000.0
                z_mm = self.z[idx] * 1000.0
                phi  = self.phi_arr[idx]
                
                # Marker for current line
                prefix = "->" if offset == 0 else "  "
                
                # Format: "-> 1234 | X: 12.3 Y: 45.6 Z: 78.9 A: 180.0"
                line = (f"{prefix} {idx:4d} | "
                        f"X{x_mm:6.1f} "
                        f"Y{y_mm:6.1f} "
                        f"Z{z_mm:6.1f} "
                        f"A{phi:6.1f}")
                lines.append(line)
            else:
                # Placeholder for out of bounds
                lines.append("        ---")
        
        self.txt_readout.set_text("\n".join(lines))

    def set_progress(self, val):
        idx = int(val)
        if idx != self.curr_idx:
            self.curr_idx = idx
            self.update_plot()

    def toggle_play(self, event):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text("Stop" if self.is_playing else "Play")
        self.ax.set_title("Playing..." if self.is_playing else "Paused")
        self.set_timer_interval()

    def rewind(self, event):
        self.curr_idx = 0
        self.is_playing = False
        self.btn_play.label.set_text("Play")
        self.update_plot()

    def step_fwd(self, event):
        self.is_playing = False
        self.btn_play.label.set_text("Play")
        if self.curr_idx < self.N - 1:
            self.curr_idx += 1
            self.update_plot()

    def step_back(self, event):
        self.is_playing = False
        self.btn_play.label.set_text("Play")
        if self.curr_idx > 0:
            self.curr_idx -= 1
            self.update_plot()

    def set_speed(self, text):
        try:
            val = float(text)
            if val > 0:
                self.ppm = val
                self.set_timer_interval()
        except ValueError:
            pass

    def set_timer_interval(self):
        ms = (60.0 / self.ppm) * 1000.0
        ms = max(10, ms) 
        self.anim.event_source.interval = ms

    def set_tail_length(self, val):
        self.tail_length = int(val)
        if not self.is_playing:
            self.update_plot()

    def toggle_history_mode(self, label):
        self.use_history_fading = not self.use_history_fading
        if not self.is_playing:
            self.update_plot()

if __name__ == "__main__":
    player = PathReplayPlayer(out)