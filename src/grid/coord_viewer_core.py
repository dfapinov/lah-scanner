#!/usr/bin/env python3
"""
Pure Interactive 3D Replay Viewer Engine
-------------------------------------------------
This module provides a pure, GUI-framework agnostic rendering engine 
for visualizing 3D coordinate grids and paths. 

Features:
- 3D visualization of coordinate paths.
- Animated playback (scrubbing, play, pause, rewind).
- Auto-rotating camera views. Path history tail.
- Framework-agnostic design: This class only uses Matplotlib and
  exposes a clean API (methods like `play()`, `pause()`, `set_speed()`) 
  that allows any parent GUI to embed and control it easily.

Example Usage:
Please see `coord_viewer_util.py` for a complete example of how to wrap 
this engine inside a standalone Tkinter application.
"""

import numpy as np  # Used for high-performance mathematical operations and arrays
import matplotlib.pyplot as plt  # Core plotting library to create figures and axes
import matplotlib.colors as mcolors  # Used to create custom color gradients
import matplotlib.animation as animation  # Used to create background loops for playback
import pandas as pd  # Used for handling tabular data (DataFrames)

class CoordViewerEngine:
    """
    Pure rendering engine. Knows nothing about UI widgets or file paths.
    Expects input data as a pandas DataFrame, dict, or CSV path.
    """
    def __init__(self, input_data=None):
        # --- Playback State ---
        # These variables keep track of where we are in the animation
        self.curr_idx = 0  # The current discrete point index being displayed
        self.exact_idx = 0.0  # A floating-point index for smooth speed calculations
        self.is_playing = False  # Boolean flag to know if the animation is currently running
        self.ppm = 600.0  # Points Per Minute (playback speed)
        self.tail_length = 50  # How many previous points to show trailing behind the current point
        self.use_history_fading = False  # Toggle whether to show the entire past path faintly
        self.use_ortho = False  # Toggle between orthographic and perspective 3D projections
        self.timer_interval_ms = 50  # How often the animation loop updates (in milliseconds)
        self.show_readout = True  # Toggle the text box showing coordinates

        # --- Rotation Animation State ---
        # These variables handle the automated camera panning back and forth
        self.is_rotating = False  # Flag to know if the camera is currently panning
        self.rot_full_angle = 45.0  # The total angle sweep of the camera pan
        self.rot_target_angle = 22.5  # The half-way target to reverse direction
        self.rot_dir = 1  # Direction multiplier (1 for right, -1 for left)
        self.rot_accumulated = 0.0  # Tracks how far the camera has rotated in the current sweep

        # --- Setup Figure ---
        # Create the main Matplotlib figure window. figsize is in inches (width, height).
        self.fig = plt.figure(figsize=(10, 8))
        # Adjust the margins of the plot to make room for the text readout at the bottom
        self.fig.subplots_adjust(top=1.0, bottom=0.10, left=0.0, right=1.0)
        # Add a 3D axis to the figure
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Set the labels for the 3 axes
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        # Set the initial camera viewing angle (elevation and azimuth)
        self.ax.view_init(elev=30, azim=135)

        # --- Custom Colormap ---
        # Creates a color gradient that starts at black and transitions through the HSV rainbow
        self.custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            'black_hsv', 
            ['black'] + [plt.get_cmap('hsv')(i) for i in np.linspace(0, 1, 256)]
        )

        # --- Fast line plots ---
        # To make animations fast, we don't redraw the whole plot. Instead, we create empty 
        # line objects here (with [], [], []) and just update their data arrays later.
        self.base_pts, = self.ax.plot([], [], [], marker='o', linestyle='none', color=self.custom_cmap(0.0), markersize=2, alpha=0.2)
        self.line_hist, = self.ax.plot([], [], [], c='gray', alpha=0.3, linewidth=1.0)
        self.line_active, = self.ax.plot([], [], [], c='red', linewidth=2.0)
        self.head_pt, = self.ax.plot([], [], [], marker='o', c='blue', markersize=6)

        # --- Coordinate Readout Text ---
        # Place a text box anchored to the figure window (not the 3D space) at 2% X and 2% Y
        self.txt_readout = self.fig.text(
            0.02, 0.02, "",
            fontsize=9, family='monospace', verticalalignment='bottom',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        # Initialize default data properties as empty numpy arrays
        self.N = 0 
        self.x = self.y = self.z = self.phi_arr = np.array([]) 

        # If data was passed when creating the engine, load it immediately
        if input_data is not None:
            self.load_data(input_data)

        # --- Internal animation loops ---
        # These are background timers that call a function repeatedly at set intervals.
        # `self._on_frame` handles the playback movement. `self._on_rotate_frame` handles camera panning.
        self.anim = animation.FuncAnimation(self.fig, self._on_frame, interval=self.timer_interval_ms, cache_frame_data=False)
        self.rot_anim = animation.FuncAnimation(self.fig, self._on_rotate_frame, interval=20, cache_frame_data=False)

    def load_data(self, input_data):
        # Determine what type of data was passed in and standardize it into a Pandas DataFrame
        if isinstance(input_data, pd.DataFrame):
            df = input_data
        elif isinstance(input_data, dict):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, str) and input_data.lower().endswith('.csv'):
            df = pd.read_csv(input_data)
        else:
            raise TypeError("Engine load_data expects a pandas DataFrame, dict, or CSV file path.")

        # Extract specific columns into fast Numpy arrays
        self.phi_arr = df["phi_deg"].to_numpy()
        self.N = len(self.phi_arr)  # Total number of points
        self.r_m = df["r_xy_mm"].to_numpy() / 1000.0  # Convert mm to meters
        self.z_m = df["z_mm"].to_numpy() / 1000.0  # Convert mm to meters
        self.phi_rad = np.radians(self.phi_arr)  # Convert degrees to radians for math functions
        
        # Convert polar coordinates (radius, angle) into cartesian coordinates (X, Y)
        self.x = -self.r_m * np.cos(self.phi_rad)
        self.y = self.r_m * np.sin(self.phi_rad)
        self.z = self.z_m

        # Update the base (background) points with the full dataset
        self.base_pts.set_data(self.x, self.y)
        self.base_pts.set_3d_properties(self.z)

        # Ensure our playback index hasn't exceeded the length of the new dataset
        if self.curr_idx >= self.N:
            self.curr_idx = max(0, self.N - 1)
            self.exact_idx = float(self.curr_idx)

        # Adjust the 3D axis limits so the plot doesn't look stretched or squished
        self._set_axes_equal()
        # Force the plot to redraw with the new data
        self.update_plot()

    def _set_axes_equal(self):
        # Matplotlib 3D axes don't naturally maintain a 1:1:1 aspect ratio.
        # This function calculates the largest spread of data and forces all axes to use that range.
        if self.N == 0: return
        # Find the maximum range across all 3 dimensions
        max_range = np.array([
            self.x.max() - self.x.min(),
            self.y.max() - self.y.min(),
            self.z.max() - self.z.min()
        ]).max() / 2.0
        
        # Find the center point of the data
        mid_x = (self.x.max() + self.x.min()) * 0.5
        mid_y = (self.y.max() + self.y.min()) * 0.5
        mid_z = (self.z.max() + self.z.min()) * 0.5
        
        # Apply the limits so the bounding box is a perfect cube
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- External Control API ---
    # These methods are designed to be called by buttons and sliders in the parent GUI.

    def set_current_index(self, idx):
        # Jump to a specific point in the playback (e.g., from a slider)
        if self.is_playing: return  # Ignore manual scrub if currently playing
        idx = max(0, min(int(idx), self.N - 1))  # Clamp value to valid range
        if idx != self.curr_idx:
            self.curr_idx = idx
            self.exact_idx = float(idx)
            self.update_plot()

    def play(self): 
        # Start animation
        self.is_playing = True
        self.ax.set_title("Playing...")
        
    def pause(self): 
        # Stop animation
        self.is_playing = False
        self.ax.set_title("Paused")
        
    def rewind(self): 
        # Reset animation to the beginning
        self.pause()
        self.curr_idx = 0
        self.exact_idx = 0.0
        self.update_plot()
    
    def step_fwd(self):
        # Move forward exactly one point
        self.pause()
        if self.curr_idx < self.N - 1: 
            self.curr_idx += 1
            self.update_plot()

    def step_back(self):
        # Move backward exactly one point
        self.pause()
        if self.curr_idx > 0: 
            self.curr_idx -= 1
            self.update_plot()

    def set_speed(self, ppm):
        # Update playback speed (Points Per Minute)
        if ppm > 0: self.ppm = ppm

    def set_tail_length(self, val):
        # Update how long the red highlighted trail should be
        self.tail_length = int(val)
        if not self.is_playing: self.update_plot()

    def set_history_mode(self, enabled: bool):
        # Toggle the grey faded history line
        self.use_history_fading = enabled
        if not self.is_playing: self.update_plot()

    def set_ortho(self, enabled: bool):
        # Switch between orthographic (flat 3D) and perspective (depth 3D) rendering
        self.use_ortho = enabled
        self.ax.set_proj_type('ortho' if self.use_ortho else 'persp')
        self.fig.canvas.draw_idle()  # Request a redraw from matplotlib

    def set_view(self, elev, azim):
        # Snap the camera to a specific angle
        self.ax.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def set_alpha(self, val):
        # Change the transparency of the background points
        self.base_pts.set_alpha(val)
        self.fig.canvas.draw_idle()

    def set_color(self, val):
        # Pick a new color from our custom rainbow colormap
        self.base_pts.set_color(self.custom_cmap(val))
        self.fig.canvas.draw_idle()

    def toggle_readout(self, force_state=None):
        # Turn the bottom text box on or off
        self.show_readout = not self.show_readout if force_state is None else force_state
        self.txt_readout.set_visible(self.show_readout)
        
        # Adjust bottom margin to make room for the readout box if it's visible
        bottom_margin = 0.15 if self.show_readout else 0.05
        self.fig.subplots_adjust(bottom=bottom_margin)
        self.fig.canvas.draw_idle()

    def start_rotation(self, angle=45.0):
        # Initiate the automated camera panning
        self.is_rotating = True
        self.rot_full_angle = angle
        self.rot_target_angle = angle / 2.0
        self.rot_dir = 1
        self.rot_accumulated = 0.0

    def stop_rotation(self):
        # Stop the automated camera panning
        self.is_rotating = False

    # --- Render Logic ---
    def update_plot(self):
        # This is the core function that updates the visuals based on the current state.
        if self.N == 0: return

        i = self.curr_idx  # Current point index
        # Calculate where the red highlighted trail should start
        start_active = max(0, i - self.tail_length) if self.use_history_fading else 0

        # Update the blue "head" point to the current XYZ location
        self.head_pt.set_data([self.x[i]], [self.y[i]])
        self.head_pt.set_3d_properties([self.z[i]])
        
        # Update the red active line to stretch from start_active to the current point
        self.line_active.set_data(self.x[start_active:i+1], self.y[start_active:i+1])
        self.line_active.set_3d_properties(self.z[start_active:i+1])
        
        # If history mode is on, draw a gray line for all points *before* the active tail
        if self.use_history_fading and start_active > 0:
            self.line_hist.set_data(self.x[0:start_active], self.y[0:start_active])
            self.line_hist.set_3d_properties(self.z[0:start_active])
        else:
            # Otherwise, clear the gray history line
            self.line_hist.set_data([], [])
            self.line_hist.set_3d_properties([])

        # Update the text box showing the coordinate readout
        if self.show_readout:
            lines = []
            # Displaying 4 lines: 1 back, current, 2 forward
            for offset in range(-1, 3):
                idx = i + offset
                if 0 <= idx < self.N:
                    prefix = "->" if offset == 0 else "  "  # Add an arrow to the current row
                    # Format the text with proper spacing and decimal places
                    lines.append(f"{prefix} {idx:4d} | X{self.x[idx]*1000:6.1f} Y{self.y[idx]*1000:6.1f} Z{self.z[idx]*1000:6.1f} A{self.phi_arr[idx]:6.1f}")
                else:
                    lines.append("        ---")  # Placeholder if out of bounds
            self.txt_readout.set_text("\n".join(lines))

        # Tell matplotlib to redraw the figure when it has free time
        self.fig.canvas.draw_idle()

    def _on_frame(self, frame):
        # This method is called repeatedly by the animation timer.
        if self.is_playing and self.N > 0:
            # Calculate how many points we should advance based on our speed (ppm) and timer interval
            step = (self.ppm / 60.0) * (self.timer_interval_ms / 1000.0)
            self.exact_idx += step
            new_idx = int(self.exact_idx)
            
            # Only update the plot if we've accumulated enough steps to move to the next integer index
            if new_idx > self.curr_idx:
                if new_idx >= self.N:
                    # Reached the end of the data, stop playing
                    self.curr_idx = self.N - 1
                    self.pause()
                    self.ax.set_title("Replay Finished")
                else:
                    self.curr_idx = new_idx
                # Actually redraw the plot with the new index
                self.update_plot()
        # Animation functions must return an iterable of artists they modified
        return self.head_pt,

    def _on_rotate_frame(self, frame):
        # This method is called repeatedly by the secondary rotation animation timer.
        if not self.is_rotating: return None
        
        step = 0.25  # Degrees to rotate per frame
        self.ax.azim += (step * self.rot_dir)  # Modify the camera's azimuth angle
        self.rot_accumulated += step
        
        # Reverse direction if we've hit our target sweep angle
        if self.rot_accumulated >= self.rot_target_angle:
            self.rot_dir *= -1
            self.rot_accumulated = 0.0
            self.rot_target_angle = self.rot_full_angle
            
        self.fig.canvas.draw_idle()
        return None