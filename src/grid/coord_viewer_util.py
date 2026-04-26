#!/usr/bin/env python3
"""
Standalone Tkinter Wrapper for the 3D Replay Viewer
---------------------------------------------------
This script acts as the "Driver" or "Wrapper" for the pure rendering engine 
found in `coord_viewer_core.py`. 

Features:
- Creates a standalone desktop application window using Tkinter.
- Builds a user interface (UI) with buttons, sliders, and checkboxes.
- Handles File I/O: Can open CSV files or scan directories of WAV files 
  to extract coordinate data from their filenames.
- Embeds the Matplotlib 3D figure directly into the Tkinter window.
- Demonstrates how to connect GUI events (like button clicks) to the 
  framework-agnostic methods of the `CoordViewerEngine`.
"""

import os  # Operating system interactions (paths, directories)
import sys  # System-specific parameters and functions (like path modification)
import glob  # Used for finding all files in a folder that match a pattern (e.g., *.wav)
import pandas as pd  # Used for creating and manipulating data tables (DataFrames)
import numpy as np  # Used for math operations and arrays
import tkinter as tk  # The standard Python GUI toolkit
from tkinter import filedialog, messagebox, ttk  # Specialized Tkinter modules for dialogs and modern widgets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # The crucial bridge that lets Matplotlib draw inside a Tkinter window

from coord_viewer_core import CoordViewerEngine  # Import our pure rendering engine

# --- Helper function for file parsing ---
# This function looks at a folder full of audio files, extracts the 3D coordinates 
# from their filenames, and builds a dataset we can plot.
def _get_xyz_from_wavs(wav_dir):
    # Add the process directory to the system path so we can import our custom 'utils' script
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'process')))
    try:
        # Import helper functions to decode filenames and convert coordinate types
        from utils import parse_coords_from_filename, spherical_to_cartesian
    except ImportError:
        # Show a helpful error if the utility script is missing
        raise ImportError("Could not import utils.py. Ensure it is located in the process directory.")
        
    # Find every file ending in '.wav' in the chosen directory
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {wav_dir}")

    # Create empty lists to hold the extracted coordinates
    r_list, th_list, ph_list = [], [], []
    for f in wav_files:
        # Extract just the filename (e.g., 'point_1.wav') from the full path
        fname = os.path.basename(f)
        try:
            # Use the imported function to decode the spherical coordinates from the filename string
            r_sph, theta, phi = parse_coords_from_filename(fname)
            r_list.append(r_sph)
            th_list.append(theta)
            ph_list.append(phi)
        except Exception:
            # If a file doesn't match our naming convention, safely ignore it and move on
            pass
            
    if not r_list:
        raise ValueError(f"No valid coordinate-encoded .wav files found in {wav_dir}")

    # Convert the collected spherical coordinates (Radius, Theta, Phi) into standard Cartesian (X, Y, Z)
    x, y, z = spherical_to_cartesian(np.array(r_list), np.array(th_list), np.array(ph_list))
    
    # Build a Pandas DataFrame to hold the final converted data
    # The engine expects polar-cylindrical columns (r_xy_mm, phi_deg, z_mm)
    df = pd.DataFrame({
        'r_xy_mm': np.hypot(x, y) * 1000.0,      # Calculate distance from center in the XY plane, convert to mm
        'phi_deg': np.degrees(np.arctan2(y, x)), # Calculate the azimuth angle in degrees
        'z_mm': z * 1000.0                       # Convert Z height to mm
    })
    return df

# ── The Native Tkinter Wrapper ────────────────────────────────────────────
class TkinterCoordApp:
    def __init__(self, root):
        # Save a reference to the main Tkinter window
        self.root = root
        self.root.title("Standalone 3D Replay Viewer")
        self.root.geometry("1100x850")
        
        # Instantiate our pure rendering engine
        # Notice we don't pass any data yet; it starts empty.
        self.engine = CoordViewerEngine()

        # Build all the buttons and sliders
        self._build_ui()
        # Embed the Matplotlib figure into the Tkinter window
        self._pack_canvas()
        
        # Start a repeating timer to keep the UI slider in sync with the animation
        self.root.after(100, self._sync_ui_state)

    def _build_ui(self):
        # Create a container frame at the bottom of the window for all our controls
        ui_frame = tk.Frame(self.root, padx=10, pady=5)
        ui_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # -- Row 1: File & Scrubbing --
        # Create a sub-frame for the first row of controls
        row1 = tk.Frame(ui_frame)
        row1.pack(fill=tk.X, pady=2)
        # Add buttons to load data, linking their 'command' to methods defined later in this class
        tk.Button(row1, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="Load WAVs", command=self.load_wav_dir).pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="Toggle Coords", command=self.toggle_coords).pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1, text="Timeline:").pack(side=tk.LEFT, padx=5)
        # Create a slider (Scale) to scrub through the animation manually
        self.scrub_slider = tk.Scale(row1, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_scrub)
        self.scrub_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Label to show the exact point index we are currently looking at
        self.lbl_idx = tk.Label(row1, text="Point: 0 / 0", width=15)
        self.lbl_idx.pack(side=tk.LEFT, padx=5)

        # -- Row 2: VCR & Playback Config --
        row2 = tk.Frame(ui_frame)
        row2.pack(fill=tk.X, pady=2)
        # These buttons directly call methods on our pure engine instance!
        tk.Button(row2, text="|<", command=self.engine.rewind, width=4).pack(side=tk.LEFT, padx=2)
        tk.Button(row2, text="<", command=self.engine.step_back, width=4).pack(side=tk.LEFT, padx=2)
        self.btn_play = tk.Button(row2, text="Play", command=self.toggle_play, width=6)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        tk.Button(row2, text=">", command=self.engine.step_fwd, width=4).pack(side=tk.LEFT, padx=2)

        tk.Label(row2, text="Rate (PPM):").pack(side=tk.LEFT, padx=(15,2))
        # Create a text entry box for playback speed
        self.ent_speed = tk.Entry(row2, width=6)
        self.ent_speed.insert(0, str(int(self.engine.ppm)))
        # Bind the Enter key ("<Return>") so pressing it updates the engine's speed
        # We use a lambda function here to capture the event and pass the value
        self.ent_speed.bind("<Return>", lambda e: self.engine.set_speed(float(self.ent_speed.get())))
        self.ent_speed.pack(side=tk.LEFT)

        tk.Label(row2, text="Tail Len:").pack(side=tk.LEFT, padx=(15,2))
        # Slider to adjust how long the red trail is behind the active point
        self.scl_tail = tk.Scale(row2, from_=1, to=500, orient=tk.HORIZONTAL, command=lambda v: self.engine.set_tail_length(int(v)))
        self.scl_tail.set(self.engine.tail_length)
        self.scl_tail.pack(side=tk.LEFT)

        # A BooleanVar is a special Tkinter variable that automatically tracks True/False states
        self.var_hist = tk.BooleanVar(value=False)
        tk.Checkbutton(row2, text="Fade Hist", variable=self.var_hist, command=lambda: self.engine.set_history_mode(self.var_hist.get())).pack(side=tk.LEFT, padx=15)

        # -- Row 3: Visual & Camera Settings --
        row3 = tk.Frame(ui_frame)
        row3.pack(fill=tk.X, pady=2)
        # Camera preset buttons. They tell the engine to snap to specific Elevation and Azimuth angles.
        tk.Button(row3, text="Top", command=lambda: self.engine.set_view(90, 0)).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="Front", command=lambda: self.engine.set_view(0, 0)).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="Side", command=lambda: self.engine.set_view(0, -90)).pack(side=tk.LEFT, padx=2)

        tk.Label(row3, text="Color:").pack(side=tk.LEFT, padx=(15,2))
        # Sliders for visual tweaks
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
        # Create a frame taking up the rest of the top of the window
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # FigureCanvasTkAgg bridges Matplotlib to Tkinter. We give it the engine's figure and the frame to live in.
        self.canvas = FigureCanvasTkAgg(self.engine.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_csv(self):
        # Open a standard OS file dialog asking the user to pick a CSV
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            # Pass the path directly to the engine
            self.engine.load_data(path)
            # Adjust the scrub slider so it knows how many points are in the new file
            self._update_slider_range()
            
    def load_wav_dir(self):
        # Ask the user to select a folder
        path = filedialog.askdirectory(title="Select directory containing WAV files")
        if path:
            try:
                # Use our helper function to extract data from the folder
                df = _get_xyz_from_wavs(path)
                # Pass the resulting DataFrame to the engine
                self.engine.load_data(df)
                self._update_slider_range()
            except Exception as e:
                # If anything goes wrong (bad files, no files), show a pop-up error box
                messagebox.showerror("Parsing Error", str(e))

    def _update_slider_range(self):
        # Update the maximum limit of the timeline slider based on the data length
        max_idx = max(0, self.engine.N - 1)
        self.scrub_slider.config(to=max_idx)

    def on_scrub(self, val):
        # Called whenever the user drags the timeline slider
        self.engine.set_current_index(int(val))

    def toggle_play(self):
        # Swap between playing and paused, updating the button text accordingly
        if self.engine.is_playing:
            self.engine.pause()
            self.btn_play.config(text="Play")
        else:
            self.engine.play()
            self.btn_play.config(text="Pause")

    def toggle_coords(self):
        # Turn the bottom text readout on or off
        self.engine.toggle_readout()

    def toggle_rotation(self):
        # Start or stop the automated camera panning
        if self.engine.is_rotating:
            self.engine.stop_rotation()
            self.btn_rot.config(text="Rotate")
        else:
            try: angle = float(self.ent_rot.get())
            except ValueError: angle = 45.0
            self.engine.start_rotation(angle)
            self.btn_rot.config(text="Stop Rot")

    def _sync_ui_state(self):
        # This function runs constantly in the background (every 50ms)
        # It ensures our UI labels and sliders match what the engine is actually doing.
        max_idx = max(0, self.engine.N - 1)
        self.lbl_idx.config(text=f"Point: {self.engine.curr_idx} / {max_idx}")
        
        # If the animation is playing, automatically drag the slider forward
        if self.engine.is_playing and self.scrub_slider.get() != self.engine.curr_idx:
            self.scrub_slider.set(self.engine.curr_idx)
            
        # If the animation finished on its own, reset the Play button text
        if not self.engine.is_playing and self.btn_play['text'] == "Pause":
            self.btn_play.config(text="Play")
            
        # Ask Tkinter to call this function again in 50 milliseconds
        self.root.after(50, self._sync_ui_state)

# This block only runs if this script is executed directly (not imported by another file)
if __name__ == "__main__":
    def on_closing():
        # Cleanly stop background animation timers so Python can close completely
        if hasattr(app.engine, 'anim') and app.engine.anim is not None:
            app.engine.anim.event_source.stop()
        if hasattr(app.engine, 'rot_anim') and app.engine.rot_anim is not None:
            app.engine.rot_anim.event_source.stop()
        root.quit()
        root.destroy()
        sys.exit(0)

    # Create the root window
    root = tk.Tk()
    # Intercept the "X" close button on the window to run our cleanup function
    root.protocol("WM_DELETE_WINDOW", on_closing)
    # Create our application class, attaching it to the root window
    app = TkinterCoordApp(root)
    # Start the Tkinter event loop (this blocks forever, waiting for user clicks)
    root.mainloop()