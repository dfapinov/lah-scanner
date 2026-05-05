import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import traceback
import pandas as pd
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import queue
import datetime

# Append script directories to sys.path so we can import them as libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'grid'))
sys.path.append(os.path.join(current_dir, 'process'))

try:
    from grid_gen import generate_measurement_grid
    from path_plan import plan_path
    from coord_viewer_core import CoordViewerEngine
    from stage1_fdwsmooth import fdwsmooth
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import project scripts:\n{e}")

# --- Configuration ---
# Set to True to enable debug logging. This will write all CLI output to a timestamped 'hals_debug_*.log' file 
# and mirror it back to the original console terminal for easier debugging.
DEBUG_MODE = False # True / False


class QueueRedirector:
    def __init__(self, queue_obj, original_stream=None, log_file_obj=None):
        self.queue = queue_obj
        self.original_stream = original_stream
        self.log_file_obj = log_file_obj
        self._at_line_start = True

    def write(self, text):
        self.queue.put(text)
        if self.original_stream:
            self.original_stream.write(text)
        if self.log_file_obj:
            if not text:
                return
            
            text_for_log = text.replace('\r\n', '\n').replace('\r', '\n')
            
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f]")[:-4] + "] "
            
            lines = text_for_log.split('\n')
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    if self._at_line_start:
                        self.log_file_obj.write(f"{timestamp}{line}\n")
                    else:
                        self.log_file_obj.write(f"{line}\n")
                    self._at_line_start = True
                else:
                    if line:
                        if self._at_line_start:
                            self.log_file_obj.write(f"{timestamp}{line}")
                            self._at_line_start = False
                        else:
                            self.log_file_obj.write(line)
            self.log_file_obj.flush()

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()
        if self.log_file_obj:
            self.log_file_obj.flush()


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.id = None
        self.widget.bind("<Enter>", self.schedule_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def schedule_tip(self, event=None):
        if self.id:
            self.widget.after_cancel(self.id)
        self.id = self.widget.after(1000, self.show_tip)

    def show_tip(self):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 1
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1, font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollable_frame = ttk.Frame(self.canvas, padding="10")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_frame_configure(self, event):
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=(0, 0, bbox[2], max(bbox[3], self.canvas.winfo_height())))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=(0, 0, bbox[2], max(bbox[3], event.height)))

    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if self.canvas.yview() == (0.0, 1.0): return
        if getattr(event, 'num', 0) == 4: self.canvas.yview_scroll(-1, "units")
        elif getattr(event, 'num', 0) == 5: self.canvas.yview_scroll(1, "units")
        else: self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")


class SpkrScannerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Hide the main GUI window during initialization
        self.withdraw()
        
        # --- Splash Screen ---
        self.splash = tk.Toplevel(self)
        self.splash.overrideredirect(True)
        self.splash.attributes('-topmost', True)
        
        splash_img_path = os.path.join(current_dir, "splash.png")
        if os.path.exists(splash_img_path):
            self._splash_img = tk.PhotoImage(file=splash_img_path)
            w, h = self._splash_img.width(), self._splash_img.height()
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            x, y = int((sw - w) / 2), int((sh - h) / 2)
            self.splash.geometry(f"{w}x{h}+{x}+{y}")
            
            canvas = tk.Canvas(self.splash, width=w, height=h, highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            canvas.create_image(0, 0, anchor="nw", image=self._splash_img)
            canvas.create_text(w / 2, 20, text="Initializing components...", fill="white", font=("Arial", 15, "bold"))
        else:
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            w, h = 400, 200
            x = int((sw - w) / 2)
            y = int((sh - h) / 2)
            self.splash.geometry(f"{w}x{h}+{x}+{y}")
            
            splash_frame = tk.Frame(self.splash, bg="#2E3440", highlightbackground="#88C0D0", highlightcolor="#88C0D0", highlightthickness=2)
            splash_frame.pack(fill=tk.BOTH, expand=True)
            tk.Label(splash_frame, text="Speaker Scanner", font=("Arial", 24, "bold"), fg="#ECEFF4", bg="#2E3440").pack(expand=True)
            tk.Label(splash_frame, text="Initializing components...", font=("Arial", 10), fg="#D8DEE9", bg="#2E3440").pack(side=tk.BOTTOM, pady=10)
            
        self.splash.update() # Force draw immediately
        
        self._set_icon()

        self.title("HALS Post-Processing")
        self.geometry("1600x800")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Shared Data States
        self.project_dir = tk.StringVar(value=os.getcwd())
        self.project_name = tk.StringVar(value="MySpeaker")
        self.grid_data = None  # Holds the raw grid DataFrame
        self.planned_data = None  # Holds the planned path DataFrame

        # Track state for pop-up UI viewers
        self.stage5_viewer = None
        self.stage5_canvas = None
        self.stage5_update_job = None

        self._build_ui()
        
        # Schedule splash screen to close and main window to show after 1000ms
        self.after(1500, self._close_splash)

    def _set_icon(self):
        icon_path = os.path.join(current_dir, "HALS_icon.ico")
        png_path = os.path.join(current_dir, "HALS_icon.png")
        try:
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
            elif os.path.exists(png_path):
                img = tk.PhotoImage(file=png_path)
                self.iconphoto(True, img)
        except Exception as e:
            print(f"Note: Could not set window icon: {e}")

    def _close_splash(self):
        if hasattr(self, 'splash') and self.splash:
            self.splash.destroy()
        self.deiconify() # Reveal the fully initialized main window

    def report_callback_exception(self, exc, val, tb):
        err_msg = "".join(traceback.format_exception(exc, val, tb))
        try:
            print(f"Exception in UI Callback:\n{err_msg}", file=sys.stderr)
        except Exception:
            pass
        messagebox.showerror("UI Error", f"An unexpected UI error occurred:\n\n{err_msg}")

    def _build_ui(self):
        # --- STATUS BAR ---
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        self.left_panel = ttk.Frame(self.main_paned)
        self.right_panel = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_panel, weight=1)
        self.main_paned.add(self.right_panel, weight=1)

        # Configure grid for right_panel to allow swapping frames
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # --- TOP LEVEL: Project Directory ---
        top_frame = ttk.Frame(self.left_panel, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        top_frame.columnconfigure(1, weight=1)

        ttk.Label(top_frame, text="Project Name:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        ttk.Entry(top_frame, textvariable=self.project_name, width=20).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(top_frame, text="Save Project", command=self._action_save_project).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)

        ttk.Label(top_frame, text="Project Directory:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        ttk.Entry(top_frame, textvariable=self.project_dir, width=40).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(top_frame, text="Browse", command=self._browse_dir).grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

        # --- BOTTOM STATUS NOTE ---
        bottom_frame = ttk.Frame(self.left_panel)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        ttk.Label(bottom_frame, text="Hover over fields for help tooltips.", font=("Arial", 8, "italic")).pack(side=tk.LEFT)

        # --- MAIN TABS ---
        self.main_notebook = ttk.Notebook(self.left_panel)
        self.main_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Grid Operations
        self.tab_grid_ops = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.tab_grid_ops, text="Grid Generation & Planning")

        # Tab 2: Processing
        self.tab_processing = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.tab_processing, text="Processing Pipelines")
        
        self.proc_notebook = ttk.Notebook(self.tab_processing)
        self.proc_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_stage1 = ttk.Frame(self.proc_notebook)
        self.proc_notebook.add(self.tab_stage1, text="Stage 1: FDW & Smoothing")

        self.tab_stage2 = ttk.Frame(self.proc_notebook)
        self.proc_notebook.add(self.tab_stage2, text="Stage 2: Acoustic Origin")

        self.tab_stage3 = ttk.Frame(self.proc_notebook)
        self.proc_notebook.add(self.tab_stage3, text="Stage 3: Optimize SHE")

        self.tab_stage4 = ttk.Frame(self.proc_notebook)
        self.proc_notebook.add(self.tab_stage4, text="Stage 4: SHE Solve")

        self.tab_stage5 = ttk.Frame(self.proc_notebook)
        self.proc_notebook.add(self.tab_stage5, text="Stage 5: Extract Pressures")

        # --- RIGHT PANEL: View Swapping ---
        self.right_frame_plot = ttk.Frame(self.right_panel) # For Grid/Planning viewer
        self.right_frame_processing = ttk.Frame(self.right_panel) # For all Processing UI

        self.right_frame_plot.grid(row=0, column=0, sticky="nsew")
        self.right_frame_processing.grid(row=0, column=0, sticky="nsew")
        self.right_frame_plot.tkraise() # Start with Plot visible on top

        self.main_notebook.bind("<<NotebookTabChanged>>", self._on_main_tab_changed)
        self.proc_notebook.bind("<<NotebookTabChanged>>", self._on_proc_tab_changed)

        self.viewer = None
        self.canvas_viewer = None

        # --- CLI Output (Unified Right Panel for Processing) ---
        # This PanedWindow will live inside the right_frame_processing
        self.processing_paned_window = ttk.PanedWindow(self.right_frame_processing, orient=tk.VERTICAL)
        self.processing_paned_window.pack(fill=tk.BOTH, expand=True)

        self.stage5_viewer_frame = ttk.Frame(self.processing_paned_window)
        cli_container_frame = ttk.Frame(self.processing_paned_window)

        self.processing_paned_window.add(self.stage5_viewer_frame, weight=0) # Hidden by default
        self.processing_paned_window.add(cli_container_frame, weight=1)

        ttk.Label(cli_container_frame, text="Processing CLI Output:", font=("Arial", 10, "bold")).pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)

        self.cli_text = tk.Text(cli_container_frame, bg="black", fg="lightgray", font=("Consolas", 10), wrap=tk.WORD)
        cli_scrollbar = ttk.Scrollbar(cli_container_frame, orient="vertical", command=self.cli_text.yview)
        self.cli_text.configure(yscrollcommand=cli_scrollbar.set)
        
        cli_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.cli_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.cli_text.insert(tk.END, "Ready...\n")
        self.cli_text.config(state=tk.DISABLED)
        
        self.cli_queue = queue.Queue()
        
        if DEBUG_MODE:
            try:
                import datetime
                import platform
                import importlib

                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.debug_log_file = open(f"hals_debug_{timestamp_str}.log", "w", encoding="utf-8")
                self.debug_log_file.write(f"\n--- HALS GUI Debug Session Started: {datetime.datetime.now()} ---\n")
                self.debug_log_file.write(f"System: {platform.system()} {platform.release()} ({platform.version()})\n")
                self.debug_log_file.write(f"Python: {sys.version}\n")
                self.debug_log_file.write(f"Tkinter: {tk.TkVersion}\n")
                self.debug_log_file.write(f"Screen: {self.winfo_screenwidth()}x{self.winfo_screenheight()}\n")
                
                project_libs = ['numpy', 'scipy', 'pandas', 'matplotlib', 'h5py', 'soundfile', 'sounddevice', 'schema']
                self.debug_log_file.write("Project Libraries:\n")
                for lib in project_libs:
                    try:
                        mod = importlib.import_module(lib)
                        ver = getattr(mod, '__version__', 'Unknown')
                        self.debug_log_file.write(f"  - {lib}: {ver}\n")
                    except ImportError:
                        self.debug_log_file.write(f"  - {lib}: Not Installed\n")
                
                if platform.system() == "Linux":
                    session = os.environ.get("XDG_SESSION_TYPE", "Unknown")
                    desktop = os.environ.get("DESKTOP_SESSION", "Unknown")
                    wayland = os.environ.get("WAYLAND_DISPLAY", "Not Set")
                    self.debug_log_file.write(f"Linux Display: Session={session}, Desktop={desktop}, Wayland={wayland}\n")
                elif platform.system() == "Windows":
                    try:
                        import ctypes
                        awareness = ctypes.c_int(0)
                        error = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
                        if error == 0: # S_OK
                            awareness_map = {0: "Unaware", 1: "System-Aware", 2: "Per-Monitor-Aware"}
                            dpi_status = awareness_map.get(awareness.value, f"Unknown value ({awareness.value})")
                            self.debug_log_file.write(f"Windows DPI: {dpi_status}\n")
                        else:
                            self.debug_log_file.write(f"Windows DPI: Call failed (Error code: {error})\n")
                    except Exception as e:
                        self.debug_log_file.write(f"Windows DPI: Could not determine ({e})\n")
                self.debug_log_file.write("-" * 50 + "\n")
                sys.stdout = QueueRedirector(self.cli_queue, sys.__stdout__, self.debug_log_file)
                sys.stderr = QueueRedirector(self.cli_queue, sys.__stderr__, self.debug_log_file)
            except Exception as e:
                print(f"Warning: Failed to create debug log file: {e}")
                sys.stdout = QueueRedirector(self.cli_queue)
                sys.stderr = QueueRedirector(self.cli_queue)
        else:
            sys.stdout = QueueRedirector(self.cli_queue)
            sys.stderr = QueueRedirector(self.cli_queue)
            
        self._update_cli()

        self._build_grid_ops_ui()
        self._build_stage1_ui()
        self._build_stage2_ui()
        self._build_stage3_ui()
        self._build_stage4_ui()
        self._build_stage5_ui()

        # Attempt to load settings from the default directory on startup
        self._load_settings(self.project_dir.get())

        # Create the initial viewer for the default tab
        self._create_viewer()

        # Force the paned window divider line to the exact middle
        self._center_divider()
        
        self._last_geometry = self.geometry()
        self.bind("<Configure>", self._on_window_configure)

        if DEBUG_MODE:
            self._log_directory_tree(current_dir, "Program Directory Listing")
            self._log_directory_tree(self.project_dir.get(), "Project Directory Listing")

    def _log_directory_tree(self, path, title):
        if not DEBUG_MODE or not hasattr(self, 'debug_log_file') or not self.debug_log_file:
            return
        self.debug_log_file.write(f"\n{title} ({path}):\n")
        try:
            for root_dir, dirs, files in os.walk(path):
                if '__pycache__' in root_dir or '.git' in root_dir:
                    continue
                rel_path = os.path.relpath(root_dir, path)
                level = 0 if rel_path == '.' else rel_path.count(os.sep) + 1
                indent = ' ' * 4 * level
                folder_name = os.path.basename(root_dir) if os.path.basename(root_dir) else path
                self.debug_log_file.write(f"{indent}{folder_name}/\n")
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    if f.endswith('.pyc'): continue
                    try:
                        size = os.path.getsize(os.path.join(root_dir, f))
                        self.debug_log_file.write(f"{subindent}{f} ({size} bytes)\n")
                    except OSError:
                        self.debug_log_file.write(f"{subindent}{f} (Size unknown)\n")
        except Exception as e:
            self.debug_log_file.write(f"Could not read directory: {e}\n")
        self.debug_log_file.write("-" * 50 + "\n")
        self.debug_log_file.flush()

    def _on_window_configure(self, event):
        if DEBUG_MODE and event.widget == self:
            current_geometry = self.geometry()
            if current_geometry != self._last_geometry:
                print(f"[DEBUG] Main Window geometry changed to: {current_geometry}")
                self._last_geometry = current_geometry

    def _center_divider(self):
        w = self.main_paned.winfo_width()
        if w < 100:  # If the window hasn't fully drawn yet, wait and try again
            self.after(50, self._center_divider)
        else:
            if DEBUG_MODE:
                print(f"[DEBUG] Setting main paned window sash to {w // 2}")
            self.main_paned.sashpos(0, w // 2)

    def _on_main_tab_changed(self, event):
        selected_idx = self.main_notebook.index(self.main_notebook.select())
        if DEBUG_MODE:
            print(f"[DEBUG] Main tab changed to index: {selected_idx}")
        if selected_idx == 0:
            # Switched to Grid/Planning Tab
            self._create_viewer()
            self.right_frame_plot.tkraise()
        elif selected_idx == 1:
            # Switched to Processing Tab
            self.right_frame_processing.tkraise()
            self._destroy_viewer()

    def _on_proc_tab_changed(self, event):
        selected_idx = self.proc_notebook.index(self.proc_notebook.select())
        if DEBUG_MODE:
            print(f"[DEBUG] Processing notebook tab changed to index: {selected_idx}")
        # Stage 5 is the 5th tab, so index 4
        if selected_idx == 4:
            self._create_stage5_viewer()
            # Use after to ensure the window is drawn before setting sash
            self.after(50, lambda: self.processing_paned_window.sashpos(0, self.processing_paned_window.winfo_height() * 3 // 4))
        else:
            self._destroy_stage5_viewer()

    def _create_stage5_viewer(self):
        if self.stage5_viewer is not None:
            return

        from viewers import Stage5Viewer
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Add a control bar at the top of the viewer pane
        controls_frame = ttk.LabelFrame(self.stage5_viewer_frame, text="DUT (Visual Cue Only)")
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Link these entries to the same variables as the main settings
        self._add_labeled_entry(controls_frame, "Width (Y):", self.stage5_vars['dut_width_y'], 6)
        self._add_labeled_entry(controls_frame, "Depth (X):", self.stage5_vars['dut_depth_x'], 6)
        self._add_labeled_entry(controls_frame, "Height (Z):", self.stage5_vars['dut_height_z'], 6)

        note_text = "Note: Left mouse click to drag view, middle mouse click to shift view, right mouse click to zoom."
        self.stage5_note_label = ttk.Label(self.stage5_viewer_frame, text=note_text, font=("Arial", 8, "italic"))
        self.stage5_note_label.pack(side=tk.TOP, pady=(0, 2))

        if DEBUG_MODE:
            print("[DEBUG] Creating Stage 5 3D Viewer canvas")
        self.stage5_viewer = Stage5Viewer()
        self.stage5_canvas = FigureCanvasTkAgg(self.stage5_viewer.fig, master=self.stage5_viewer_frame)
        canvas_widget = self.stage5_canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Make the viewer pane visible
        self.processing_paned_window.pane(self.stage5_viewer_frame, weight=3)
        self.processing_paned_window.pane(self.processing_paned_window.panes()[1], weight=1)
        
        self._schedule_update_stage5_preview()

    def _destroy_stage5_viewer(self):
        if self.stage5_viewer is None:
            return

        if DEBUG_MODE:
            print("[DEBUG] Destroying Stage 5 3D Viewer canvas")
        # Hide the viewer pane
        self.processing_paned_window.pane(self.stage5_viewer_frame, weight=0)
        self.processing_paned_window.pane(self.processing_paned_window.panes()[1], weight=1)

        self.stage5_canvas.get_tk_widget().destroy()
        plt.close(self.stage5_viewer.fig)
        self.stage5_viewer = None
        self.stage5_canvas = None
        for widget in self.stage5_viewer_frame.winfo_children():
            widget.destroy()

    def _create_viewer(self):
        if self.viewer is None:
            self.viewer = CoordViewerEngine()

            # --- 1. Create the UI Control Toolbar ---
            self.viewer_ctrl_frame = ttk.Frame(self.right_frame_plot)
            self.viewer_ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
            
            # Row 1: Scrubbing
            row1 = ttk.Frame(self.viewer_ctrl_frame)
            row1.pack(fill=tk.X, pady=2)
            ttk.Label(row1, text="Scrub:").pack(side=tk.LEFT, padx=(10, 2))
            self.viewer_slider = ttk.Scale(row1, from_=0, to=100, orient=tk.HORIZONTAL, command=lambda v: self.viewer.set_current_index(int(float(v))))
            self.viewer_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            # Row 2: VCR, Speed, Tail, History
            row2 = ttk.Frame(self.viewer_ctrl_frame)
            row2.pack(fill=tk.X, pady=2)
            ttk.Button(row2, text="|<", command=self.viewer.rewind, width=3).pack(side=tk.LEFT, padx=2)
            ttk.Button(row2, text="<", command=self.viewer.step_back, width=3).pack(side=tk.LEFT, padx=2)
            self.btn_play = ttk.Button(row2, text="Play", command=self._toggle_viewer_play, width=6)
            self.btn_play.pack(side=tk.LEFT, padx=2)
            ttk.Button(row2, text=">", command=self.viewer.step_fwd, width=3).pack(side=tk.LEFT, padx=2)

            ttk.Label(row2, text="Rate (PPM):").pack(side=tk.LEFT, padx=(10, 2))
            self.ent_speed = ttk.Entry(row2, width=6)
            self.ent_speed.insert(0, str(int(self.viewer.ppm)))
            self.ent_speed.bind("<Return>", lambda e: self.viewer.set_speed(float(self.ent_speed.get())))
            self.ent_speed.pack(side=tk.LEFT)

            ttk.Label(row2, text="Tail Len:").pack(side=tk.LEFT, padx=(10, 2))
            self.scl_tail = ttk.Scale(row2, from_=1, to=500, orient=tk.HORIZONTAL, command=lambda v: self.viewer.set_tail_length(int(float(v))))
            self.scl_tail.set(self.viewer.tail_length)
            self.scl_tail.pack(side=tk.LEFT)

            self.var_hist = tk.BooleanVar(value=False)
            ttk.Checkbutton(row2, text="Fade Hist", variable=self.var_hist, command=lambda: self.viewer.set_history_mode(self.var_hist.get())).pack(side=tk.LEFT, padx=10)

            # Row 3: Views, Visuals, Rotation
            row3 = ttk.Frame(self.viewer_ctrl_frame)
            row3.pack(fill=tk.X, pady=2)
            ttk.Button(row3, text="Top", command=lambda: self.viewer.set_view(90, 0)).pack(side=tk.LEFT, padx=2)
            ttk.Button(row3, text="Front", command=lambda: self.viewer.set_view(0, 0)).pack(side=tk.LEFT, padx=2)
            ttk.Button(row3, text="Side", command=lambda: self.viewer.set_view(0, -90)).pack(side=tk.LEFT, padx=2)

            ttk.Label(row3, text="Color:").pack(side=tk.LEFT, padx=(10, 2))
            self.scl_col = ttk.Scale(row3, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=lambda v: self.viewer.set_color(float(v)))
            self.scl_col.pack(side=tk.LEFT)

            ttk.Label(row3, text="Opacity:").pack(side=tk.LEFT, padx=(10, 2))
            self.scl_alpha = ttk.Scale(row3, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=lambda v: self.viewer.set_alpha(float(v)))
            self.scl_alpha.set(0.2)
            self.scl_alpha.pack(side=tk.LEFT)

            self.var_ortho = tk.BooleanVar(value=False)
            ttk.Checkbutton(row3, text="Ortho", variable=self.var_ortho, command=lambda: self.viewer.set_ortho(self.var_ortho.get())).pack(side=tk.LEFT, padx=10)

            ttk.Label(row3, text="Rot Ang:").pack(side=tk.LEFT)
            self.ent_rot = ttk.Entry(row3, width=5)
            self.ent_rot.insert(0, "45")
            self.ent_rot.pack(side=tk.LEFT)

            self.btn_rot = ttk.Button(row3, text="Rotate", command=self._toggle_viewer_rotation)
            self.btn_rot.pack(side=tk.LEFT, padx=5)

            # --- 2. Pack the Canvas ---
            note_text = "Note: Left mouse click to drag view, middle mouse click to shift view, right mouse click to zoom."
            self.lbl_viewer_note = ttk.Label(self.right_frame_plot, text=note_text, font=("Arial", 8, "italic"))
            self.lbl_viewer_note.pack(side=tk.TOP, pady=(5, 0))

            # Create a dedicated container frame to prevent the canvas from overlapping siblings on Linux
            self.viewer_canvas_frame = ttk.Frame(self.right_frame_plot)
            self.viewer_canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            if DEBUG_MODE:
                print("[DEBUG] Creating Grid/Planning 3D Viewer canvas")
            self.canvas_viewer = FigureCanvasTkAgg(self.viewer.fig, master=self.viewer_canvas_frame)
            self.canvas_viewer.draw()
            self.canvas_viewer.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # --- 3. Reload Data ---
            if self.planned_data is not None:
                self.viewer.load_data(self.planned_data)
                self._update_viewer_slider_range()
            elif self.grid_data is not None:
                self.viewer.load_data(self.grid_data)
                self._update_viewer_slider_range()
            else:
                proj_name = self.project_name.get().strip() or "project"
                candidate_path = os.path.join(self.project_dir.get(), f"{proj_name}_scan_path.csv")
                if os.path.exists(candidate_path):
                    try:
                        self.viewer.load_data(candidate_path)
                        self._update_viewer_slider_range()
                    except Exception as e:
                        print(f"Note: Could not auto-reload path file on tab switch: {e}")
                        
            # Start UI sync loop
            self._sync_viewer_ui()

    def _toggle_viewer_play(self):
        if self.viewer.is_playing:
            self.viewer.pause()
            self.btn_play.config(text="Play")
        else:
            self.viewer.play()
            self.btn_play.config(text="Pause")

    def _toggle_viewer_rotation(self):
        if self.viewer.is_rotating:
            self.viewer.stop_rotation()
            self.btn_rot.config(text="Rotate")
        else:
            try: angle = float(self.ent_rot.get())
            except ValueError: angle = 45.0
            self.viewer.start_rotation(angle)
            self.btn_rot.config(text="Stop Rot")
            
    def _update_viewer_slider_range(self):
        if self.viewer and hasattr(self, 'viewer_slider'):
            max_idx = max(0, self.viewer.N - 1)
            self.viewer_slider.config(to=max_idx)

    def _sync_viewer_ui(self):
        if self.viewer:
            # Sync slider while playing
            if self.viewer.is_playing and self.viewer_slider.get() != self.viewer.curr_idx:
                self.viewer_slider.set(self.viewer.curr_idx)
            # Reset Play button if it finished naturally
            if not self.viewer.is_playing and self.btn_play['text'] == "Pause":
                self.btn_play.config(text="Play")
            self.after(50, self._sync_viewer_ui)

    def _destroy_viewer(self):
        if self.viewer is not None:
            try:
                if hasattr(self.viewer, 'anim') and self.viewer.anim is not None:
                    self.viewer.anim.event_source.stop()
                if hasattr(self.viewer, 'rot_anim') and self.viewer.rot_anim is not None:
                    self.viewer.rot_anim.event_source.stop()
                
                if DEBUG_MODE:
                    print("[DEBUG] Destroying Grid/Planning 3D Viewer canvas")
                self.canvas_viewer.get_tk_widget().destroy()
                plt.close(self.viewer.fig)
            except Exception as e:
                print(f"Note: Could not cleanly destroy viewer: {e}")
            finally:
                if hasattr(self, 'viewer_ctrl_frame') and self.viewer_ctrl_frame.winfo_exists():
                    self.viewer_ctrl_frame.destroy()
                if hasattr(self, 'lbl_viewer_note') and self.lbl_viewer_note.winfo_exists():
                    self.lbl_viewer_note.destroy()
                if hasattr(self, 'viewer_canvas_frame') and self.viewer_canvas_frame.winfo_exists():
                    self.viewer_canvas_frame.destroy()
                self.viewer, self.canvas_viewer = None, None

    def _build_grid_ops_ui(self):
        self.grid_vars = {}
        
        # --- Scrollable Setup ---
        canvas = tk.Canvas(self.tab_grid_ops, highlightthickness=0)
        self.canvas = canvas
        scrollbar = ttk.Scrollbar(self.tab_grid_ops, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        main_container = ttk.Frame(canvas, padding="10")
        canvas_window = canvas.create_window((0, 0), window=main_container, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=(0, 0, bbox[2], max(bbox[3], event.height)))

        def on_frame_configure(event):
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=(0, 0, bbox[2], max(bbox[3], canvas.winfo_height())))

        canvas.bind("<Configure>", on_canvas_configure)
        main_container.bind("<Configure>", on_frame_configure)
        
        # Mouse wheel scrolling support
        def _on_mousewheel(event):
            # Prevent scrolling if all content is already visible
            if canvas.yview() == (0.0, 1.0):
                return
            if getattr(event, 'num', 0) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, 'num', 0) == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                
        canvas.bind("<Enter>", lambda e: [canvas.bind_all("<MouseWheel>", _on_mousewheel), canvas.bind_all("<Button-4>", _on_mousewheel), canvas.bind_all("<Button-5>", _on_mousewheel)])
        canvas.bind("<Leave>", lambda e: [canvas.unbind_all("<MouseWheel>"), canvas.unbind_all("<Button-4>"), canvas.unbind_all("<Button-5>")])
        # ------------------------

        # --- Geometry ---
        geom_frame = ttk.LabelFrame(main_container, text="Geometry", padding="10")
        geom_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        geom_left = ttk.Frame(geom_frame); geom_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, anchor=tk.N)
        geom_right = ttk.Frame(geom_frame); geom_right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, anchor=tk.N)
        
        self.grid_vars['cyl_radius'] = self._add_form_entry(geom_left, "Cylinder Radius (m):", "0.20", "Cylinder internal radius (m)")
        self.grid_vars['cyl_height'] = self._add_form_entry(geom_left, "Cylinder Height (m):", "0.50", "Cylinder internal height (m)")
        self.grid_vars['num_points'] = self._add_form_entry(geom_right, "Total Points:", "1000", "Total points for the generated grid (forward + reverse spirals combined)")
        self.grid_vars['azimuth_density_ratio'] = self._add_form_entry(geom_right, "Azimuth Density Ratio:", "1.0", "Front-to-back point density ratio. 1.0 = uniform. 5.0 = front is 5x denser than back.\nSmaller values (2.0 - 10.0) create a wide, smooth fade.\nLarge values (>20) create a narrow band.")

        # --- Keep Out ---
        ko_frame = ttk.LabelFrame(main_container, text="Keep Out", padding="10")
        ko_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ko_left = ttk.Frame(ko_frame); ko_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, anchor=tk.N)
        ko_right = ttk.Frame(ko_frame); ko_right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, anchor=tk.N)
        
        self.grid_vars['phi_min_deg'] = self._add_form_entry(ko_left, "Phi Min (deg):", "-170.0", "Phi cut limits (degrees, cylindrical azimuth)")
        self.grid_vars['phi_max_deg'] = self._add_form_entry(ko_right, "Phi Max (deg):", "170.0", "Phi cut limits (degrees, cylindrical azimuth)")
        self.grid_vars['bottom_cutoff_mm'] = self._add_form_entry(ko_left, "Bottom Cutoff (mm):", "30.0", "Remove bottom-cap points within this radius from center (mm) for support pole")

        # --- Path Plan ---
        plan_frame = ttk.LabelFrame(main_container, text="Path Plan", padding="10")
        plan_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.grid_vars['delta_theta_deg'] = self._add_form_entry(plan_frame, "Delta Theta (deg):", "7.5", "θ-bin width (deg). Smaller -> more bins, smoother θ progression.")

        # --- Advanced Options ---
        self.btn_advanced = ttk.Button(main_container, text="Show Advanced Settings ▼", command=self._toggle_advanced)
        self.btn_advanced.pack(side=tk.TOP, pady=10)

        self.adv_frame = ttk.LabelFrame(main_container, text="Advanced Settings", padding="10")
        
        adv_cols_frame = ttk.Frame(self.adv_frame)
        adv_cols_frame.pack(side=tk.TOP, fill=tk.X, expand=True)
        
        adv_left = ttk.Frame(adv_cols_frame); adv_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, anchor=tk.N)
        adv_right = ttk.Frame(adv_cols_frame); adv_right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, anchor=tk.N)
        
        self.grid_vars['wall_thickness_mm'] = self._add_form_entry(adv_left, "Wall Thickness (mm):", "50.0", "The maximum distance points can be pulled outwards (D_max)")
        self.grid_vars['cap_fraction'] = self._add_form_entry(adv_right, "Cap Fraction (0-1 or 'Auto'):", "Auto", "Fraction of points on both end-caps combined. None = Auto (end cap to side wall area based). 0-1 = Manually enter a fraction.")
        self.grid_vars['P_side'] = self._add_form_entry(adv_left, "P_side (Pull Power):", "0.5", "Magnetic pull bias on cylinder sides - >0.5")
        self.grid_vars['P_caps'] = self._add_form_entry(adv_right, "P_caps (Pull Power):", "0.8", "Power for magnetic pull on cylinder caps")
        self.grid_vars['cap_tol_mm'] = self._add_form_entry(adv_left, "Cap Tolerance (mm):", "Auto: wall_thickness_mm + 1mm", "Points within ±CAP_TOL_MM of min/max z are treated as caps (top/bottom).")
        self.grid_vars['azimuth_weight_center_deg'] = self._add_form_entry(adv_right, "Azimuth Weight Center (deg):", "0.0", "Angle (deg) for the center of the high-density zone.")
        self.grid_vars['z_rotation_deg'] = self._add_form_entry(adv_left, "Z Rotation 2nd Spiral (deg):", "90.0", "Rotate second spiral around Z (deg)")
        self.grid_vars['side_snake_start'] = self._add_combobox(adv_right, "Side Snake Start:", ["up", "down"], "up", "Initial z traversal direction for sidewall bins: 'up' or 'down'.")
        
        cb_frame_container = ttk.Frame(self.adv_frame)
        cb_frame_container.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        cb_frame1 = ttk.Frame(cb_frame_container); cb_frame1.pack(anchor=tk.W, pady=2)
        self.grid_vars['generate_reverse_spiral'] = self._add_checkbutton(cb_frame1, "Generate Reverse Spiral", True, "Make second (reverse) spiral")
        
        cb_frame2 = ttk.Frame(cb_frame_container); cb_frame2.pack(anchor=tk.W, pady=2)
        self.grid_vars['flip_poles'] = self._add_checkbutton(cb_frame2, "Flip Poles (2nd Spiral)", False, "Flip Z sign of second spiral")
        
        cb_frame3 = ttk.Frame(cb_frame_container); cb_frame3.pack(anchor=tk.W, pady=2)
        self.grid_vars['z_midpoint_zero'] = self._add_checkbutton(cb_frame3, "Z Midpoint = 0", True, "True = Z axis centred at 0 mm (equal negative and positive values).\nFalse = Z axis all positive like physical robot axis.")

        # --- Physical Waypoints ---
        ttk.Label(self.adv_frame, text="Physical Waypoints (Optional):", font=("Arial", 9, "bold")).pack(side=tk.TOP, anchor=tk.W, pady=(15, 5))
        
        wp_frame = ttk.Frame(self.adv_frame)
        wp_frame.pack(side=tk.TOP, fill=tk.X, padx=5)

        ttk.Label(wp_frame, text="Waypoint", font=("Arial", 8, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        ttk.Label(wp_frame, text="Radius (r_mm)", font=("Arial", 8, "bold")).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(wp_frame, text="Azimuth (phi_deg)", font=("Arial", 8, "bold")).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(wp_frame, text="Height (z_mm)", font=("Arial", 8, "bold")).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        def _add_wp_entry(row, col, default_val):
            var = tk.StringVar(value=default_val)
            entry = ttk.Entry(wp_frame, textvariable=var, width=12)
            entry.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            return var

        lbl_top = ttk.Label(wp_frame, text="Top Critical:")
        lbl_top.grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        ToolTip(lbl_top, "Overrides cylinder radius and height.")
        self.grid_vars['wp_top_r']   = _add_wp_entry(1, 1, "")
        self.grid_vars['wp_top_phi'] = _add_wp_entry(1, 2, "")
        self.grid_vars['wp_top_z']   = _add_wp_entry(1, 3, "")

        lbl_bot = ttk.Label(wp_frame, text="Bottom Critical:")
        lbl_bot.grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        ToolTip(lbl_bot, "Overrides bottom cutoff.")
        self.grid_vars['wp_bot_r']   = _add_wp_entry(2, 1, "")
        self.grid_vars['wp_bot_phi'] = _add_wp_entry(2, 2, "")
        self.grid_vars['wp_bot_z']   = _add_wp_entry(2, 3, "")

        lbl_tw = ttk.Label(wp_frame, text="Tweeter:")
        lbl_tw.grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        ToolTip(lbl_tw, "Optional metadata for downstream acoustic origin optimization.")
        self.grid_vars['wp_tw_r']    = _add_wp_entry(3, 1, "")
        self.grid_vars['wp_tw_phi']  = _add_wp_entry(3, 2, "")
        self.grid_vars['wp_tw_z']    = _add_wp_entry(3, 3, "")

        # --- Buttons ---
        self.btn_frame = ttk.Frame(main_container)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        ttk.Button(self.btn_frame, text="Generate & Plan Path", command=self._action_generate_and_plan).pack(side=tk.LEFT, padx=10)
        ttk.Button(self.btn_frame, text="Reset Replay", command=self._action_reset_replay).pack(side=tk.LEFT, padx=10)

    def _toggle_advanced(self):
        if self.adv_frame.winfo_ismapped():
            self.adv_frame.pack_forget()
            self.btn_advanced.config(text="Show Advanced Settings ▼")
            self.canvas.yview_moveto(0)
        else:
            self.adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_frame)
            self.btn_advanced.config(text="Hide Advanced Settings ▲")

    def _add_form_entry(self, parent, label_text, default_val, help_text=None, state_var=None):
        lbl_frame = ttk.Frame(parent)
        lbl_frame.pack(anchor=tk.W, fill=tk.X, pady=(5, 0))
        lbl = ttk.Label(lbl_frame, text=label_text)
        lbl.pack(side=tk.LEFT)
            
        var = tk.StringVar(value=default_val)
        entry = ttk.Entry(parent, textvariable=var)
        entry.pack(anchor=tk.W, fill=tk.X, pady=(0, 5))
        
        if help_text:
            ToolTip(lbl, help_text)
            ToolTip(entry, help_text)
            
        if state_var:
            def update_state(*args):
                state = tk.NORMAL if state_var.get() else tk.DISABLED
                entry.config(state=state)
                lbl.config(state=state)
            state_var.trace_add("write", update_state)
            update_state()
            
        return var

    def _add_labeled_entry(self, parent, label_text, default_val, width):
        frame = ttk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
        if isinstance(default_val, tk.StringVar):
            var = default_val
        else:
            var = tk.StringVar(value=default_val)
        entry = ttk.Entry(frame, textvariable=var, width=width)
        entry.pack(side=tk.LEFT)
        # Trigger update on focus out or enter
        entry.bind("<FocusOut>", self._schedule_update_stage5_preview)
        entry.bind("<Return>", self._schedule_update_stage5_preview)
        return var

    def _add_checkbutton(self, parent, text, default_val, help_text=None):
        var = tk.BooleanVar(value=default_val)
        cb = ttk.Checkbutton(parent, text=text, variable=var)
        cb.pack(side=tk.LEFT, padx=(0, 10))
        if help_text:
            ToolTip(cb, help_text)
        return var

    def _add_combobox(self, parent, label_text, values, default_val, help_text=None):
        lbl_frame = ttk.Frame(parent)
        lbl_frame.pack(anchor=tk.W, fill=tk.X, pady=(5, 0))
        lbl = ttk.Label(lbl_frame, text=label_text)
        lbl.pack(side=tk.LEFT)
        
        var = tk.StringVar(value=default_val)
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        cb.pack(anchor=tk.W, pady=(0, 5))
        
        if help_text:
            ToolTip(lbl, help_text)
            ToolTip(cb, help_text)
            
        return var

    def _browse_dir(self):
        directory = filedialog.askdirectory(initialdir=self.project_dir.get(), title="Select Project Directory")
        if directory:
            if DEBUG_MODE:
                print(f"[DEBUG] Browsed and selected project directory: {directory}")
                self._log_directory_tree(directory, "New Project Directory Listing")
            self.project_dir.set(directory)
            self._load_settings(directory)
            
    def _browse_stage5_output_dir(self):
        directory = filedialog.askdirectory(initialdir=self.project_dir.get(), title="Select Output Directory")
        if directory:
            if DEBUG_MODE:
                print(f"[DEBUG] Browsed and selected stage 5 output directory: {directory}")
            # Store relative path if possible
            proj_dir = self.project_dir.get()
            try:
                rel_path = os.path.relpath(directory, proj_dir)
                self.stage5_vars['output_dir'].set(rel_path)
            except ValueError:
                self.stage5_vars['output_dir'].set(directory)
                
    def _browse_mic_cal_file(self):
        filepath = filedialog.askopenfilename(initialdir=self.project_dir.get(), title="Select Mic Calibration File", filetypes=[("Text Files", "*.txt"), ("FRD Files", "*.frd"), ("All Files", "*.*")])
        if filepath:
            try:
                if DEBUG_MODE:
                    print(f"[DEBUG] Browsed and selected mic cal file: {filepath}")
                rel_path = os.path.relpath(filepath, self.project_dir.get())
                self.stage5_vars['mic_cal_file'].set(rel_path)
            except ValueError:
                self.stage5_vars['mic_cal_file'].set(filepath)

    def _action_save_project(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Save Project")
        self._save_settings()
        proj_name = self.project_name.get().strip() or "scanner"
        save_path = os.path.join(self.project_dir.get(), f"{proj_name}_project.json")
        self.status_var.set(f"Project Saved: {save_path}")

    def _save_project_if_not_exists(self):
        proj_name = self.project_name.get().strip() or "scanner"
        save_path = os.path.join(self.project_dir.get(), f"{proj_name}_project.json")
        if not os.path.exists(save_path):
            if DEBUG_MODE:
                print(f"[DEBUG] Auto-saving initial project settings to {save_path}")
            self._save_settings()

    def _save_settings(self):
        settings = {
            "project_name": self.project_name.get(),
            "grid_vars": {k: v.get() for k, v in self.grid_vars.items()},
            "stage1_vars": {k: v.get() for k, v in getattr(self, 'stage1_vars', {}).items()},
            "stage2_vars": {k: v.get() for k, v in getattr(self, 'stage2_vars', {}).items()},
            "stage3_vars": {k: v.get() for k, v in getattr(self, 'stage3_vars', {}).items()},
            "stage4_vars": {k: v.get() for k, v in getattr(self, 'stage4_vars', {}).items()},
            "stage5_vars": {k: v.get() for k, v in getattr(self, 'stage5_vars', {}).items()},
            "stage4_manual_table": {str(k): v for k, v in getattr(self, 'stage4_manual_table', {}).items()}
        }

        if DEBUG_MODE and hasattr(self, 'debug_log_file') and self.debug_log_file:
            self.debug_log_file.write(f"[DEBUG] Saving project settings:\n{json.dumps(settings, indent=4)}\n")
            self.debug_log_file.flush()

        proj_name = self.project_name.get().strip() or "scanner"
        save_path = os.path.join(self.project_dir.get(), f"{proj_name}_project.json")
        try:
            with open(save_path, "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save project settings: {e}")

    def _load_settings(self, directory):
        proj_name = self.project_name.get().strip() or "scanner"
        load_path = os.path.join(directory, f"{proj_name}_project.json")
        
        if not os.path.exists(load_path):
            try:
                json_files = [f for f in os.listdir(directory) if f.endswith("_project.json")]
                if json_files:
                    load_path = os.path.join(directory, json_files[0])
                    discovered_name = json_files[0].replace("_project.json", "")
                    self.project_name.set(discovered_name)
                    if DEBUG_MODE:
                        print(f"[DEBUG] Auto-discovered project name '{discovered_name}' from folder.")
            except OSError:
                pass

        if os.path.exists(load_path):
            if DEBUG_MODE:
                print(f"[DEBUG] Loading project settings from: {load_path}")
            try:
                with open(load_path, "r") as f:
                    settings = json.load(f)

                if DEBUG_MODE and hasattr(self, 'debug_log_file') and self.debug_log_file:
                    self.debug_log_file.write(f"[DEBUG] Loaded settings:\n{json.dumps(settings, indent=4)}\n")
                    self.debug_log_file.flush()

                if "project_name" in settings:
                    self.project_name.set(settings["project_name"])
                if "grid_vars" in settings:
                    for k, v in settings["grid_vars"].items():
                        if k in self.grid_vars:
                            if isinstance(self.grid_vars[k], tk.BooleanVar):
                                self.grid_vars[k].set(bool(v))
                            else:
                                self.grid_vars[k].set(str(v))
                if "stage1_vars" in settings:
                    for k, v in settings["stage1_vars"].items():
                        if k in getattr(self, 'stage1_vars', {}):
                            if isinstance(self.stage1_vars[k], tk.BooleanVar):
                                self.stage1_vars[k].set(bool(v))
                            else:
                                val = str(v)
                                if k == 'smoothing_oct_res' and val.strip() == '':
                                    val = 'Auto'
                                self.stage1_vars[k].set(val)
                if "stage2_vars" in settings:
                    for k, v in settings["stage2_vars"].items():
                        if k in getattr(self, 'stage2_vars', {}):
                            if isinstance(self.stage2_vars[k], tk.BooleanVar):
                                self.stage2_vars[k].set(bool(v))
                            else:
                                self.stage2_vars[k].set(str(v))
                if "stage3_vars" in settings:
                    for k, v in settings["stage3_vars"].items():
                        if k in getattr(self, 'stage3_vars', {}):
                            if isinstance(self.stage3_vars[k], tk.BooleanVar):
                                self.stage3_vars[k].set(bool(v))
                            else:
                                self.stage3_vars[k].set(str(v))
                if "stage4_vars" in settings:
                    for k, v in settings["stage4_vars"].items():
                        if k in getattr(self, 'stage4_vars', {}):
                            if isinstance(self.stage4_vars[k], tk.BooleanVar):
                                self.stage4_vars[k].set(bool(v))
                            else:
                                self.stage4_vars[k].set(str(v))
                if "stage5_vars" in settings:
                    for k, v in settings["stage5_vars"].items():
                        if k in getattr(self, 'stage5_vars', {}):
                            if isinstance(self.stage5_vars[k], tk.BooleanVar):
                                self.stage5_vars[k].set(bool(v))
                            else:
                                self.stage5_vars[k].set(str(v))
                    self.stage5_manual_coords = settings.get("stage5_vars", {}).get("manual_coord_list", [])
                if "stage4_manual_table" in settings:
                    self.stage4_manual_table = {float(k): int(v) for k, v in settings["stage4_manual_table"].items()}

            except Exception as e:
                print(f"Warning: Failed to load project settings: {e}")

    def _action_generate_and_plan(self):
        try:
            if DEBUG_MODE:
                print("[DEBUG] Action: Generate & Plan Path")
            self._save_project_if_not_exists()
            proj_name = self.project_name.get().strip() or "project"
            
            # Auto-fill empty numerical fields with "0" to prevent float conversion crashes
            fields_to_zero = [
                'cyl_radius', 'cyl_height', 'num_points', 'wall_thickness_mm',
                'bottom_cutoff_mm', 'P_side', 'P_caps', 'z_rotation_deg',
                'phi_min_deg', 'phi_max_deg', 'azimuth_density_ratio',
                'azimuth_weight_center_deg', 'delta_theta_deg'
            ]
            for key in fields_to_zero:
                if key in self.grid_vars and not self.grid_vars[key].get().strip():
                    self.grid_vars[key].set("0")

            # Parse variables safely
            cap_str = self.grid_vars['cap_fraction'].get().strip()
            cap_frac = None if cap_str.lower() in ("auto", "none", "") else float(cap_str)
            
            def get_wp_tuple(prefix):
                r_str = self.grid_vars.get(f'wp_{prefix}_r', tk.StringVar(value="")).get().strip()
                phi_str = self.grid_vars.get(f'wp_{prefix}_phi', tk.StringVar(value="")).get().strip()
                z_str = self.grid_vars.get(f'wp_{prefix}_z', tk.StringVar(value="")).get().strip()
                if not r_str and not phi_str and not z_str:
                    return None
                
                r_str = r_str if r_str else "0.0"
                phi_str = phi_str if phi_str else "0.0"
                z_str = z_str if z_str else "0.0"
                
                try:
                    return (float(r_str), float(phi_str), float(z_str))
                except ValueError:
                    print(f"Warning: Incomplete or invalid {prefix} waypoint provided. Ignoring.")
                    return None

            top_pos = get_wp_tuple('top')
            bot_pos = get_wp_tuple('bot')
            tw_pos = get_wp_tuple('tw')

            gen_params = {
                'cyl_radius': float(self.grid_vars['cyl_radius'].get()),
                'cyl_height': float(self.grid_vars['cyl_height'].get()),
                'num_points': int(self.grid_vars['num_points'].get()),
                'wall_thickness_mm': float(self.grid_vars['wall_thickness_mm'].get()),
                'bottom_cutoff_mm': float(self.grid_vars['bottom_cutoff_mm'].get()),
                'cap_fraction': cap_frac,
                'P_side': float(self.grid_vars['P_side'].get()),
                'P_caps': float(self.grid_vars['P_caps'].get()),
                'generate_reverse_spiral': self.grid_vars['generate_reverse_spiral'].get(),
                'z_rotation_deg': float(self.grid_vars['z_rotation_deg'].get()),
                'flip_poles': self.grid_vars['flip_poles'].get(),
                'z_midpoint_zero': self.grid_vars['z_midpoint_zero'].get(),
                'phi_min_deg': float(self.grid_vars['phi_min_deg'].get()),
                'phi_max_deg': float(self.grid_vars['phi_max_deg'].get()),
                'azimuth_density_ratio': float(self.grid_vars['azimuth_density_ratio'].get()),
                'azimuth_weight_center_deg': float(self.grid_vars['azimuth_weight_center_deg'].get()),
                'top_crit_pos': top_pos,
                'bot_crit_pos': bot_pos,
                'tweeter_pos': tw_pos
            }

            self.grid_data = generate_measurement_grid(**gen_params)

            # Path Plan immediately
            cap_tol_str = self.grid_vars['cap_tol_mm'].get().strip()
            if cap_tol_str.lower().startswith('auto') or cap_tol_str == '':
                cap_tol_mm = float(self.grid_vars['wall_thickness_mm'].get()) + 1.0
            else:
                cap_tol_mm = float(cap_tol_str)

            plan_params = {
                'input_data': self.grid_data,
                'cap_tol_mm': cap_tol_mm,
                'delta_theta_deg': float(self.grid_vars['delta_theta_deg'].get()),
                'side_snake_start': self.grid_vars['side_snake_start'].get(),
                'output_path': os.path.join(self.project_dir.get(), f"{proj_name}_scan_path.csv"),
                'show_replay': False
            }

            self.planned_data = plan_path(**plan_params)
            
            # Live update the viewer
            if self.planned_data is not None:
                self.viewer.load_data(self.planned_data)
            else:
                self.viewer.load_data(plan_params['output_path'])
            self._update_viewer_slider_range()

            self.status_var.set(f"Success: Grid generated and path planned successfully. Saved to: {plan_params['output_path']}")

        except Exception as e:
            self.status_var.set(f"Error: Failed to generate and plan path: {str(e)}")

    def _action_reset_replay(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Reset Replay")
        if self.viewer and self.viewer.N > 0:
            self.viewer.rewind()
        else:
            # Try loading from disk
            proj_name = self.project_name.get().strip() or "project"
            candidate_path = os.path.join(self.project_dir.get(), f"{proj_name}_scan_path.csv")
            if os.path.exists(candidate_path):
                try:
                    if self.viewer:
                        self.viewer.load_data(candidate_path)
                        self._update_viewer_slider_range()
                    self.status_var.set(f"Loaded planned path from: {candidate_path}")
                except Exception as e:
                    self.status_var.set(f"Error: Failed to load planned path from file: {str(e)}")
            else:
                self.status_var.set("Warning: No Data. Please generate and plan the path first.")

    def _set_widget_state(self, widget, state):
        try:
            if isinstance(widget, ttk.Combobox):
                widget.config(state=tk.DISABLED if state == tk.DISABLED else "readonly")
            else:
                widget.config(state=state)
        except tk.TclError:
            pass # Some widgets like Frames don't have a state
        for child in widget.winfo_children():
            self._set_widget_state(child, state)
            
    def _build_stage1_ui(self):
        # --- Scrollable Setup ---
        canvas = tk.Canvas(self.tab_stage1, highlightthickness=0)
        self.stage1_canvas = canvas
        scrollbar = ttk.Scrollbar(self.tab_stage1, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        main_container = ttk.Frame(canvas, padding="10")
        canvas_window = canvas.create_window((0, 0), window=main_container, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=(0, 0, bbox[2], max(bbox[3], event.height)))

        def on_frame_configure(event):
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=(0, 0, bbox[2], max(bbox[3], canvas.winfo_height())))

        canvas.bind("<Configure>", on_canvas_configure)
        main_container.bind("<Configure>", on_frame_configure)
        
        def _on_mousewheel(event):
            if canvas.yview() == (0.0, 1.0): return
            if getattr(event, 'num', 0) == 4: canvas.yview_scroll(-1, "units")
            elif getattr(event, 'num', 0) == 5: canvas.yview_scroll(1, "units")
            else: canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                
        canvas.bind("<Enter>", lambda e: [canvas.bind_all("<MouseWheel>", _on_mousewheel), canvas.bind_all("<Button-4>", _on_mousewheel), canvas.bind_all("<Button-5>", _on_mousewheel)])
        canvas.bind("<Leave>", lambda e: [canvas.unbind_all("<MouseWheel>"), canvas.unbind_all("<Button-4>"), canvas.unbind_all("<Button-5>")])
        
        self.stage1_vars = {}

        # --- Main Settings ---
        main_settings_frame = ttk.LabelFrame(main_container, text="Main Settings", padding="10")
        main_settings_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.stage1_vars['fdw_rft_ms'] = self._add_form_entry(main_settings_frame, "Reflection Free Time (ms):", "5.0", "Reflection Free Time (ms): Defines the fixed window length at high frequencies.")
        self.stage1_vars['fdw_oct_res'] = self._add_form_entry(main_settings_frame, "Octave Resolution (1/x):", "12", "Target Octave Resolution: Sets the fractional octave smoothing (e.g., 12 for 1/12th oct).")
        self.stage1_vars['fdw_max_cap_ms'] = self._add_form_entry(main_settings_frame, "Max Window Cap (ms):", "200.0", "Optional cap (ms) on the maximum window length. Will limit oct res at LF.")
        self.stage1_vars['enable_auto_gain'] = self._add_checkbutton(main_settings_frame, "Enable Auto Gain", False, "Enable global normalization across all files in the batch.")
        self.stage1_vars['target_peak_db'] = self._add_form_entry(main_settings_frame, "Target Peak (dB):", "-3.0", "Target peak level (dBFS) for the loudest file in the set.", state_var=self.stage1_vars['enable_auto_gain'])
        
        # --- Advanced Settings ---
        self.btn_stage1_advanced = ttk.Button(main_container, text="Show Advanced Settings ▼", command=self._toggle_stage1_advanced)
        self.btn_stage1_advanced.pack(side=tk.TOP, pady=10)

        self.stage1_adv_frame = ttk.LabelFrame(main_container, text="Advanced Settings", padding="10")
        
        self.stage1_vars['smoothing_oct_res'] = self._add_form_entry(self.stage1_adv_frame, "Smoothing Octave Res (1/x):", "Auto", "Auto = octave resolution x2 so that smoothing does not reduce resolution of initial FDW.")
        self.stage1_vars['fdw_alpha_hf'] = self._add_form_entry(self.stage1_adv_frame, "Alpha HF:", "0.2", "Taper alpha for High Frequencies (0.0=Rectangular, 1.0=Hann).")
        self.stage1_vars['fdw_alpha_lf'] = self._add_form_entry(self.stage1_adv_frame, "Alpha LF:", "1.0", "Taper alpha for Low Frequencies.")
        self.stage1_vars['fdw_windows_per_oct'] = self._add_form_entry(self.stage1_adv_frame, "Windows per Octave:", "3", "Windows per octave. Interpolation is performed in complex domain between windows.")
        self.stage1_vars['peak_detect_threshold_db'] = self._add_form_entry(self.stage1_adv_frame, "Peak Detect Threshold (dB):", "-12.0", "Peak detection finds loudest peak, then searches for earlier peaks above this threshold. A reflection may be louder than the true direct sound peak.")
        self.stage1_vars['fdw_f_min'] = self._add_form_entry(self.stage1_adv_frame, "Min Frequency (Hz):", "20.0", "Minimum frequency (Hz) for the X axis in the plot view of the FDW analysis. Visual only.")
        self.stage1_vars['enable_smoothing'] = self._add_checkbutton(self.stage1_adv_frame, "Enable Smoothing", True, "Enable or disable complex smoothing.")
        self.stage1_vars['keep_raw_and_smoothed'] = self._add_checkbutton(self.stage1_adv_frame, "Keep Raw & Smoothed", False, "Save both files if True.")
        self.stage1_vars['show_plot'] = self._add_checkbutton(self.stage1_adv_frame, "Show Plot on Completion", True, "If True, launches the interactive data viewer after processing.")

        # --- Button ---
        self.btn_stage1_run = ttk.Button(main_container, text="Run Stage 1", command=self._action_run_stage1)
        self.btn_stage1_run.pack(side=tk.TOP, pady=20)
        

    def _toggle_stage1_advanced(self):
        if self.stage1_adv_frame.winfo_ismapped():
            self.stage1_adv_frame.pack_forget()
            self.btn_stage1_advanced.config(text="Show Advanced Settings ▼")
            self.stage1_canvas.yview_moveto(0)
        else:
            self.stage1_adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_stage1_run)
            self.btn_stage1_advanced.config(text="Hide Advanced Settings ▲")
            
    def _update_cli(self):
        # Amalgamate all messages in the queue to process in a single batch
        all_strings = []
        try:
            while True:
                all_strings.append(self.cli_queue.get_nowait())
        except queue.Empty:
            pass

        if all_strings:
            full_string = "".join(all_strings)
            import re
            full_string = re.sub(r'\x1b\[[0-9;]*[mK]', '', full_string)

            # Check if we are currently within ~3 lines of the bottom of the scroll
            total_lines = int(self.cli_text.index("end-1c").split('.')[0])
            lines_from_bottom = (1.0 - self.cli_text.yview()[1]) * total_lines
            at_bottom = lines_from_bottom <= 3.0

            self.cli_text.config(state=tk.NORMAL)
            
            # Efficiently handle carriage returns and newlines
            lines = full_string.split('\n')
            for i, line in enumerate(lines):
                parts = line.split('\r')
                if parts:
                    self.cli_text.insert(tk.END, parts[0])
                    for part in parts[1:]:
                        self.cli_text.delete("end-1c linestart", "end-1c lineend")
                        self.cli_text.insert(tk.END, part)
                if i < len(lines) - 1:
                    self.cli_text.insert(tk.END, '\n')
            
            if at_bottom:
                self.cli_text.see(tk.END)
                
            self.cli_text.config(state=tk.DISABLED)

        self.after(20, self._update_cli)

    def _action_run_stage1(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Run Stage 1")
        self._save_project_if_not_exists()
        self.btn_stage1_run.config(state=tk.DISABLED)
        self.cli_text.config(state=tk.NORMAL)
        self.cli_text.insert(tk.END, "\n--- Starting Stage 1 ---\n")
        self.cli_text.config(state=tk.DISABLED)
        
        threading.Thread(target=self._run_stage1_thread, daemon=True).start()

    def _run_stage1_thread(self):
        try:
            proj_dir = self.project_dir.get()
            input_dir = os.path.join(proj_dir, "Recordings")
            out_dir = os.path.join(proj_dir, "outputs")
            output_filename = f"{self.project_name.get()}_complex_data.npz"
            
            rft_ms = float(self.stage1_vars['fdw_rft_ms'].get())
            oct_res = float(self.stage1_vars['fdw_oct_res'].get())
            max_cap_ms = float(self.stage1_vars['fdw_max_cap_ms'].get())
            auto_gain = self.stage1_vars['enable_auto_gain'].get()
            target_peak = float(self.stage1_vars['target_peak_db'].get())
            show_plot_val = self.stage1_vars['show_plot'].get()
            
            en_smooth = self.stage1_vars['enable_smoothing'].get()
            smooth_res_str = self.stage1_vars['smoothing_oct_res'].get().strip().lower()
            if en_smooth:
                if smooth_res_str == 'auto' or not smooth_res_str:
                    smooth_res = oct_res * 2
                else:
                    try:
                        smooth_res = float(smooth_res_str)
                    except ValueError:
                        print("Warning: Invalid smoothing resolution entered, falling back to Auto (Octave Res x2).")
                        smooth_res = oct_res * 2
            else:
                smooth_res = oct_res * 2 # Fallback, not used if smoothing is disabled
            keep_raw = self.stage1_vars['keep_raw_and_smoothed'].get()
            alpha_hf = float(self.stage1_vars['fdw_alpha_hf'].get())
            alpha_lf = float(self.stage1_vars['fdw_alpha_lf'].get())
            win_per_oct = int(self.stage1_vars['fdw_windows_per_oct'].get())
            peak_thresh = float(self.stage1_vars['peak_detect_threshold_db'].get())
            f_min = float(self.stage1_vars['fdw_f_min'].get())
            
            results = fdwsmooth(
                input_dir=input_dir,
                out_dir=out_dir,
                output_filename=output_filename,
                fdw_rft_ms=rft_ms,
                fdw_oct_res=oct_res,
                fdw_max_cap_ms=max_cap_ms,
                enable_smoothing=en_smooth,
                smoothing_oct_res=smooth_res,
                show_plot=False,  # Prevent plotting in the background thread
                save_to_disk=True,
                fdw_alpha_hf=alpha_hf,
                fdw_alpha_lf=alpha_lf,
                fdw_f_min=f_min,
                fdw_windows_per_oct=win_per_oct,
                peak_detect_threshold_db=peak_thresh,
                enable_auto_gain=auto_gain,
                target_peak_db=target_peak,
                keep_raw_and_smoothed=keep_raw
            )
            
            if results and show_plot_val:
                freqs, results_raw, results_smooth, meta = results
                n_fft = (len(freqs) - 1) * 2
                df_val = freqs[1] - freqs[0]
                fs_common = int(round(df_val * n_fft))
                crop_samples = n_fft
                
                if not en_smooth:
                    plot_data = results_raw
                    plot_smooth = None
                else:
                    if keep_raw:
                        plot_data = results_raw
                        plot_smooth = results_smooth
                    else:
                        plot_data = results_smooth 
                        plot_smooth = None 
                
                def launch_viewer():
                    from viewers import FDWViewer
                    self.stage1_ui_instance = FDWViewer(freqs, plot_data, meta, fs_common, input_dir, crop_samples, data_dict_smooth=plot_smooth, fdw_f_min=f_min, fdw_rft_ms=rft_ms)
                
                print("Opening Viewer...")
                self.after(0, launch_viewer)

            print("Stage 1 completed successfully.")
        except Exception as e:
            print(f"Error during Stage 1: {e}")
        finally:
            self.after(0, lambda: self.btn_stage1_run.config(state=tk.NORMAL))

    def _build_stage2_ui(self):
        canvas = tk.Canvas(self.tab_stage2, highlightthickness=0)
        self.stage2_canvas = canvas
        scrollbar = ttk.Scrollbar(self.tab_stage2, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        main_container = ttk.Frame(canvas, padding="10")
        canvas_window = canvas.create_window((0, 0), window=main_container, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=(0, 0, bbox[2], max(bbox[3], event.height)))

        def on_frame_configure(event):
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=(0, 0, bbox[2], max(bbox[3], canvas.winfo_height())))

        canvas.bind("<Configure>", on_canvas_configure)
        main_container.bind("<Configure>", on_frame_configure)
        
        def _on_mousewheel(event):
            if canvas.yview() == (0.0, 1.0): return
            if getattr(event, 'num', 0) == 4: canvas.yview_scroll(-1, "units")
            elif getattr(event, 'num', 0) == 5: canvas.yview_scroll(1, "units")
            else: canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                
        canvas.bind("<Enter>", lambda e: [canvas.bind_all("<MouseWheel>", _on_mousewheel), canvas.bind_all("<Button-4>", _on_mousewheel), canvas.bind_all("<Button-5>", _on_mousewheel)])
        canvas.bind("<Leave>", lambda e: [canvas.unbind_all("<MouseWheel>"), canvas.unbind_all("<Button-4>"), canvas.unbind_all("<Button-5>")])
        
        self.stage2_vars = {}

        # --- Main Settings ---
        main_settings_frame = ttk.LabelFrame(main_container, text="Main Settings", padding="10")
        main_settings_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.stage2_vars['octave_resolution'] = self._add_form_entry(main_settings_frame, "Octave Resolution (1/x):", "6", "Octave frequency step resolution.")
        
        ttk.Label(main_settings_frame, text="User Defined Tweeter Coordinates:", font=("Arial", 9, "bold")).pack(side=tk.TOP, anchor=tk.W, pady=(10, 5))
        
        tweeter_frame = ttk.Frame(main_settings_frame)
        tweeter_frame.pack(side=tk.TOP, fill=tk.X)
        tw_x_frame = ttk.Frame(tweeter_frame); tw_x_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tw_y_frame = ttk.Frame(tweeter_frame); tw_y_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tw_z_frame = ttk.Frame(tweeter_frame); tw_z_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        self.stage2_vars['tweeter_x'] = self._add_form_entry(tw_x_frame, "Tweeter X (mm):", "0.0", "Seed coordinate for search (Depth).")
        self.stage2_vars['tweeter_y'] = self._add_form_entry(tw_y_frame, "Tweeter Y (mm):", "0.0", "Seed coordinate for search (Width).")
        self.stage2_vars['tweeter_z'] = self._add_form_entry(tw_z_frame, "Tweeter Z (mm):", "0.0", "Seed coordinate for search (Height).")
        
        # --- Advanced Settings ---
        self.btn_stage2_advanced = ttk.Button(main_container, text="Show Advanced Settings ▼", command=self._toggle_stage2_advanced)
        self.btn_stage2_advanced.pack(side=tk.TOP, pady=10)

        self.stage2_adv_frame = ttk.LabelFrame(main_container, text="Advanced Settings", padding="10")
        
        freq_frame = ttk.Frame(self.stage2_adv_frame)
        freq_frame.pack(side=tk.TOP, fill=tk.X)
        f_start_frame = ttk.Frame(freq_frame); f_start_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        f_end_frame = ttk.Frame(freq_frame); f_end_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        self.stage2_vars['freq_start_hz'] = self._add_form_entry(f_start_frame, "Start Frequency (Hz):", "20.0", "Lower boundary of the acoustic origin search.")
        self.stage2_vars['freq_end_hz'] = self._add_form_entry(f_end_frame, "End Frequency (Hz):", "20000.0", "Upper boundary of the acoustic origin search.")
        
        # --- Search Algorithm ---
        ttk.Label(self.stage2_adv_frame, text="Search Algorithm:", font=("Arial", 9, "bold")).pack(side=tk.TOP, anchor=tk.W, pady=(10, 5))
        self.stage2_vars['target_n_max_origins'] = self._add_form_entry(self.stage2_adv_frame, "Max Harmonic Order (N):", "4", "Max cap on the spherical harmonic order.")
        self.stage2_vars['initial_simplex_step'] = self._add_form_entry(self.stage2_adv_frame, "Initial Simplex Step (mm):", "15.0", "Size of the initial Nelder-Mead simplex.")
        self.stage2_vars['max_iterations'] = self._add_form_entry(self.stage2_adv_frame, "Max Iterations:", "50", "Maximum amoeba optimization steps per frequency.")
        
        

        # --- Full Grid Scan ---
        ttk.Label(self.stage2_adv_frame, text="Full Grid Scan:", font=("Arial", 9, "bold")).pack(side=tk.TOP, anchor=tk.W, pady=(10, 5))
        
        bounds_frame = ttk.Frame(self.stage2_adv_frame)
        bounds_frame.pack(side=tk.TOP, fill=tk.X)
        x_b_frame = ttk.Frame(bounds_frame); x_b_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        y_b_frame = ttk.Frame(bounds_frame); y_b_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        z_b_frame = ttk.Frame(bounds_frame); z_b_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        self.stage2_vars['x_bounds'] = self._add_form_entry(x_b_frame, "X Bounds:", "-500.0, 500.0", "Search area constraint for depth.")
        self.stage2_vars['y_bounds'] = self._add_form_entry(y_b_frame, "Y Bounds:", "-500.0, 500.0", "Search area constraint for width.")
        self.stage2_vars['z_bounds'] = self._add_form_entry(z_b_frame, "Z Bounds:", "-500.0, 500.0", "Search area constraint for height.")
        
        self.stage2_vars['grid_res_mm'] = self._add_form_entry(self.stage2_adv_frame, "Grid Res (mm):", "15.0", "Resolution of the physical grid slices.")
        
        # --- Show Validation Plot Checkbox ---
        self.stage2_vars['plot_results_origins'] = self._add_checkbutton(self.stage2_adv_frame, "Show Validation Plot", True, "Launch the interactive Validation UI after search completes.")

        # --- Button ---
        self.btn_stage2_run = ttk.Button(main_container, text="Run Stage 2", command=self._action_run_stage2)
        self.btn_stage2_run.pack(side=tk.TOP, pady=20)
        
    def _toggle_stage2_advanced(self):
        if self.stage2_adv_frame.winfo_ismapped():
            self.stage2_adv_frame.pack_forget()
            self.btn_stage2_advanced.config(text="Show Advanced Settings ▼")
            self.stage2_canvas.yview_moveto(0)
        else:
            self.stage2_adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_stage2_run)
            self.btn_stage2_advanced.config(text="Hide Advanced Settings ▲")

    def _action_run_stage2(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Run Stage 2")
        self._save_project_if_not_exists()
        self.btn_stage2_run.config(state=tk.DISABLED)
        self.cli_text.config(state=tk.NORMAL)
        self.cli_text.insert(tk.END, "\n--- Starting Stage 2 ---\n")
        self.cli_text.config(state=tk.DISABLED)
        threading.Thread(target=self._run_stage2_thread, daemon=True).start()

    def _run_stage2_thread(self):
        try:
            proj_dir = self.project_dir.get()
            input_dir = os.path.join(proj_dir, "outputs")
            input_filename = f"{self.project_name.get()}_complex_data.npz"
            output_filename = f"{self.project_name.get()}_complex_data.npz"
            
            t_x = float(self.stage2_vars['tweeter_x'].get())
            t_y = float(self.stage2_vars['tweeter_y'].get())
            t_z = float(self.stage2_vars['tweeter_z'].get())
            tweeter_coords = (t_x, t_y, t_z)
            
            oct_res_val = 1.0 / float(self.stage2_vars['octave_resolution'].get())
            f_start = float(self.stage2_vars['freq_start_hz'].get())
            f_end = float(self.stage2_vars['freq_end_hz'].get())
            
            def parse_bounds(b_str):
                parts = b_str.split(',')
                return (float(parts[0]), float(parts[1]))
            
            x_b = parse_bounds(self.stage2_vars['x_bounds'].get())
            y_b = parse_bounds(self.stage2_vars['y_bounds'].get())
            z_b = parse_bounds(self.stage2_vars['z_bounds'].get())
            
            grid_res = float(self.stage2_vars['grid_res_mm'].get())
            n_max = int(self.stage2_vars['target_n_max_origins'].get())
            sim_step = float(self.stage2_vars['initial_simplex_step'].get())
            max_iter = int(self.stage2_vars['max_iterations'].get())
            plot_results = self.stage2_vars['plot_results_origins'].get()
            
            from stage2_centre_origin import run_origin_search, export_interpolated_origins
            
            res = run_origin_search(
                input_dir_origins=input_dir,
                input_filename_origins=input_filename,
                output_filename_origins=output_filename,
                tweeter_coords_mm=tweeter_coords,
                octave_resolution=oct_res_val,
                freq_start_hz=f_start,
                freq_end_hz=f_end,
                initial_simplex_step=sim_step,
                max_iterations=max_iter,
                x_bounds=x_b,
                y_bounds=y_b,
                z_bounds=z_b,
                grid_res_mm=grid_res,
                target_n_max_origins=n_max,
                manual_order_table=None,
                save_to_disk=True,
                plot_results_origins=False,  # Prevent blocking Tkinter in thread
                return_state=True
            )
            
            if res is not None:
                sweep_results, f_all, keys, d_dict, geom, cfg, data = res
                
                def do_save():
                    history_freq, history_search_x, history_search_y, history_search_z = [], [], [], []
                    for f_hz in sorted(sweep_results.keys()):
                        d = sweep_results[f_hz]
                        if d['final_c'] is not None:
                            history_freq.append(f_hz)
                            history_search_x.append(d['final_c'][0])
                            history_search_y.append(d['final_c'][1])
                            history_search_z.append(d['final_c'][2])
                    
                    export_interpolated_origins(history_freq, history_search_x, history_search_y, history_search_z, f_all, data, input_dir, output_filename, True)
                    print("Stage 2 completed successfully.")

                def finish_stage2():
                    do_save()
                    if plot_results:
                        print("\nOpening Validation UI...")
                        from viewers import ValidationUI
                        import matplotlib.pyplot as plt
                        self.stage2_ui_instance = ValidationUI(sweep_results, f_all, keys, d_dict, geom, cfg)
                        
                        def wait_for_ui():
                            if plt.fignum_exists(self.stage2_ui_instance.view.fig.number):
                                self.after(200, wait_for_ui)
                            else:
                                if self.stage2_ui_instance.accepted:
                                    print("Saving adjusted results...")
                                    do_save()
                                else:
                                    print("Validation UI closed.")
                        wait_for_ui()
                
                self.after(0, finish_stage2)
            
        except Exception as e:
            print(f"Error during Stage 2: {e}")
        finally:
            self.after(0, lambda: self.btn_stage2_run.config(state=tk.NORMAL))

    def _build_stage3_ui(self):
        canvas = tk.Canvas(self.tab_stage3, highlightthickness=0)
        self.stage3_canvas = canvas
        scrollbar = ttk.Scrollbar(self.tab_stage3, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        main_container = ttk.Frame(canvas, padding="10")
        canvas_window = canvas.create_window((0, 0), window=main_container, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=(0, 0, bbox[2], max(bbox[3], event.height)))

        def on_frame_configure(event):
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=(0, 0, bbox[2], max(bbox[3], canvas.winfo_height())))

        canvas.bind("<Configure>", on_canvas_configure)
        main_container.bind("<Configure>", on_frame_configure)
        
        def _on_mousewheel(event):
            if canvas.yview() == (0.0, 1.0): return
            if getattr(event, 'num', 0) == 4: canvas.yview_scroll(-1, "units")
            elif getattr(event, 'num', 0) == 5: canvas.yview_scroll(1, "units")
            else: canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                
        canvas.bind("<Enter>", lambda e: [canvas.bind_all("<MouseWheel>", _on_mousewheel), canvas.bind_all("<Button-4>", _on_mousewheel), canvas.bind_all("<Button-5>", _on_mousewheel)])
        canvas.bind("<Leave>", lambda e: [canvas.unbind_all("<MouseWheel>"), canvas.unbind_all("<Button-4>"), canvas.unbind_all("<Button-5>")])
        
        self.stage3_vars = {}

        # --- Advanced Settings ---
        self.btn_stage3_advanced = ttk.Button(main_container, text="Show Advanced Settings ▼", command=self._toggle_stage3_advanced)
        self.btn_stage3_advanced.pack(side=tk.TOP, pady=10)

        self.stage3_adv_frame = ttk.LabelFrame(main_container, text="Advanced Settings", padding="10")
        
        self.stage3_vars['test_order_range'] = self._add_form_entry(self.stage3_adv_frame, "Test Order Range (min, max):", "4, 15", "Range of orders N to test.")
        self.stage3_vars['test_start_db_range'] = self._add_form_entry(self.stage3_adv_frame, "Test Start dB Range (highest, lowest):", "-20.0, -60.0", "Highest and lowest dB start point to test.")
        self.stage3_vars['test_lambda_range'] = self._add_form_entry(self.stage3_adv_frame, "Test Lambda Range (min, max):", "0.0000001, 0.01", "Range of max lambdas to test.")
        self.stage3_vars['test_db_transition_span'] = self._add_form_entry(self.stage3_adv_frame, "Test dB Transition Span:", "20.0", "The dB range between start of damping and maximum damping.")

        # --- Button ---
        self.btn_stage3_run = ttk.Button(main_container, text="Run Stage 3", command=self._action_run_stage3)
        self.btn_stage3_run.pack(side=tk.TOP, pady=20)

    def _toggle_stage3_advanced(self):
        if self.stage3_adv_frame.winfo_ismapped():
            self.stage3_adv_frame.pack_forget()
            self.btn_stage3_advanced.config(text="Show Advanced Settings ▼")
            self.stage3_canvas.yview_moveto(0)
        else:
            self.stage3_adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_stage3_run)
            self.btn_stage3_advanced.config(text="Hide Advanced Settings ▲")

    def _action_run_stage3(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Run Stage 3")
        self._save_project_if_not_exists()
        self.btn_stage3_run.config(state=tk.DISABLED)
        self.cli_text.config(state=tk.NORMAL)
        self.cli_text.insert(tk.END, "\n--- Starting Stage 3 ---\n")
        self.cli_text.config(state=tk.DISABLED)
        threading.Thread(target=self._run_stage3_thread, daemon=True).start()

    def _run_stage3_thread(self):
        try:
            proj_dir = self.project_dir.get()
            input_dir = os.path.join(proj_dir, "outputs")
            # Attempt to use centered data (from stage 2) first, else fallback to raw data
            input_filename = f"{self.project_name.get()}_complex_data_centered.npz"
            if not os.path.exists(os.path.join(input_dir, input_filename)):
                input_filename = f"{self.project_name.get()}_complex_data.npz"
            
            def parse_bounds(b_str):
                parts = b_str.split(',')
                return (float(parts[0]), float(parts[1]))
            
            def parse_bounds_int(b_str):
                parts = b_str.split(',')
                return (int(parts[0]), int(parts[1]))

            order_range = parse_bounds_int(self.stage3_vars['test_order_range'].get())
            start_db_range = parse_bounds(self.stage3_vars['test_start_db_range'].get())
            lambda_range = parse_bounds(self.stage3_vars['test_lambda_range'].get())
            transition_span = float(self.stage3_vars['test_db_transition_span'].get())

            from stage3_optimize_she_settings import run_open_branch_optimizer
            
            target_n_max, noise_floor_start_db, noise_floor_max_db, max_lambda = run_open_branch_optimizer(
                input_dir_opti=input_dir,
                input_filename_opti=input_filename,
                test_order_range=order_range,
                test_start_db_range=start_db_range,
                test_lambda_range=lambda_range,
                test_db_transition_span=transition_span,
                use_optimized_origins=True,
                speed_of_sound=343.0,
                kr_offset=2.0
            )

            # Pass suggested values seamlessly into stage 4 forms
            def update_stage4():
                if 'target_n_max' in self.stage4_vars:
                    self.stage4_vars['target_n_max'].set(str(target_n_max))
                if 'noise_floor_start_db' in self.stage4_vars:
                    self.stage4_vars['noise_floor_start_db'].set(str(noise_floor_start_db))
                if 'noise_floor_max_db' in self.stage4_vars:
                    self.stage4_vars['noise_floor_max_db'].set(str(noise_floor_max_db))
                if 'max_lambda' in self.stage4_vars:
                    self.stage4_vars['max_lambda'].set(f"{max_lambda:.8f}")

            self.after(0, update_stage4)
            print("Stage 3 completed successfully. Suggested max order and lambda settings updated in Stage 4.")
        except Exception as e:
            print(f"Error during Stage 3: {e}")
        finally:
            self.after(0, lambda: self.btn_stage3_run.config(state=tk.NORMAL))

    def _build_stage4_ui(self):
        scroll_frame = ScrollableFrame(self.tab_stage4)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        self.stage4_canvas = scroll_frame.canvas
        main_container = scroll_frame.scrollable_frame
        
        self.stage4_vars = {}
        if not hasattr(self, 'stage4_manual_table'):
            self.stage4_manual_table = {
                350.0: 3, 500.0: 4, 625.0: 5, 750.0: 6, 875.0: 7, 1500.0: 8,
                2100.0: 9, 2800.0: 10, 3500.0: 11, 4100.0: 12, 4600.0: 13, 5000.0: 14, 6000.0: 15
            }

        # --- Main Settings ---
        main_settings_frame = ttk.LabelFrame(main_container, text="Main Settings", padding="10")
        main_settings_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.stage4_vars['target_n_max'] = self._add_form_entry(main_settings_frame, "Target N Max:", "8", "Hard upper cap on harmonic order N.")
        
        # --- Advanced Settings ---
        self.btn_stage4_advanced = ttk.Button(main_container, text="Show Advanced Settings ▼", command=self._toggle_stage4_advanced)
        self.btn_stage4_advanced.pack(side=tk.TOP, pady=10)

        self.stage4_adv_frame = ttk.LabelFrame(main_container, text="Advanced Settings", padding="10")
        
        ttk.Label(self.stage4_adv_frame, text="Order_N Growth", font=("Arial", 9, "bold")).pack(side=tk.TOP, anchor=tk.W, pady=(5, 5))
        
        self.stage4_vars['kr_offset'] = self._add_form_entry(self.stage4_adv_frame, "KR Offset:", "2.0", "Shifts the growth of Order N with frequency.")
        
        table_frame = ttk.Frame(self.stage4_adv_frame)
        table_frame.pack(anchor=tk.W, fill=tk.X, pady=2)
        
        self.stage4_vars['use_manual_table'] = tk.BooleanVar(value=False)
        cb_table = ttk.Checkbutton(table_frame, text="Use Manual Order Table", variable=self.stage4_vars['use_manual_table'])
        cb_table.pack(side=tk.LEFT)
        ttk.Button(table_frame, text="Edit Table", command=self._open_manual_table_editor).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(self.stage4_adv_frame, text="Regularization", font=("Arial", 9, "bold")).pack(side=tk.TOP, anchor=tk.W, pady=(15, 5))
        
        self.stage4_vars['noise_floor_start_db'] = self._add_form_entry(self.stage4_adv_frame, "Noise Floor Start (dB):", "-30.0", "The point where damping starts.")
        self.stage4_vars['noise_floor_max_db'] = self._add_form_entry(self.stage4_adv_frame, "Noise Floor Max (dB):", "-50.0", "The point where damping hits MAX_LAMBDA.")
        self.stage4_vars['max_lambda'] = self._add_form_entry(self.stage4_adv_frame, "Max Lambda:", "0.00000010", "The maximum penalty applied to modes.")
        self.stage4_vars['use_optimized_origins'] = self._add_checkbutton(self.stage4_adv_frame, "Use Optimized Origins", True, "Essential for best fit.")
        
        # --- Button ---
        self.btn_stage4_run = ttk.Button(main_container, text="Run Stage 4", command=self._action_run_stage4)
        self.btn_stage4_run.pack(side=tk.TOP, pady=20)

    def _toggle_stage4_advanced(self):
        if self.stage4_adv_frame.winfo_ismapped():
            self.stage4_adv_frame.pack_forget()
            self.btn_stage4_advanced.config(text="Show Advanced Settings ▼")
            self.stage4_canvas.yview_moveto(0)
        else:
            self.stage4_adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_stage4_run)
            self.btn_stage4_advanced.config(text="Hide Advanced Settings ▲")

    def _open_manual_table_editor(self):
        top = tk.Toplevel(self)
        top.title("Manual Order Table Editor")
        top.geometry("300x400")
        
        frame = ttk.Frame(top, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Frequency', 'Order_N')
        tree = ttk.Treeview(frame, columns=columns, show='headings')
        tree.heading('Frequency', text='Frequency (Hz)')
        tree.heading('Order_N', text='Order N')
        tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        for f in sorted(self.stage4_manual_table.keys(), key=float):
            tree.insert('', tk.END, values=(f, self.stage4_manual_table[f]))
            
        def sync_table():
            new_table = {}
            for item in tree.get_children():
                vals = tree.item(item, 'values')
                try:
                    # Only valid floats/ints get added to the config, ignoring placeholders
                    f = float(vals[0])
                    n = int(vals[1])
                    new_table[f] = n
                except ValueError:
                    pass
            self.stage4_manual_table = new_table

        def on_double_click(event):
            region = tree.identify_region(event.x, event.y)
            if region != 'cell':
                return
            item = tree.identify_row(event.y)
            column = tree.identify_column(event.x)
            
            bbox = tree.bbox(item, column)
            if not bbox:
                return
            x, y, w, h = bbox
            
            col_idx = int(column[1:]) - 1
            current_value = tree.item(item, 'values')[col_idx]
            
            entry = ttk.Entry(tree)
            entry.place(x=x, y=y, width=w, height=h)
            entry.insert(0, current_value)
            entry.select_range(0, tk.END)
            entry.focus_set()
            
            def save_edit(event=None):
                if not entry.winfo_exists():
                    return
                new_val = entry.get()
                values = list(tree.item(item, 'values'))
                values[col_idx] = new_val
                tree.item(item, values=values)
                entry.destroy()
                sync_table()
                
            entry.bind('<Return>', save_edit)
            entry.bind('<FocusOut>', save_edit)
            
        tree.bind('<Double-1>', on_double_click)

        entry_frame = ttk.Frame(frame)
        entry_frame.pack(fill=tk.X, pady=5)
        
        def add_row():
            tree.insert('', tk.END, values=('frequency', 'order_n'))
                
        def delete_row():
            selected = tree.selection()
            for item in selected:
                tree.delete(item)
            sync_table()
                
        ttk.Button(entry_frame, text="Add Row", command=add_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(entry_frame, text="Delete", command=delete_row).pack(side=tk.LEFT, padx=5)
        
        def close_editor():
            sync_table()
            top.destroy()
            
        ttk.Button(frame, text="Close", command=close_editor).pack(side=tk.BOTTOM, pady=5)

    def _action_run_stage4(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Run Stage 4")
        self._save_project_if_not_exists()
        self.btn_stage4_run.config(state=tk.DISABLED)
        self.cli_text.config(state=tk.NORMAL)
        self.cli_text.insert(tk.END, "\n--- Starting Stage 4 ---\n")
        self.cli_text.config(state=tk.DISABLED)
        threading.Thread(target=self._run_stage4_thread, daemon=True).start()

    def _run_stage4_thread(self):
        try:
            proj_dir = self.project_dir.get()
            input_dir = os.path.join(proj_dir, "outputs")
            input_filename = f"{self.project_name.get()}_complex_data.npz"
            output_dir = os.path.join(proj_dir, "outputs", "coefficients")
            output_filename = f"{self.project_name.get()}_coefficients.h5"
            
            target_n_max = int(self.stage4_vars['target_n_max'].get())
            kr_offset = float(self.stage4_vars['kr_offset'].get())
            use_manual_table = self.stage4_vars['use_manual_table'].get()
            noise_floor_start = float(self.stage4_vars['noise_floor_start_db'].get())
            noise_floor_max = float(self.stage4_vars['noise_floor_max_db'].get())
            max_lambda = float(self.stage4_vars['max_lambda'].get())
            use_optimized_origins = self.stage4_vars['use_optimized_origins'].get()
            
            from stage4_run_she_solve import run_she_solve
            from viewers import plot_she_results
            
            manual_table = {float(k): int(v) for k, v in getattr(self, 'stage4_manual_table', {}).items()}
            
            results = run_she_solve(
                input_filename_she=input_filename,
                output_filename_she=output_filename,
                input_dir_she=input_dir,
                output_dir_she=output_dir,
                target_n_max=target_n_max,
                use_manual_table=use_manual_table,
                manual_order_table=manual_table,
                noise_floor_start_db=noise_floor_start,
                noise_floor_max_db=noise_floor_max,
                max_lambda=max_lambda,
                condition_metrics=True,
                use_optimized_origins=use_optimized_origins,
                save_to_disk=True,
                speed_of_sound=343.0,
                kr_offset=kr_offset,
                jobs=None,
                show_plot=False
            )
            
            def launch_plotter():
                if results:
                    save_prefix = os.path.join(output_dir, f"{self.project_name.get()}_coefficients")
                    she_dict = {
                        "freqs": results["freqs"],
                        "coeffs": results["coeffs"],
                        "N_used": results["N_used"],
                        "origins_mm": results["origins_mm"]
                    }
                    plot_she_results(
                        f_sel=results["freqs"],
                        pct_error=results["pct_error"],
                        res_cond=results["cond"],
                        n_used=results["N_used"],
                        condition_metrics=True,
                        P_measured=results.get("P_measured"),
                        resid_vec=results.get("residual_vector"),
                        save_path_prefix=save_prefix,
                        she_dict=she_dict,
                        coords_sph=results.get("coords_sph"),
                        c_sound=343.0
                    )
            
            self.after(0, launch_plotter)
            print("Stage 4 completed successfully.")
        except Exception as e:
            print(f"Error during Stage 4: {e}")
        finally:
            self.after(0, lambda: self.btn_stage4_run.config(state=tk.NORMAL))

    def _build_stage5_ui(self):
        scroll_frame = ScrollableFrame(self.tab_stage5)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        self.stage5_scroll_canvas = scroll_frame.canvas
        main_container = scroll_frame.scrollable_frame
        
        self.stage5_vars = {}
        if not hasattr(self, 'stage5_manual_coords'):
            self.stage5_manual_coords = []


        # --- Save Settings ---
        main_settings_frame = ttk.LabelFrame(main_container, text="Save Settings", padding="10")
        main_settings_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        out_dir_frame = ttk.Frame(main_settings_frame)
        out_dir_frame.pack(fill=tk.X, pady=2)
        out_dir_frame.columnconfigure(1, weight=1)
        ttk.Label(out_dir_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W)
        self.stage5_vars['output_dir'] = tk.StringVar(value="outputs/response_files")
        ttk.Entry(out_dir_frame, textvariable=self.stage5_vars['output_dir']).grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Button(out_dir_frame, text="Browse", command=self._browse_stage5_output_dir).grid(row=0, column=2)
        ToolTip(out_dir_frame, "Directory to save the output FRD/WAV files.")

        self.stage5_vars['frd_prefix'] = self._add_form_entry(main_settings_frame, "FRD Prefix:", self.project_name.get(), "Base name for exported files. Defaults to project name.")
        self.project_name.trace_add("write", lambda *args: self.stage5_vars['frd_prefix'].set(self.project_name.get()))

        check_frame = ttk.Frame(main_settings_frame)
        check_frame.pack(fill=tk.X, pady=5)
        self.stage5_vars['subtract_tof'] = self._add_checkbutton(check_frame, "Subtract Time-of-Flight Phase", True, "Subtract TOF phase to reduce FRD phase wrapping.")
        self.stage5_vars['generate_ir_files'] = self._add_checkbutton(check_frame, "Generate IR Files (.wav)", False, "Generate .wav impulse responses from complex pressures.")

        # --- Microphone Calibration ---
        mic_cal_frame = ttk.LabelFrame(main_container, text="Microphone Calibration", padding="10")
        mic_cal_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        cal_file_frame = ttk.Frame(mic_cal_frame)
        cal_file_frame.pack(fill=tk.X, pady=(0, 5))
        cal_file_frame.columnconfigure(1, weight=1)
        ttk.Label(cal_file_frame, text="Cal File:").grid(row=0, column=0, sticky=tk.W)
        self.stage5_vars['mic_cal_file'] = tk.StringVar(value="")
        ttk.Entry(cal_file_frame, textvariable=self.stage5_vars['mic_cal_file']).grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Button(cal_file_frame, text="Browse", command=self._browse_mic_cal_file).grid(row=0, column=2)

        cal_top_frame = ttk.Frame(mic_cal_frame)
        cal_top_frame.pack(fill=tk.X, pady=2)
        
        chk_frame = ttk.Frame(cal_top_frame)
        chk_frame.pack(side=tk.LEFT, fill=tk.X)
        self.stage5_vars['apply_mic_cal'] = self._add_checkbutton(chk_frame, "Apply Microphone Calibration", False, "Apply FRD mic calibration curve to the output.")
        
        combo_frame = ttk.Frame(cal_top_frame)
        combo_frame.pack(side=tk.LEFT, fill=tk.X)
        ttk.Label(combo_frame, text="Mode:").pack(side=tk.LEFT, padx=(10, 5))
        self.stage5_vars['mic_cal_mode'] = tk.StringVar(value="subtract")
        cb_mode = ttk.Combobox(combo_frame, textvariable=self.stage5_vars['mic_cal_mode'], values=["subtract", "add"], state="readonly", width=10)
        cb_mode.pack(side=tk.LEFT)
        ToolTip(cb_mode, "Mode to apply the calibration.")

        # --- Reference Axis ---
        ref_axis_frame = ttk.LabelFrame(main_container, text="Reference Axis", padding="10")
        ref_axis_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        offset_frame = ttk.Frame(ref_axis_frame)
        offset_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(offset_frame, text="Mic Offset (m):").pack(side=tk.LEFT, padx=(0, 10))
        self.stage5_vars['offset_mic_x'] = self._add_labeled_entry(offset_frame, "X:", "0.0", 5)
        self.stage5_vars['offset_mic_y'] = self._add_labeled_entry(offset_frame, "Y:", "0.0", 5)
        self.stage5_vars['offset_mic_z'] = self._add_labeled_entry(offset_frame, "Z:", "0.0", 5)

        zero_angle_frame = ttk.Frame(ref_axis_frame)
        zero_angle_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(zero_angle_frame, text="Zero Angle (deg):").pack(side=tk.LEFT, padx=(0, 10))
        self.stage5_vars['zero_theta_deg'] = self._add_labeled_entry(zero_angle_frame, "Theta:", "90.0", 5)
        self.stage5_vars['zero_phi_deg'] = self._add_labeled_entry(zero_angle_frame, "Phi:", "0.0", 5)

        # --- Evaluation Mode ---
        eval_frame = ttk.LabelFrame(main_container, text="Evaluation Mode", padding="10")
        eval_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.stage5_dist_frame = ttk.Frame(eval_frame, padding=(0, 0, 0, 5))
        self.stage5_dist_frame.pack(side=tk.TOP, fill=tk.X)
        self.stage5_vars['dist_mic'] = self._add_form_entry(self.stage5_dist_frame, "Distance (m):", "1.0", "Radius of measurement arc / CTA-2034 sphere (m).")

        # --- CTA-2034 Mode ---
        self.stage5_cta_frame = ttk.Frame(eval_frame, padding=(0, 5, 0, 5))
        self.stage5_cta_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Label(self.stage5_cta_frame, text="CTA-2034:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.stage5_vars['cta_mode'] = self._add_checkbutton(self.stage5_cta_frame, "Generate CTA-2034 graphs", False, "Generate full CTA-2034-A metrics. Overrides Measurement Arc Sweep.")

        # --- Arc Sweep Mode ---
        self.stage5_arc_frame = ttk.Frame(eval_frame, padding=(0, 5, 0, 5))
        self.stage5_arc_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Label(self.stage5_arc_frame, text="Measurement Arc Sweep:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.stage5_vars['range_deg'] = self._add_form_entry(self.stage5_arc_frame, "Range (± deg):", "90", "Sweep ± range in degrees (e.g., 90 = ±90°).")
        self.stage5_vars['increment_deg'] = self._add_form_entry(self.stage5_arc_frame, "Increment (deg):", "10", "Increment of sweep (°).")
        self.stage5_vars['direction'] = self._add_combobox(self.stage5_arc_frame, "Sweep Direction:", ["horizontal", "vertical", "hor_vert"], "horizontal", "Sweep direction.")

        # --- Manual List Mode ---
        self.stage5_manual_frame = ttk.Frame(eval_frame, padding=(0, 5, 0, 5))
        self.stage5_manual_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Label(self.stage5_manual_frame, text="Manual Coordinate List:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        manual_inner_frame = ttk.Frame(self.stage5_manual_frame)
        manual_inner_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.stage5_vars['manual_list_mode'] = tk.BooleanVar(value=False)
        cb_manual = ttk.Checkbutton(manual_inner_frame, text="Use Manual Coordinate List", variable=self.stage5_vars['manual_list_mode'])
        cb_manual.pack(side=tk.LEFT, padx=(0,5))
        ToolTip(cb_manual, "Use a custom list of coordinates. Overrides all other evaluation modes including reference axis offsets and rotations.")
        
        self.btn_stage5_edit_coords = ttk.Button(manual_inner_frame, text="Edit List", command=self._open_stage5_coord_table_editor, state=tk.DISABLED)
        self.btn_stage5_edit_coords.pack(side=tk.LEFT, padx=10)

        self._updating_eval_mode = False

        def _on_cta_changed(*args):
            if getattr(self, '_updating_eval_mode', False): return
            self._updating_eval_mode = True
            if self.stage5_vars['cta_mode'].get():
                self.stage5_vars['manual_list_mode'].set(False)
                self.stage5_vars['dist_mic'].set("2.0")
            _sync_eval_ui()
            self._updating_eval_mode = False

        def _on_manual_changed(*args):
            if getattr(self, '_updating_eval_mode', False): return
            self._updating_eval_mode = True
            if self.stage5_vars['manual_list_mode'].get():
                self.stage5_vars['cta_mode'].set(False)
            _sync_eval_ui()
            self._updating_eval_mode = False

        def _sync_eval_ui():
            is_manual = self.stage5_vars['manual_list_mode'].get()
            is_cta = self.stage5_vars['cta_mode'].get()

            self._set_widget_state(self.stage5_dist_frame, tk.DISABLED if is_manual else tk.NORMAL)
            self._set_widget_state(self.stage5_arc_frame, tk.DISABLED if is_manual or is_cta else tk.NORMAL)
            self.btn_stage5_edit_coords.config(state=tk.NORMAL if is_manual else tk.DISABLED)
            self._set_widget_state(ref_axis_frame, tk.DISABLED if is_manual else tk.NORMAL)

        # Add traces to all relevant vars to trigger live preview update
        for key in ['cta_mode', 'manual_list_mode', 'subtract_tof', 'generate_ir_files', 'direction']:
            self.stage5_vars[key].trace_add("write", self._schedule_update_stage5_preview)
        
        # For entry boxes, the update is handled by FocusOut/Return bindings in _add_form_entry/_add_labeled_entry
        # But we still need to trace the mode changes here
        self.stage5_vars['cta_mode'].trace_add("write", lambda *args: [ _on_cta_changed(), self._schedule_update_stage5_preview()])
        self.stage5_vars['manual_list_mode'].trace_add("write", lambda *args: [ _on_manual_changed(), self._schedule_update_stage5_preview()])

        # Initialize DUT variables for the right-hand viewer pane without creating left-pane widgets
        self.stage5_vars['dut_width_y'] = tk.StringVar(value="0.20")
        self.stage5_vars['dut_depth_x'] = tk.StringVar(value="0.25")
        self.stage5_vars['dut_height_z'] = tk.StringVar(value="0.40")

        # Bind updates for entries that don't use the trace system
        for var_key in ['dist_mic', 'range_deg', 'increment_deg', 'offset_mic_x', 'offset_mic_y', 'offset_mic_z', 'zero_theta_deg', 'zero_phi_deg', 'dut_width_y', 'dut_depth_x', 'dut_height_z']:
            # This is a bit of a hack to get the widget from the var, assuming it was the last one created
            self.stage5_vars[var_key].trace_add('write', self._schedule_update_stage5_preview)

        _sync_eval_ui()

        # --- Advanced Settings ---
        self.btn_stage5_advanced = ttk.Button(main_container, text="Show Advanced Settings ▼", command=self._toggle_stage5_advanced)
        self.btn_stage5_advanced.pack(side=tk.TOP, pady=10)

        self.stage5_adv_frame = ttk.LabelFrame(main_container, text="Advanced Settings", padding="10")
        
        self.stage5_vars['obs_mode'] = self._add_combobox(self.stage5_adv_frame, "Observation Mode:", ["Internal", "External", "Full"], "Internal", "Wavefront observation mode. Internal is standard anechoic output.")
        self.stage5_vars['mic_cal_fade_octaves'] = self._add_form_entry(self.stage5_adv_frame, "Mic Cal Fade Octaves:", "1.0", "Octaves to fade out of band correction to zero.")
        self.stage5_vars['use_optimized_origins'] = self._add_checkbutton(self.stage5_adv_frame, "Use Optimized Origins", True, "Use frequency-dependent optimized acoustic origins if available in the coefficient file.")

        self.btn_stage5_run = ttk.Button(main_container, text="Run Stage 5", command=self._action_run_stage5)
        self.btn_stage5_run.pack(side=tk.TOP, pady=20)
        
    def _toggle_stage5_advanced(self):
        if self.stage5_adv_frame.winfo_ismapped():
            self.stage5_adv_frame.pack_forget()
            self.btn_stage5_advanced.config(text="Show Advanced Settings ▼")
            self.stage5_scroll_canvas.yview_moveto(0)
        else:
            self.stage5_adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_stage5_run)
            self.btn_stage5_advanced.config(text="Hide Advanced Settings ▲")

    def _open_stage5_coord_table_editor(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Open Manual Coordinate List Editor")
        top = tk.Toplevel(self)
        top.title("Manual Coordinate List Editor")
        top.geometry("450x400")
        
        frame = ttk.Frame(top, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Theta', 'Phi', 'Radius')
        tree = ttk.Treeview(frame, columns=columns, show='headings')
        tree.heading('Theta', text='Theta (deg)')
        tree.heading('Phi', text='Phi (deg)')
        tree.heading('Radius', text='Radius (m)')
        
        for col in columns:
            tree.column(col, width=120, stretch=True)
            
        tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        for row in getattr(self, 'stage5_manual_coords', []):
            tree.insert('', tk.END, values=row)
            
        def sync_table():
            new_table = []
            for item in tree.get_children():
                vals = tree.item(item, 'values')
                try:
                    th = float(vals[0])
                    ph = float(vals[1])
                    r = float(vals[2])
                    new_table.append([th, ph, r])
                except (ValueError, IndexError):
                    pass
            self.stage5_manual_coords = new_table

        def on_double_click(event):
            region = tree.identify_region(event.x, event.y)
            if region != 'cell': return
            item = tree.identify_row(event.y)
            column = tree.identify_column(event.x)
            bbox = tree.bbox(item, column)
            if not bbox: return
            x, y, w, h = bbox
            col_idx = int(column[1:]) - 1
            current_value = tree.item(item, 'values')[col_idx]
            entry = ttk.Entry(tree)
            entry.place(x=x, y=y, width=w, height=h)
            entry.insert(0, current_value)
            entry.focus_set()
            def save_edit(event=None):
                if not entry.winfo_exists(): return
                values = list(tree.item(item, 'values'))
                values[col_idx] = entry.get()
                tree.item(item, values=values)
                entry.destroy()
                sync_table()
            entry.bind('<Return>', save_edit)
            entry.bind('<FocusOut>', save_edit)
        tree.bind('<Double-1>', on_double_click)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Add Row", command=lambda: tree.insert('', tk.END, values=('90.0', '0.0', '2.0'))).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Row", command=lambda: [tree.delete(i) for i in tree.selection()]).pack(side=tk.LEFT, padx=5)
        
        def close_and_update():
            sync_table()
            top.destroy()
            self._schedule_update_stage5_preview()
            
        ttk.Button(frame, text="Close", command=close_and_update).pack(side=tk.BOTTOM, pady=5)

    def _schedule_update_stage5_preview(self, *args):
        if self.stage5_update_job:
            self.after_cancel(self.stage5_update_job)
        self.stage5_update_job = self.after(300, self._update_stage5_preview)

    def _update_stage5_preview(self):
        if self.stage5_viewer is None:
            return
        try:
            import math
            import numpy as np
            
            box_dims = (
                float(self.stage5_vars['dut_width_y'].get()),
                float(self.stage5_vars['dut_depth_x'].get()),
                float(self.stage5_vars['dut_height_z'].get())
            )
            offset_xyz = (
                float(self.stage5_vars['offset_mic_x'].get()),
                float(self.stage5_vars['offset_mic_y'].get()),
                float(self.stage5_vars['offset_mic_z'].get())
            )
            z_th = float(self.stage5_vars['zero_theta_deg'].get())
            z_ph = float(self.stage5_vars['zero_phi_deg'].get())

            mic_xyz = []
            r = float(self.stage5_vars['dist_mic'].get())

            if self.stage5_vars['manual_list_mode'].get():
                # In manual mode, coordinates are absolute spherical. No offsets or rotations are applied.
                for row in getattr(self, 'stage5_manual_coords', []):
                    try:
                        th, ph, r_man = float(row[0]), float(row[1]), float(row[2])
                        th_rad, ph_rad = math.radians(th), math.radians(ph)
                        x = r_man * math.sin(th_rad) * math.cos(ph_rad)
                        y = r_man * math.sin(th_rad) * math.sin(ph_rad)
                        z = r_man * math.cos(th_rad)
                        mic_xyz.append(np.array([x, y, z]))
                    except: pass

            elif self.stage5_vars['cta_mode'].get():
                # CTA-2034 Mode: Implement correct 3D rotation
                th_rad = math.radians(z_th)
                ph_rad = math.radians(z_ph)

                # Define the local coordinate system basis vectors
                F = np.array([math.sin(th_rad) * math.cos(ph_rad), math.sin(th_rad) * math.sin(ph_rad), math.cos(th_rad)])
                R = np.array([-math.sin(ph_rad), math.cos(ph_rad), 0])
                U = np.cross(F, R)
                rot_matrix = np.array([F, R, U]).T

                # Generate standard CTA-2034 points in a local frame and rotate them
                for ang_deg in range(0, 360, 10):
                    # Horizontal orbit (local XY plane)
                    p_local_hor = np.array([r * math.cos(math.radians(ang_deg)), r * math.sin(math.radians(ang_deg)), 0])
                    mic_xyz.append(rot_matrix @ p_local_hor)

                    # Vertical orbit (local XZ plane)
                    p_local_ver = np.array([r * math.cos(math.radians(ang_deg)), 0, r * math.sin(math.radians(ang_deg))])
                    mic_xyz.append(rot_matrix @ p_local_ver)

            else:
                # Arc Sweep Mode: Implement correct 3D rotation
                th_rad = math.radians(z_th)
                ph_rad = math.radians(z_ph)

                # Define the local coordinate system basis vectors robustly to avoid gimbal lock
                # Forward vector (local X')
                F = np.array([math.sin(th_rad) * math.cos(ph_rad), math.sin(th_rad) * math.sin(ph_rad), math.cos(th_rad)])
                
                # Right vector (local Y')
                R = np.array([-math.sin(ph_rad), math.cos(ph_rad), 0])

                # Up vector (local Z'), derived from the other two to ensure a right-handed system
                U = np.cross(F, R)
                rot_matrix = np.array([F, R, U]).T

                rng = int(self.stage5_vars['range_deg'].get())
                inc = int(self.stage5_vars['increment_deg'].get())
                direction = self.stage5_vars['direction'].get()
                angles = range(-rng, rng + 1, inc) if inc > 0 else []

                for ang_deg in angles:
                    ang_rad = math.radians(ang_deg)
                    if direction in ["horizontal", "hor_vert"]:
                        p_local = np.array([r * math.cos(ang_rad), r * math.sin(ang_rad), 0])
                        mic_xyz.append(rot_matrix @ p_local)
                    if direction in ["vertical", "hor_vert"]:
                        if direction == "hor_vert" and ang_deg == 0: continue
                        p_local = np.array([r * math.cos(ang_rad), 0, r * math.sin(ang_rad)])
                        mic_xyz.append(rot_matrix @ p_local)

            final_mic_xyz = []
            if mic_xyz:
                if not self.stage5_vars['manual_list_mode'].get():
                    offset_vec = np.array(offset_xyz)
                    final_mic_xyz = [pt + offset_vec for pt in mic_xyz]
                else:
                    final_mic_xyz = mic_xyz
            
            self.stage5_viewer.update_view(
                box_dims=box_dims, 
                mic_coords_xyz=final_mic_xyz, 
                ref_origin=offset_xyz, # The axis lines still originate from the offset
                zero_theta_deg=z_th, 
                zero_phi_deg=z_ph
            )
            
        except ValueError:
            pass # Silence float conversion errors from empty fields
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not generate preview:\n{str(e)}")

    def _action_run_stage5(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Run Stage 5")
        self._save_project_if_not_exists()
        self.btn_stage5_run.config(state=tk.DISABLED)
        self.cli_text.config(state=tk.NORMAL)
        self.cli_text.insert(tk.END, "\n--- Starting Stage 5 ---\n")
        self.cli_text.config(state=tk.DISABLED)
        threading.Thread(target=self._run_stage5_thread, daemon=True).start()

    def _run_stage5_thread(self):
        try:
            from stage5_extract_pressures import run_cta2034_extraction, run_sweep_extraction
            
            proj_dir = self.project_dir.get()
            proj_name = self.project_name.get()
            
            coeff_path = os.path.join(proj_dir, "outputs", "coefficients", f"{proj_name}_coefficients.h5")
            if not os.path.exists(coeff_path):
                raise FileNotFoundError(f"Coefficient file not found: {coeff_path}")

            output_dir_abs = os.path.join(proj_dir, self.stage5_vars['output_dir'].get())
            offset_xyz = (float(self.stage5_vars['offset_mic_x'].get()), float(self.stage5_vars['offset_mic_y'].get()), float(self.stage5_vars['offset_mic_z'].get()))
            
            apply_mic_cal = self.stage5_vars['apply_mic_cal'].get()
            mic_cal_file = os.path.join(proj_dir, self.stage5_vars['mic_cal_file'].get()) if self.stage5_vars['mic_cal_file'].get() else ""
            mic_cal_mode = self.stage5_vars['mic_cal_mode'].get()

            obs_mode = self.stage5_vars['obs_mode'].get()
            mic_cal_fade_octaves = float(self.stage5_vars['mic_cal_fade_octaves'].get())
            use_optimized_origins = self.stage5_vars['use_optimized_origins'].get()

            if self.stage5_vars['manual_list_mode'].get():
                manual_coords = getattr(self, 'stage5_manual_coords', [])
                if not manual_coords: raise ValueError("Manual Coordinate List mode is active, but the list is empty.")
                run_sweep_extraction(coeff_path=coeff_path, output_dir=output_dir_abs, use_coord_list=True, coord_list=manual_coords, zero_theta=float(self.stage5_vars['zero_theta_deg'].get()), zero_phi=float(self.stage5_vars['zero_phi_deg'].get()), dist_mic=float(self.stage5_vars['dist_mic'].get()), offset_xyz=offset_xyz, subtract_tof=self.stage5_vars['subtract_tof'].get(), frd_prefix=self.stage5_vars['frd_prefix'].get(), generate_ir_files=self.stage5_vars['generate_ir_files'].get(), apply_mic_cal=apply_mic_cal, mic_cal_file=mic_cal_file, mic_cal_mode=mic_cal_mode, obs_mode=obs_mode, mic_cal_fade_octaves=mic_cal_fade_octaves, use_optimized_origins=use_optimized_origins)
            elif self.stage5_vars['cta_mode'].get():
                run_cta2034_extraction(coeff_path=coeff_path, output_dir=output_dir_abs, dist_mic=float(self.stage5_vars['dist_mic'].get()), zero_theta=float(self.stage5_vars['zero_theta_deg'].get()), zero_phi=float(self.stage5_vars['zero_phi_deg'].get()), offset_xyz=offset_xyz, apply_mic_cal=apply_mic_cal, mic_cal_file=mic_cal_file, mic_cal_mode=mic_cal_mode, obs_mode=obs_mode, mic_cal_fade_octaves=mic_cal_fade_octaves, use_optimized_origins=use_optimized_origins)
            else:
                run_sweep_extraction(coeff_path=coeff_path, output_dir=output_dir_abs, direction=self.stage5_vars['direction'].get(), range_deg=int(self.stage5_vars['range_deg'].get()), increment_deg=int(self.stage5_vars['increment_deg'].get()), zero_theta=float(self.stage5_vars['zero_theta_deg'].get()), zero_phi=float(self.stage5_vars['zero_phi_deg'].get()), dist_mic=float(self.stage5_vars['dist_mic'].get()), offset_xyz=offset_xyz, subtract_tof=self.stage5_vars['subtract_tof'].get(), frd_prefix=self.stage5_vars['frd_prefix'].get(), generate_ir_files=self.stage5_vars['generate_ir_files'].get(), apply_mic_cal=apply_mic_cal, mic_cal_file=mic_cal_file, mic_cal_mode=mic_cal_mode, obs_mode=obs_mode, mic_cal_fade_octaves=mic_cal_fade_octaves, use_optimized_origins=use_optimized_origins)
            
            print("Stage 5 completed successfully.")
        except Exception as e:
            print(f"Error during Stage 5: {e}")
        finally:
            self.after(0, lambda: self.btn_stage5_run.config(state=tk.NORMAL))

    def on_closing(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Application closing")
        try:
            self._destroy_viewer()
        except Exception:
            pass
        if hasattr(self, 'debug_log_file') and self.debug_log_file:
            try:
                self.debug_log_file.close()
            except Exception:
                pass
        self.quit()
        self.destroy()
        os._exit(0)


def main():
    # Optional: DPI Awareness for Windows so it doesn't look blurry
    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
            
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt) or issubclass(exc_type, SystemExit):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(err_msg, file=sys.stderr)
        messagebox.showerror("Fatal Error", f"A fatal unhandled exception occurred:\n\n{err_msg}")
        
    sys.excepthook = custom_excepthook

    app = SpkrScannerApp()
    
    # Use a slightly nicer theme if available
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
        
        # Fix Combobox colors in clam theme so active (readonly) is white and disabled is gray
        style.map('TCombobox', fieldbackground=[('readonly', 'white'), ('disabled', '#e0e0e0')],
                  foreground=[('disabled', '#a0a0a0')])
        
    app.mainloop()


if __name__ == "__main__":
    main()