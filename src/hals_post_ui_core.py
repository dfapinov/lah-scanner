import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

# Clamp native math libraries before importing NumPy/SciPy/Matplotlib anywhere
# in the GUI process. Stage-local clamps are too late if the GUI has already
# initialized OpenBLAS/MKL through an earlier import.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import traceback
import json
import matplotlib

# Every Matplotlib figure in this application is embedded in Tkinter. Select
# Tk before importing pyplot so Matplotlib cannot auto-select and initialize Qt.
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import queue
import datetime
import gc

# Append script directories to sys.path so we can import them as libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'process'))
sys.path.append(os.path.join(current_dir, 'misc'))

try:
    from stage1_fdwsmooth import fdwsmooth
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import project scripts:\n{e}")

# --- Configuration ---
# Set to True to enable debug logging. This will write all CLI output to a timestamped 'hals_debug_*.log' file 
# and mirror it back to the original console terminal for easier debugging.
DEBUG_MODE = False # True / False

GUI_TOOLTIPS = {
    'output_dir': "Directory where final extracted FRD and/or WAV files will be saved.",
    'frd_prefix': "Prefix added to the beginning of all extracted file names.",
    'frd_db_offset': "Scales the exported FRD dB levels (does not affect IR wav files).",
    'subtract_tof': "Mathematically subtract the time-of-flight phase delay. 'Ref Origin' uses the physical distance. 'IR Peak' dynamically detects the impulse response peak. 'Min Phase Ref' robustly estimates the linear excess group delay relative to minimum phase.",
    'generate_ir_files': "If checked, generates .wav impulse responses along with the standard frequency response (.frd) text files.",
    'apply_mic_cal': "Applies the selected microphone calibration file to the extracted results.",
    'mic_cal_mode': "Subtract (standard for measurement mics where the cal file describes the mic's own response) or Add.",
    'offset_mic_x': "Offsets the measurement reference center relative to the physical grid center, in millimeters.",
    'offset_mic_y': "Offsets the measurement reference center relative to the physical grid center, in millimeters.",
    'offset_mic_z': "Offsets the measurement reference center relative to the physical grid center, in millimeters.",
    'dut_depth_x': "Cabinet depth in millimeters. Stage 5 converts this to meters before running the extraction scripts.",
    'show_stage2_origin': "Display the frequency-dependent acoustic origin detected by Stage 2 in the 3D viewer.",
    'stage2_origin_frequency_hz': "Requested acoustic-origin frequency. The nearest stored Stage 2 frequency bin is used and written back here.",
    'zero_theta_deg': "Defines which physical angle represents the front 'on-axis' of the speaker (Theta).",
    'zero_phi_deg': "Defines which physical angle represents the front 'on-axis' of the speaker (Phi).",
    'dist_mic': "Distance from the reference center to the virtual microphone.",
    'cta_mode': "Generates standardized CTA-2034 (Spinorama) extraction points.",
    'range_deg': "The positive and negative angular sweep range (e.g. 90 creates a 180 degree arc).",
    'increment_deg': "The angular step size between virtual microphone positions.",
    'direction': "The plane(s) to sweep the virtual microphone across.",
    'manual_list_mode': "Uses a custom list of specific coordinates rather than standard sweeps.",
    'obs_mode': "Internal (extracts direct sound from speaker), External (extracts room reflections), Full (recombines both).",
    'mic_cal_fade_octaves': "Octave span over which the microphone calibration smoothly fades to 0dB at the extremes of the measurement.",
    'use_optimized_origins_stage5': "Uses the frequency-dependent acoustic origins calculated in Stage 2 to extract the most accurate phase.",
    'manual_ir_capture_padding': "Enable manual editing of the IR capture padding correction. This is normally fixed to the value used by the Harmonic Drive capture suite and is exposed only for IR files created by another source.",
    'ir_capture_padding_samples': "Number of capture padding samples to subtract from Stage 5 phase. Leave manual editing off for Harmonic Drive capture suite IRs; change only when processing IR files from another source."
}

SPEED_OF_SOUND_TOOLTIP = "\n".join([
    "Speed of sound in air:",
    "338 m/s  approx 10 C",
    "339 m/s  approx 11.5 C",
    "340 m/s  approx 13 C",
    "341 m/s  approx 15 C",
    "342 m/s  approx 16.5 C",
    "343 m/s  approx 18 C",
    "344 m/s  approx 20 C",
    "345 m/s  approx 21.5 C",
    "346 m/s  approx 23 C",
    "347 m/s  approx 25 C",
    "348 m/s  approx 26.5 C",
])

STAGE3_UPPER_RANGE_HELP = (
    "Set this to the upper usable frequency of the driver or system under test, "
    "before significant roll-off starts.\n\n"
    "Typical examples: a full speaker or tweeter may reach 20 kHz; a midrange-only "
    "driver may reach 10 kHz; a woofer may reach 3 kHz. Use the measured response "
    "of your own source rather than treating these examples as fixed limits."
)
STAGE3_LOWER_RANGE_HELP = (
    "The default is populated from the start of the reflection-free range provided "
    "by Stage 1 windowing (the RFT boundary), when that metadata is available.\n\n"
    "If the upper test range is below the RFT boundary, Stage 3 automatically sets "
    "the lower test range to upper / 4 (two octaves below). "
    "For example: upper 2 kHz automatically gives lower 500 Hz."
)


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


# Match Tk's default panel grey for exposed areas below short scrollable forms.
SETTINGS_CANVAS_BG = "#d9d9d9"


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=SETTINGS_CANVAS_BG)
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
        self.project_file_path = None
        self.project_settings = {}
        # Track state for pop-up UI viewers
        self.stage5_viewer = None
        self.stage5_canvas = None
        self.stage5_update_job = None
        self.stage5_live_window = None
        self.stage5_live_update_job = None
        self.stage5_live_generation = 0
        self.stage5_live_running = False
        self.stage5_live_queue = queue.Queue()
        self.stage5_live_point_index = 0
        self.stage5_live_evaluator = None
        self.stage5_live_evaluator_path = None
        self.stage5_live_last_result = None
        self.stage5_live_plot_limits = {}
        self.stage5_live_plot_drag = None

        self._build_ui()
        self.bind("<FocusIn>", self._raise_stage5_live_with_main, add="+")
        self.bind("<Map>", self._raise_stage5_live_with_main, add="+")
        
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
        # Pane dimensions are only meaningful after the withdrawn main window
        # has been revealed. Build and size the initial metadata preview now.
        self.after_idle(self._sync_right_panel)
        self.after(100, self._sync_right_panel)

    def report_callback_exception(self, exc, val, tb):
        if exc is KeyboardInterrupt:
            try:
                print("UI callback interrupted by user.", file=sys.stderr)
            except Exception:
                pass
            return
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

        self.tab_project_meta = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.tab_project_meta, text="Project Metadata")

        self.tab_processing = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.tab_processing, text="Processing")
        
        self.proc_notebook = ttk.Notebook(self.tab_processing)
        self.proc_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_stage1 = ttk.Frame(self.proc_notebook)
        self.proc_notebook.add(self.tab_stage1, text="Stage 1: FDW & Smoothing")

        self.tab_stage2 = ttk.Frame(self.proc_notebook)
        self.proc_notebook.add(self.tab_stage2, text="Stage 2: Acoustic Origin")

        self.tab_stage3 = ttk.Frame(self.proc_notebook)
        self.proc_notebook.add(self.tab_stage3, text="Stage 3: Find Order N")

        self.tab_stage4 = ttk.Frame(self.proc_notebook)
        self.proc_notebook.add(self.tab_stage4, text="Stage 4: SHE Solve")

        self.tab_stage5 = ttk.Frame(self.proc_notebook)
        self.proc_notebook.add(self.tab_stage5, text="Stage 5: Extract Pressures")

        # --- RIGHT PANEL: Processing output and Stage 5 preview ---
        self.right_frame_processing = ttk.Frame(self.right_panel)
        self.right_frame_processing.grid(row=0, column=0, sticky="nsew")

        self.main_notebook.bind("<<NotebookTabChanged>>", self._on_main_tab_changed)
        self.proc_notebook.bind("<<NotebookTabChanged>>", self._on_proc_tab_changed)

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
        self.stage_job_queue = queue.Queue()
        self.active_stage_job = None
        
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

        self._build_project_metadata_ui()
        self._build_stage1_ui()
        self._build_stage2_ui()
        self._build_stage3_ui()
        self._build_stage4_ui()
        self._build_stage5_ui()

        # Force the paned window divider line to the exact middle
        self._center_divider()
        self.after(150, self._center_divider)
        self.after(400, self._center_divider)
        
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
        self.right_frame_processing.tkraise()
        self._sync_right_panel()

    def _on_proc_tab_changed(self, event):
        selected_idx = self.proc_notebook.index(self.proc_notebook.select())
        if DEBUG_MODE:
            print(f"[DEBUG] Processing notebook tab changed to index: {selected_idx}")
        self._sync_right_panel()

    def _sync_right_panel(self):
        """Show the right-hand content appropriate to the selected tab."""
        # Notebook change events can arrive while the UI is still being built.
        if not hasattr(self, 'stage5_vars'):
            return

        main_idx = self.main_notebook.index(self.main_notebook.select())
        proc_idx = self.proc_notebook.index(self.proc_notebook.select())

        if main_idx == 0:  # Project Metadata
            self._create_stage5_viewer()
            self._show_stage5_viewer_full_height()
            self._schedule_update_stage5_preview()
        elif proc_idx == 4:  # Stage 5: Extract Pressures
            self._create_stage5_viewer()
            self._show_stage5_viewer_with_cli()
            self._schedule_update_stage5_preview()
        else:
            self._destroy_stage5_viewer()
            self._restore_cli_full_height()

    def _show_stage5_viewer_full_height(self):
        """Use the complete right pane for the Project Metadata preview."""
        try:
            panes = self.processing_paned_window.panes()
            if len(panes) >= 2:
                self.processing_paned_window.pane(self.stage5_viewer_frame, weight=1)
                self.processing_paned_window.pane(panes[1], weight=0)
            self.after(50, self._set_stage5_sash_bottom)
        except tk.TclError:
            pass

    def _set_stage5_sash_bottom(self):
        try:
            self.processing_paned_window.sashpos(0, self.processing_paned_window.winfo_height())
        except tk.TclError:
            pass

    def _show_stage5_viewer_with_cli(self):
        """Restore the existing Stage 5 preview/CLI split."""
        try:
            panes = self.processing_paned_window.panes()
            if len(panes) >= 2:
                self.processing_paned_window.pane(self.stage5_viewer_frame, weight=3)
                self.processing_paned_window.pane(panes[1], weight=1)
            self.after(
                50,
                lambda: self.processing_paned_window.sashpos(
                    0, self.processing_paned_window.winfo_height() * 3 // 4
                )
            )
        except tk.TclError:
            pass

    def _restore_cli_full_height(self):
        if DEBUG_MODE:
            print("[DEBUG] Restoring CLI pane to full height")
        try:
            panes = self.processing_paned_window.panes()
            if len(panes) >= 2:
                self.processing_paned_window.pane(self.stage5_viewer_frame, weight=0)
                self.processing_paned_window.pane(panes[1], weight=1)
            self.after(50, self._set_cli_sash_top)
        except tk.TclError:
            pass

    def _set_cli_sash_top(self):
        try:
            self.processing_paned_window.sashpos(0, 0)
        except tk.TclError:
            pass

    def _create_stage5_viewer(self):
        if self.stage5_viewer is not None:
            return

        from viewers import Stage5Viewer
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Add a control bar at the top of the viewer pane
        top_bar = ttk.Frame(self.stage5_viewer_frame)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        dut_frame = ttk.Frame(top_bar, relief=tk.GROOVE, borderwidth=2)
        dut_frame.pack(side=tk.LEFT, fill=tk.Y)

        dut_row = ttk.Frame(dut_frame)
        dut_row.pack(side=tk.TOP, fill=tk.X, anchor=tk.W)
        ttk.Label(dut_row, text="DUT Baffle:").pack(side=tk.LEFT, padx=(5, 10), pady=2)
        self._add_labeled_entry(dut_row, "Depth X (mm):", self.stage5_vars['dut_depth_x'], 7, GUI_TOOLTIPS.get('dut_depth_x'))

        origin_row = ttk.Frame(dut_frame)
        origin_row.pack(side=tk.TOP, fill=tk.X, anchor=tk.W, pady=(0, 2))
        origin_toggle = ttk.Checkbutton(
            origin_row,
            text="Show Stage 2 acoustic origin",
            variable=self.stage5_vars['show_stage2_origin'],
            command=self._schedule_update_stage5_preview,
        )
        origin_toggle.pack(side=tk.LEFT, padx=(5, 3))
        ToolTip(origin_toggle, GUI_TOOLTIPS['show_stage2_origin'])
        self._add_labeled_entry(
            origin_row,
            "Frequency (Hz):",
            self.stage5_vars['stage2_origin_frequency_hz'],
            9,
            GUI_TOOLTIPS['stage2_origin_frequency_hz'],
        )
        self.stage2_origin_status_var = tk.StringVar(value="")
        ttk.Label(origin_row, textvariable=self.stage2_origin_status_var, font=("Arial", 8)).pack(
            side=tk.LEFT, padx=(4, 6)
        )
        
        ttk.Button(top_bar, text="Save View Image", command=self._save_stage5_image).pack(side=tk.RIGHT, padx=5)
        ttk.Separator(top_bar, orient='vertical').pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=2)

        note_text = "Note: Left mouse drag orbits on the turntable, middle drag shifts, and right drag zooms."
        self.stage5_note_label = ttk.Label(self.stage5_viewer_frame, text=note_text, font=("Arial", 8, "italic"))
        self.stage5_note_label.pack(side=tk.TOP, pady=(0, 2))

        if DEBUG_MODE:
            print("[DEBUG] Creating Stage 5 3D Viewer canvas")
        self.stage5_viewer = Stage5Viewer()
        self.stage5_canvas = FigureCanvasTkAgg(self.stage5_viewer.fig, master=self.stage5_viewer_frame)
        canvas_widget = self.stage5_canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
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
            
    def _save_stage5_image(self):
        if self.stage5_viewer is None:
            return
        try:
            proj_dir = self.project_dir.get()
            out_dir = os.path.join(proj_dir, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            
            proj_name = self.project_name.get().strip() or "project"
            save_path = os.path.join(out_dir, f"{proj_name}_stage5_view.png")
            
            self.stage5_viewer.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved Stage 5 view image to {save_path}")
        except Exception as e:
            print(f"Error saving Stage 5 view image: {e}")

    def _build_project_metadata_ui(self):
        self.grid_vars = {}
        self.global_vars = {}
        self.user_positions = []

        main_container = ttk.Frame(self.tab_project_meta, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        meta_frame = ttk.LabelFrame(main_container, text="Imported Project Reference Points", padding="10")
        meta_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.grid_vars['output_filename'] = tk.StringVar(value=f"{self.project_name.get()}_grid.csv")

        ttk.Label(meta_frame, text="Point", font=("Arial", 8, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        ttk.Label(meta_frame, text="Radius (r_mm)", font=("Arial", 8, "bold")).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(meta_frame, text="Azimuth (phi_deg)", font=("Arial", 8, "bold")).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(meta_frame, text="Height (z_mm)", font=("Arial", 8, "bold")).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        def _add_wp_entry(row, col):
            var = tk.StringVar(value="")
            entry = ttk.Entry(meta_frame, textvariable=var, width=12)
            entry.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            entry.bind("<FocusOut>", self._on_project_metadata_changed)
            entry.bind("<Return>", self._on_project_metadata_changed)
            return var

        for row, label, prefix in [
            (1, "Top Critical:", "top"),
            (2, "Bottom Critical:", "bot"),
            (3, "Tweeter:", "tw"),
            (4, "Reference Origin:", "ref_origin"),
            (5, "Baffle Bottom L:", "baffle_bl"),
            (6, "Baffle Top L:", "baffle_tl"),
            (7, "Baffle Top R:", "baffle_tr"),
        ]:
            ttk.Label(meta_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            self.grid_vars[f'wp_{prefix}_r'] = _add_wp_entry(row, 1)
            self.grid_vars[f'wp_{prefix}_phi'] = _add_wp_entry(row, 2)
            self.grid_vars[f'wp_{prefix}_z'] = _add_wp_entry(row, 3)

        global_frame = ttk.LabelFrame(main_container, text="Global Settings", padding="10")
        global_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 5))

        self.global_vars['enable_manual_speed_of_sound'] = tk.BooleanVar(value=False)
        speed_check = ttk.Checkbutton(
            global_frame,
            text="Enable Manual Speed of Sound",
            variable=self.global_vars['enable_manual_speed_of_sound']
        )
        speed_check.pack(side=tk.LEFT, padx=(0, 14))

        speed_label = ttk.Label(global_frame, text="Speed of Sound (m/s):")
        speed_label.pack(side=tk.LEFT, padx=(0, 10))
        self.global_vars['speed_of_sound'] = tk.StringVar(value="343")
        speed_entry = ttk.Entry(global_frame, textvariable=self.global_vars['speed_of_sound'], width=12)
        speed_entry.pack(side=tk.LEFT)
        speed_entry.bind("<FocusOut>", self._on_project_metadata_changed)
        speed_entry.bind("<Return>", self._on_project_metadata_changed)
        speed_check.config(command=lambda: [update_speed_state(), self._on_project_metadata_changed()])
        def update_speed_state(*args):
            state = tk.NORMAL if self.global_vars['enable_manual_speed_of_sound'].get() else tk.DISABLED
            speed_entry.config(state=state)
            speed_label.config(state=state)
        self.global_vars['enable_manual_speed_of_sound'].trace_add("write", update_speed_state)
        update_speed_state()
        ToolTip(speed_check, "Use the entered speed of sound instead of the default 343 m/s.")
        ToolTip(speed_label, SPEED_OF_SOUND_TOOLTIP)
        ToolTip(speed_entry, SPEED_OF_SOUND_TOOLTIP)

        add_frame = ttk.LabelFrame(main_container, text="User Positions", padding="10")
        add_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 5))
        self.user_positions_frame = add_frame
        self._refresh_user_positions_view()

        metadata_note = ttk.Label(
            main_container,
            text="Note: These values are imported from the upstream project JSON/CSV and are used by Stage 2 and the Stage 5 preview.",
            font=("Arial", 9, "italic")
        )
        metadata_note.pack(side=tk.TOP, anchor=tk.W, fill=tk.X, pady=(8, 0))
        main_container.bind("<Configure>", lambda e, lbl=metadata_note: lbl.configure(wraplength=max(240, e.width - 24)), add="+")

    def _on_project_metadata_changed(self, event=None):
        self._sync_upstream_project_fields()

    def _refresh_user_positions_view(self):
        frame = getattr(self, 'user_positions_frame', None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        self.user_position_display_vars = []

        ttk.Label(frame, text="Point", font=("Arial", 8, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        ttk.Label(frame, text="Radius (r_mm)", font=("Arial", 8, "bold")).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(frame, text="Azimuth (phi_deg)", font=("Arial", 8, "bold")).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(frame, text="Height (z_mm)", font=("Arial", 8, "bold")).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        positions = getattr(self, 'user_positions', []) or []
        if not positions:
            ttk.Label(frame, text="None").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            return

        def add_value(row, col, value):
            var = tk.StringVar(value=value)
            self.user_position_display_vars.append(var)
            entry = ttk.Entry(frame, textvariable=var, width=12, state="readonly")
            entry.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)

        for row, pos in enumerate(positions, start=1):
            try:
                name = str(pos.get('name', '')).strip() or "User Position"
                r = float(pos.get('r'))
                phi = float(pos.get('phi'))
                z = float(pos.get('z'))
                ttk.Label(frame, text=f"{name}:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=2)
                add_value(row, 1, f"{r:.3f}")
                add_value(row, 2, f"{phi:.3f}")
                add_value(row, 3, f"{z:.3f}")
            except Exception:
                continue

    def _coerce_user_positions(self, value):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            try:
                import ast
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    def _user_positions_from_mapping(self, mapping):
        for key in ('user_positions', 'User_positions'):
            positions = self._coerce_user_positions(mapping.get(key))
            if positions:
                return positions
        return []

    def _make_help_icon(self, parent, command):
        scale = parent.winfo_fpixels('1i') / 96
        size = round(24 * scale)
        icon = tk.PhotoImage(master=parent, width=size, height=size)
        radius, center = 10 * scale, size / 2
        for y in range(size):
            dy = y + .5 - center
            if abs(dy) < radius:
                half = (radius * radius - dy * dy) ** .5
                icon.put('#64717b', to=(round(center-half), y, round(center+half), y+1))
        badge = ttk.Label(parent, image=icon, text='?', compound='center',
                          foreground='white', font=('Segoe UI', 10, 'bold'),
                          cursor='hand2', takefocus=True, padding=0)
        badge.help_icon = icon
        for event in ('<Button-1>', '<Return>', '<space>'):
            badge.bind(event, lambda event: command())
        badge.bind('<FocusIn>', lambda event: badge.configure(foreground='#bce3ff'))
        badge.bind('<FocusOut>', lambda event: badge.configure(foreground='white'))
        return badge

    def _add_form_entry(self, parent, label_text, default_val, help_text=None, state_var=None, button_text=None, button_command=None):
        lbl_frame = ttk.Frame(parent)
        lbl_frame.pack(anchor=tk.W, fill=tk.X, pady=(5, 0))
        lbl = ttk.Label(lbl_frame, text=label_text)
        lbl.pack(side=tk.LEFT)
            
        var = tk.StringVar(value=default_val)
        if button_text and button_command:
            entry_frame = ttk.Frame(parent)
            entry_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 5))
            entry = ttk.Entry(entry_frame, textvariable=var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if button_text == '?':
                scale = entry_frame.winfo_fpixels('1i') / 96
                size = round(24 * scale)
                # Transparent pixels let the native ttk theme paint the surrounding panel.
                icon = tk.PhotoImage(master=entry_frame, width=size, height=size)
                radius = 10 * scale
                center = size / 2
                for y in range(size):
                    dy = y + .5 - center
                    if abs(dy) < radius:
                        half_width = (radius * radius - dy * dy) ** .5
                        left = round(center - half_width)
                        right = round(center + half_width)
                        icon.put('#64717b', to=(left, y, right, y + 1))
                btn = ttk.Label(entry_frame, image=icon, text='?', compound='center',
                                foreground='white', font=('Segoe UI', 10, 'bold'),
                                cursor='hand2', takefocus=True, padding=0)
                btn.help_icon = icon
                def open_help(event=None):
                    if str(btn.cget('state')) != 'disabled':
                        button_command()
                    return 'break'
                btn.bind('<Button-1>', open_help)
                btn.bind('<Return>', open_help)
                btn.bind('<space>', open_help)
                btn.bind('<FocusIn>', lambda e: btn.configure(foreground='#bce3ff'))
                btn.bind('<FocusOut>', lambda e: btn.configure(foreground='white'))
            else:
                btn = ttk.Button(entry_frame, text=button_text, command=button_command)
            btn.pack(side=tk.LEFT, padx=(6, 0))
        else:
            entry = ttk.Entry(parent, textvariable=var)
            entry.pack(anchor=tk.W, fill=tk.X, pady=(0, 5))
            btn = None
        
        if help_text:
            ToolTip(lbl, help_text)
            ToolTip(entry, help_text)
            if btn is not None:
                ToolTip(btn, help_text)
            
        if state_var:
            def update_state(*args):
                state = tk.NORMAL if state_var.get() else tk.DISABLED
                entry.config(state=state)
                lbl.config(state=state)
                if btn is not None:
                    btn.config(state=state)
            state_var.trace_add("write", update_state)
            update_state()
            
        return var

    def _add_labeled_entry(self, parent, label_text, default_val, width, help_text=None):
        frame = ttk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=5)
        lbl = ttk.Label(frame, text=label_text)
        lbl.pack(side=tk.LEFT)
        if isinstance(default_val, tk.StringVar):
            var = default_val
        else:
            var = tk.StringVar(value=default_val)
        entry = ttk.Entry(frame, textvariable=var, width=width)
        entry.pack(side=tk.LEFT)
        # Trigger update on focus out or enter
        entry.bind("<FocusOut>", self._schedule_update_stage5_preview)
        entry.bind("<Return>", self._schedule_update_stage5_preview)
        
        if help_text:
            ToolTip(lbl, help_text)
            ToolTip(entry, help_text)
            
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
            self._load_settings(directory, notify_metadata=True)

    def _find_stage1_ir_dir(self, project_dir):
        expected_names = ("measurement_set", "recordings")
        try:
            entries = os.listdir(project_dir)
        except OSError:
            entries = []

        for expected in expected_names:
            exact_path = os.path.join(project_dir, expected)
            if os.path.isdir(exact_path):
                return exact_path
            for entry in entries:
                candidate = os.path.join(project_dir, entry)
                if entry.lower() == expected and os.path.isdir(candidate):
                    return candidate

        return os.path.join(project_dir, expected_names[0])

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

    def _check_mic_cal_status(self, *args):
        if not hasattr(self, 'lbl_mic_cal_fallback'): return
        cal_file = self.stage5_vars.get('mic_cal_file', tk.StringVar()).get()
        if cal_file:
            if os.path.isabs(cal_file):
                full_path = cal_file
            else:
                full_path = os.path.join(self.project_dir.get(), cal_file)
            if not os.path.exists(full_path):
                if getattr(self, 'mic_cal_fallback_content', ""):
                    self.lbl_mic_cal_fallback.config(text="File missing. Using project save fallback.", foreground="orange")
                else:
                    self.lbl_mic_cal_fallback.config(text="File missing. No fallback available.", foreground="red")
            else:
                self.lbl_mic_cal_fallback.config(text="")
        else:
            self.lbl_mic_cal_fallback.config(text="")

    def _action_save_project(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Save Project")
        save_path = self._save_settings()
        self.status_var.set(f"Project Saved: {save_path}")

    def _save_project_if_not_exists(self):
        save_path = self._get_project_save_path()
        if not os.path.exists(save_path):
            if DEBUG_MODE:
                print(f"[DEBUG] Auto-saving initial project settings to {save_path}")
            self._save_settings()

    def _get_project_save_path(self):
        project_path = getattr(self, 'project_file_path', None)
        if project_path:
            return project_path
        proj_name = self.project_name.get().strip() or "scanner"
        return os.path.join(self.project_dir.get(), f"{proj_name}_project.json")

    def _read_existing_project_settings(self, save_path):
        for candidate in (save_path, getattr(self, 'project_file_path', None)):
            if not candidate or not os.path.exists(candidate):
                continue
            try:
                with open(candidate, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    return loaded
            except Exception as e:
                print(f"Warning: Could not read existing project settings for merge: {e}")
        return dict(getattr(self, 'project_settings', {}) or {})

    def _merged_section(self, base_settings, section_name, current_values):
        existing = base_settings.get(section_name, {})
        if isinstance(existing, dict):
            merged = dict(existing)
        else:
            merged = {}
        merged.update(current_values)
        return merged

    def _save_settings(self):
        save_path = self._get_project_save_path()
        base_settings = self._read_existing_project_settings(save_path)

        stage5_vars = getattr(self, 'stage5_vars', {})
        mic_cal_var = stage5_vars.get('mic_cal_file')
        mic_cal_path = mic_cal_var.get() if mic_cal_var else ""
        fallback_content = ""
        if mic_cal_path:
            if os.path.isabs(mic_cal_path):
                full_path = mic_cal_path
            else:
                full_path = os.path.join(self.project_dir.get(), mic_cal_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as cf:
                        fallback_content = cf.read()
                except Exception as e:
                    print(f"Warning: Could not read mic cal file for fallback: {e}")

        grid_settings = {k: v.get() for k, v in self.grid_vars.items()}
        if hasattr(self, 'user_positions'):
            grid_settings['user_positions'] = self.user_positions

        settings = dict(base_settings)
        settings["project_name"] = self.project_name.get()
        settings["global_vars"] = self._merged_section(
            base_settings,
            "global_vars",
            {k: v.get() for k, v in getattr(self, 'global_vars', {}).items()},
        )
        settings["grid_vars"] = self._merged_section(base_settings, "grid_vars", grid_settings)
        settings["stage1_vars"] = self._merged_section(
            base_settings,
            "stage1_vars",
            {k: v.get() for k, v in getattr(self, 'stage1_vars', {}).items()},
        )
        settings["stage2_vars"] = self._merged_section(
            base_settings,
            "stage2_vars",
            {k: v.get() for k, v in getattr(self, 'stage2_vars', {}).items()},
        )
        settings["stage3_vars"] = self._merged_section(
            base_settings,
            "stage3_vars",
            {k: v.get() for k, v in getattr(self, 'stage3_vars', {}).items()},
        )
        # Remove obsolete controls even when merging a previously saved project.
        settings["stage3_vars"].pop('spl_change_enabled', None)
        settings["stage3_vars"].pop('spl_floor_db', None)
        settings["stage4_vars"] = self._merged_section(
            base_settings,
            "stage4_vars",
            {k: v.get() for k, v in getattr(self, 'stage4_vars', {}).items()},
        )
        settings["stage5_vars"] = self._merged_section(
            base_settings,
            "stage5_vars",
            {k: v.get() for k, v in getattr(self, 'stage5_vars', {}).items()},
        )
        settings["stage5_gui_units"] = "mm"
        settings["stage4_manual_table"] = {str(k): v for k, v in getattr(self, 'stage4_manual_table', {}).items()}
        settings["mic_cal_fallback"] = fallback_content

        if DEBUG_MODE and hasattr(self, 'debug_log_file') and self.debug_log_file:
            self.debug_log_file.write(f"[DEBUG] Saving project settings:\n{json.dumps(settings, indent=4)}\n")
            self.debug_log_file.flush()

        try:
            with open(save_path, "w") as f:
                json.dump(settings, f, indent=4)
            self.project_file_path = save_path
            self.project_settings = settings
        except Exception as e:
            print(f"Warning: Failed to save project settings: {e}")
        return save_path

    def _load_settings(self, directory, notify_metadata=True):
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
                self.project_file_path = load_path
                self.project_settings = settings

                if DEBUG_MODE and hasattr(self, 'debug_log_file') and self.debug_log_file:
                    self.debug_log_file.write(f"[DEBUG] Loaded settings:\n{json.dumps(settings, indent=4)}\n")
                    self.debug_log_file.flush()

                if "project_name" in settings:
                    self.project_name.set(settings["project_name"])
                if "global_vars" in settings:
                    for k, v in settings["global_vars"].items():
                        if k in getattr(self, 'global_vars', {}):
                            if isinstance(self.global_vars[k], tk.BooleanVar):
                                self.global_vars[k].set(str(v).strip().lower() in {"1", "true", "yes", "on"})
                            else:
                                self.global_vars[k].set(str(v))
                if "grid_vars" in settings:
                    for k, v in settings["grid_vars"].items():
                        if k in getattr(self, 'grid_vars', {}):
                            if isinstance(self.grid_vars[k], tk.BooleanVar):
                                self.grid_vars[k].set(bool(v))
                            else:
                                self.grid_vars[k].set(str(v))
                    self.user_positions = self._user_positions_from_mapping(settings["grid_vars"])
                    self._refresh_user_positions_view()
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
                    loaded_stage3_keys = set(settings.get("stage3_vars", {}))
                    for k, v in settings["stage3_vars"].items():
                        if k in getattr(self, 'stage3_vars', {}):
                            if isinstance(self.stage3_vars[k], tk.BooleanVar):
                                self.stage3_vars[k].set(bool(v))
                            else:
                                self.stage3_vars[k].set(str(v))
                    self._seed_stage3_start_from_npz(force=('freq_start_hz' not in loaded_stage3_keys))
                else:
                    self._seed_stage3_start_from_npz(force=True)
                if "stage4_vars" in settings:
                    for k, v in settings["stage4_vars"].items():
                        if k in getattr(self, 'stage4_vars', {}):
                            if isinstance(self.stage4_vars[k], tk.BooleanVar):
                                self.stage4_vars[k].set(bool(v))
                            else:
                                self.stage4_vars[k].set(str(v))
                if "stage5_vars" in settings:
                    loaded_stage5_keys = set(settings.get("stage5_vars", {}))
                    stage5_gui_units = settings.get("stage5_gui_units", "m")
                    for k, v in settings["stage5_vars"].items():
                        if k in getattr(self, 'stage5_vars', {}):
                            if isinstance(self.stage5_vars[k], tk.BooleanVar):
                                self.stage5_vars[k].set(bool(v))
                            else:
                                val = str(v)
                                if k == 'subtract_tof':
                                    if val == "True" or val == "Grid Origin": val = "Ref Origin"
                                    elif val == "False": val = "Off"
                                if stage5_gui_units != "mm" and k in {'offset_mic_x', 'offset_mic_y', 'offset_mic_z', 'dut_depth_x'}:
                                    val = self._m_to_mm_text(val)
                                self.stage5_vars[k].set(val)
                    self.stage5_manual_coords = settings.get("stage5_vars", {}).get("manual_coord_list", [])
                    self._default_dut_depth_from_baffle(force=('dut_depth_x' not in loaded_stage5_keys))
                if "stage4_manual_table" in settings:
                    self.stage4_manual_table = {float(k): int(v) for k, v in settings["stage4_manual_table"].items()}

                if "mic_cal_fallback" in settings:
                    self.mic_cal_fallback_content = settings["mic_cal_fallback"]
                else:
                    self.mic_cal_fallback_content = ""
                    
                if hasattr(self, '_check_mic_cal_status'):
                    self._check_mic_cal_status()
                if notify_metadata:
                    self._reconcile_project_folder_metadata(project_exists=True, notify=True)
                self._sync_upstream_project_fields()

            except Exception as e:
                print(f"Warning: Failed to load project settings: {e}")
        else:
            self.project_file_path = None
            self.project_settings = {}
            if notify_metadata:
                self._reconcile_project_folder_metadata(project_exists=False, notify=True)
            self._refresh_user_positions_view()
            self._sync_upstream_project_fields()

    def _cyl_mm_to_xyz_mm(self, r_mm, phi_deg, z_mm):
        import math
        r = float(r_mm)
        ph = math.radians(float(phi_deg))
        return (r * math.cos(ph), r * math.sin(ph), float(z_mm))

    def _grid_value(self, key):
        var = getattr(self, 'grid_vars', {}).get(key)
        return var.get().strip() if var is not None else ""

    def _m_to_mm_text(self, value, decimals=3):
        try:
            return f"{float(value) * 1000.0:.{decimals}f}"
        except (TypeError, ValueError):
            return str(value)

    def _mm_to_m(self, value):
        return float(value) / 1000.0

    def _stage5_offset_m(self):
        return (
            self._mm_to_m(self.stage5_vars['offset_mic_x'].get()),
            self._mm_to_m(self.stage5_vars['offset_mic_y'].get()),
            self._mm_to_m(self.stage5_vars['offset_mic_z'].get()),
        )

    def _sync_upstream_project_fields(self):
        ref_origin_set = False
        try:
            tw_r = self._grid_value('wp_tw_r')
            tw_phi = self._grid_value('wp_tw_phi')
            tw_z = self._grid_value('wp_tw_z')
            if tw_r and tw_phi and tw_z and hasattr(self, 'stage2_vars'):
                x_mm, y_mm, z_mm = self._cyl_mm_to_xyz_mm(tw_r, tw_phi, tw_z)
                self.stage2_vars['tweeter_x'].set(f"{x_mm:.3f}")
                self.stage2_vars['tweeter_y'].set(f"{y_mm:.3f}")
                self.stage2_vars['tweeter_z'].set(f"{z_mm:.3f}")
        except Exception as e:
            print(f"Warning: Could not import tweeter position from project file: {e}")

        try:
            if hasattr(self, 'stage5_vars'):
                ref_r = self._grid_value('wp_ref_origin_r')
                ref_phi = self._grid_value('wp_ref_origin_phi')
                ref_z = self._grid_value('wp_ref_origin_z')
                if ref_r and ref_phi and ref_z:
                    x_mm, y_mm, z_mm = self._cyl_mm_to_xyz_mm(ref_r, ref_phi, ref_z)
                    ref_origin_set = True
                else:
                    x_mm, y_mm, z_mm = self._cyl_mm_to_xyz_mm(tw_r, tw_phi, tw_z)
                self.stage5_vars['offset_mic_x'].set(f"{x_mm:.3f}")
                self.stage5_vars['offset_mic_y'].set(f"{y_mm:.3f}")
                self.stage5_vars['offset_mic_z'].set(f"{z_mm:.3f}")
        except Exception as e:
            if ref_origin_set:
                print(f"Warning: Could not import reference origin from project file: {e}")

        self._default_dut_depth_from_baffle()

        if hasattr(self, '_schedule_update_stage5_preview'):
            self._schedule_update_stage5_preview()

    def _reference_metadata_keys(self):
        keys = []
        for prefix in ['top', 'bot', 'tw', 'ref_origin', 'baffle_bl', 'baffle_tl', 'baffle_tr']:
            for field in ['r', 'phi', 'z']:
                keys.append(f'wp_{prefix}_{field}')
        return keys

    def _get_project_baffle_width_m(self):
        try:
            _, tl_xyz, tr_xyz = self._get_project_baffle_front_corners_m()
            import math
            width_m = math.dist(tl_xyz, tr_xyz)
            return width_m if width_m > 0 else None
        except Exception:
            return None

    def _get_project_baffle_front_corners_m(self):
        """Return BL/TL/TR in metres, inferring TL for a two-corner baffle."""
        bl = (
            self._grid_value('wp_baffle_bl_r'),
            self._grid_value('wp_baffle_bl_phi'),
            self._grid_value('wp_baffle_bl_z'),
        )
        tl = (
            self._grid_value('wp_baffle_tl_r'),
            self._grid_value('wp_baffle_tl_phi'),
            self._grid_value('wp_baffle_tl_z'),
        )
        tr = (
            self._grid_value('wp_baffle_tr_r'),
            self._grid_value('wp_baffle_tr_phi'),
            self._grid_value('wp_baffle_tr_z'),
        )
        if not all(bl + tr):
            raise ValueError("Baffle bottom-left and top-right waypoints are required.")

        bl_xyz_mm = self._cyl_mm_to_xyz_mm(*bl)
        tr_xyz_mm = self._cyl_mm_to_xyz_mm(*tr)
        if all(tl):
            tl_xyz_mm = self._cyl_mm_to_xyz_mm(*tl)
        elif any(tl):
            raise ValueError("Baffle top-left waypoint is incomplete.")
        else:
            # Two diagonal corners define a vertical baffle: the left edge is
            # parallel to Z, so inferred TL shares BL's X/Y and TR's height.
            tl_xyz_mm = (bl_xyz_mm[0], bl_xyz_mm[1], tr_xyz_mm[2])

        scale = 1.0 / 1000.0
        return tuple(
            tuple(float(coord) * scale for coord in point)
            for point in (bl_xyz_mm, tl_xyz_mm, tr_xyz_mm)
        )

    def _default_dut_depth_from_baffle(self, force=False):
        if not hasattr(self, 'stage5_vars') or 'dut_depth_x' not in self.stage5_vars:
            return
        width_m = self._get_project_baffle_width_m()
        if width_m is None:
            return
        current = self.stage5_vars['dut_depth_x'].get().strip()
        try:
            current_val = float(current) if current else 0.0
        except ValueError:
            current_val = 0.0
        if force or current_val <= 0.0:
            self.stage5_vars['dut_depth_x'].set(f"{width_m * 1000.0:.3f}")

    def _find_metadata_csv_path(self):
        if not hasattr(self, 'grid_vars'):
            return None
        out_name = self.grid_vars.get('output_filename', tk.StringVar(value="")).get().strip()
        candidates = []
        if out_name:
            candidates.append(os.path.join(self.project_dir.get(), out_name))
        proj_name = self.project_name.get().strip() or "scanner"
        candidates.append(os.path.join(self.project_dir.get(), f"{proj_name}_grid.csv"))
        candidates.append(os.path.join(self.project_dir.get(), f"{proj_name}_scan_path.csv"))
        found = next((p for p in candidates if os.path.exists(p)), None)
        if found:
            return found

        try:
            for filename in os.listdir(self.project_dir.get()):
                if filename.endswith("_grid.csv") or filename.endswith("_scan_path.csv"):
                    discovered_name = filename.replace("_grid.csv", "").replace("_scan_path.csv", "")
                    if discovered_name:
                        self.project_name.set(discovered_name)
                    return os.path.join(self.project_dir.get(), filename)
        except OSError:
            pass
        return None

    def _metadata_from_current_project(self):
        metadata = {}
        for key in self._reference_metadata_keys():
            if key in getattr(self, 'grid_vars', {}):
                val = self._grid_value(key)
                if val:
                    metadata[key] = val
        if getattr(self, 'user_positions', None):
            metadata['user_positions'] = self.user_positions
        return metadata

    def _metadata_from_csv(self, csv_path):
        try:
            if not csv_path:
                return {}

            import ast
            import csv

            metadata = {}
            settings = {}
            explicit_keys = self._reference_metadata_keys()

            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in explicit_keys:
                        if key not in metadata:
                            val = str(row.get(key, "")).strip()
                            if val:
                                metadata[key] = val
                    row_user_positions = self._user_positions_from_mapping(row)
                    if "user_positions" not in metadata and row_user_positions:
                        metadata["user_positions"] = row_user_positions

                    item = str(row.get("gen_settings", "")).strip()
                    if "=" in item:
                        key, value = item.split("=", 1)
                        settings[key.strip()] = value.strip()

            for key in explicit_keys:
                if key not in metadata and key in settings:
                    metadata[key] = settings[key]

            def parse_tuple(value):
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                    return parsed
                return None

            for key, prefix in [
                ("tweeter_pos", "tw"),
                ("top_crit_pos", "top"),
                ("bot_crit_pos", "bot"),
                ("ref_origin_pos", "ref_origin"),
                ("baffle_bl_pos", "baffle_bl"),
                ("baffle_tl_pos", "baffle_tl"),
                ("baffle_tr_pos", "baffle_tr"),
                ("baffle_bot_l_pos", "baffle_bl"),
                ("baffle_top_l_pos", "baffle_tl"),
                ("baffle_top_r_pos", "baffle_tr"),
            ]:
                if key not in settings:
                    continue
                vals = parse_tuple(settings[key])
                if not vals:
                    continue
                for field, val in zip(["r", "phi", "z"], vals[:3]):
                    metadata[f"wp_{prefix}_{field}"] = str(val)

            user_positions = []
            for key, value in settings.items():
                if not key.startswith("user_position_"):
                    continue
                vals = parse_tuple(value)
                if not vals:
                    continue
                raw_name = key.replace("user_position_", "", 1)
                display_name = raw_name.replace("_", " ").strip().title()
                user_positions.append({
                    "name": display_name,
                    "r": float(vals[0]),
                    "phi": float(vals[1]),
                    "z": float(vals[2]),
                })

            user_positions_key = next((k for k in ("user_positions", "User_positions") if k in settings), None)
            if user_positions_key:
                parsed = self._coerce_user_positions(settings[user_positions_key])
                if parsed:
                    metadata["user_positions"] = parsed
            elif user_positions:
                metadata["user_positions"] = user_positions
            return metadata
        except Exception:
            return {}

    def _metadata_has_reference_points(self, metadata):
        keys = set(self._reference_metadata_keys())
        keys.add('user_positions')
        return any(k in metadata for k in keys)

    def _normalized_metadata(self, metadata):
        norm = {}
        for key in self._reference_metadata_keys():
            if key not in metadata:
                continue
            try:
                norm[key] = round(float(metadata[key]), 6)
            except (TypeError, ValueError):
                norm[key] = str(metadata[key]).strip()

        positions = []
        for pos in metadata.get('user_positions', []) or []:
            try:
                name = str(pos.get('name', '')).strip().lower().replace('_', ' ')
                name = " ".join(name.split())
                positions.append((
                    name,
                    round(float(pos.get('r')), 6),
                    round(float(pos.get('phi')), 6),
                    round(float(pos.get('z')), 6),
                ))
            except Exception:
                continue
        if positions:
            norm['user_positions'] = sorted(positions)
        return norm

    def _apply_reference_metadata(self, metadata):
        for key, val in metadata.items():
            if key == 'user_positions':
                if isinstance(val, list):
                    self.user_positions = val
                    self._refresh_user_positions_view()
            elif key in getattr(self, 'grid_vars', {}):
                self.grid_vars[key].set(str(val))

    def _summarize_metadata_difference(self, project_metadata, csv_metadata):
        project_norm = self._normalized_metadata(project_metadata)
        csv_norm = self._normalized_metadata(csv_metadata)
        lines = []
        for key in sorted(set(project_norm) | set(csv_norm)):
            if project_norm.get(key) != csv_norm.get(key):
                lines.append(f"{key}: project={project_norm.get(key, '<missing>')} | csv={csv_norm.get(key, '<missing>')}")
        return "\n".join(lines[:8])

    def _ask_metadata_source(self, project_metadata, csv_metadata, csv_path):
        try:
            diff_text = self._summarize_metadata_difference(project_metadata, csv_metadata) or "Values differ."
            msg = (
                "Project JSON and CSV metadata do not match.\n\n"
                f"CSV: {os.path.basename(csv_path)}\n\n"
                f"{diff_text}\n\n"
                "Use the CSV metadata?\n\n"
                "Yes = CSV\nNo = Project JSON"
            )
            return 'csv' if messagebox.askyesno("Project Metadata Mismatch", msg) else 'project'
        except KeyboardInterrupt:
            print("Metadata mismatch popup was interrupted; keeping project JSON metadata.")
            return 'project'
        except Exception as e:
            print(f"Warning: Could not show metadata mismatch popup: {e}")
            return 'project'

    def _reconcile_project_folder_metadata(self, project_exists, notify=True):
        csv_path = self._find_metadata_csv_path()
        project_metadata = self._metadata_from_current_project() if project_exists else {}
        csv_metadata = self._metadata_from_csv(csv_path) if csv_path else {}
        project_has_metadata = self._metadata_has_reference_points(project_metadata)
        csv_has_metadata = self._metadata_has_reference_points(csv_metadata)

        if project_has_metadata and csv_has_metadata:
            if self._normalized_metadata(project_metadata) != self._normalized_metadata(csv_metadata):
                source = self._ask_metadata_source(project_metadata, csv_metadata, csv_path) if notify else 'project'
                if source == 'csv':
                    self._apply_reference_metadata(csv_metadata)
                    if notify:
                        messagebox.showinfo("Metadata Source", "Using CSV metadata as the reference for this project folder.")
                else:
                    if notify:
                        messagebox.showinfo("Metadata Source", "Using project JSON metadata as the reference for this project folder.")
            return

        if csv_has_metadata and not project_has_metadata:
            self._apply_reference_metadata(csv_metadata)
            if notify:
                if not project_exists:
                    messagebox.showinfo("Metadata Source", "Project file is missing, so the CSV metadata is being used as the reference.")
                else:
                    messagebox.showinfo("Metadata Source", "Project file metadata is missing, so the CSV metadata is being used as the reference.")
            return

        if project_has_metadata and not csv_has_metadata:
            if notify:
                if not csv_path:
                    messagebox.showinfo("Metadata Source", "CSV file is missing, so the project JSON metadata is being used as the reference.")
                else:
                    messagebox.showinfo("Metadata Source", "CSV metadata is missing, so the project JSON metadata is being used as the reference.")

    def _get_project_named_points_m(self):
        points = []
        for name, prefix in [
            ('Tweeter', 'tw'),
            ('Reference Origin', 'ref_origin'),
        ]:
            try:
                r = self._grid_value(f'wp_{prefix}_r')
                phi = self._grid_value(f'wp_{prefix}_phi')
                z = self._grid_value(f'wp_{prefix}_z')
                if r and phi and z:
                    x_mm, y_mm, z_mm = self._cyl_mm_to_xyz_mm(r, phi, z)
                    points.append({'name': name, 'xyz': (x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0)})
            except Exception:
                pass

        for pos in getattr(self, 'user_positions', []) or []:
            try:
                name = str(pos.get('name', 'User Position')).strip() or 'User Position'
                x_mm, y_mm, z_mm = self._cyl_mm_to_xyz_mm(pos['r'], pos['phi'], pos['z'])
                points.append({'name': name, 'xyz': (x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0)})
            except Exception:
                continue
        return points

    def _get_stage2_acoustic_origin_point(self):
        """Load and select the requested frequency-dependent Stage 2 origin."""
        status_var = getattr(self, 'stage2_origin_status_var', None)
        show_var = self.stage5_vars.get('show_stage2_origin')
        if show_var is None or not show_var.get():
            if status_var is not None:
                status_var.set("")
            return None

        try:
            import numpy as np
            import schema
            from stage5_pressure_utils import nearest_acoustic_origin

            requested_hz = float(self.stage5_vars['stage2_origin_frequency_hz'].get())
            source_path = os.path.join(
                self.project_dir.get(),
                "outputs",
                f"{self.project_name.get()}_complex_data.npz",
            )
            with np.load(source_path, allow_pickle=True) as stage2_data:
                if schema.ORIGINS_MM not in stage2_data:
                    raise ValueError("Stage 2 origins have not been calculated yet.")
                actual_hz, origin_m = nearest_acoustic_origin(
                    stage2_data[schema.FREQS],
                    stage2_data[schema.ORIGINS_MM],
                    requested_hz,
                )

            # Make the control and persisted project setting describe the bin
            # actually used by the viewer, rather than the original request.
            actual_text = f"{actual_hz:.0f}"
            if self.stage5_vars['stage2_origin_frequency_hz'].get() != actual_text:
                self.stage5_vars['stage2_origin_frequency_hz'].set(actual_text)
            if status_var is not None:
                status_var.set("")
            return {
                'name': f"Stage 2 Origin ({actual_hz:g} Hz)",
                'xyz': tuple(float(value) for value in origin_m),
            }
        except (OSError, KeyError, TypeError, ValueError):
            if status_var is not None:
                status_var.set("unavailable")
            return None

    def _get_project_baffle_box_m(self):
        try:
            import numpy as np

            bl, tl, tr = self._get_project_baffle_front_corners_m()
            bl_xyz = np.array(bl, dtype=float)
            tl_xyz = np.array(tl, dtype=float)
            tr_xyz = np.array(tr, dtype=float)
            depth_x = max(0.0, self._mm_to_m(self.stage5_vars['dut_depth_x'].get()))

            width_vec = tr_xyz - tl_xyz
            height_vec = tl_xyz - bl_xyz
            width_m = max(float(np.linalg.norm(width_vec)), 0.001)
            height_m = max(float(np.linalg.norm(height_vec)), 0.001)
            normal = np.cross(width_vec, height_vec)
            normal_len = float(np.linalg.norm(normal))
            if normal_len <= 0.0:
                raise ValueError("Baffle waypoints do not define a plane.")

            br_xyz = bl_xyz + width_vec
            front = [bl_xyz, tl_xyz, tr_xyz, br_xyz]

            top_front_x = (tl_xyz[0] + tr_xyz[0]) / 2.0
            back_x = top_front_x - depth_x
            back = [np.array([back_x, pt[1], pt[2]], dtype=float) for pt in front]
            vertices = [tuple(pt) for pt in front + back]
            center = tuple(np.mean(np.array(vertices), axis=0))
            return (width_m, depth_x, height_m), center, vertices
        except Exception:
            try:
                depth_x = max(0.0, self._mm_to_m(self.stage5_vars.get('dut_depth_x', tk.StringVar(value="0.0")).get()))
            except Exception:
                depth_x = 0.0
            return (0.20, depth_x, 0.40), (0.0, 0.0, 0.0), None

    def _get_project_z_center_m(self):
        try:
            top_z = self._grid_value('wp_top_z')
            bot_z = self._grid_value('wp_bot_z')
            if top_z and bot_z:
                return ((float(top_z) + float(bot_z)) / 2.0) / 1000.0
        except Exception:
            pass
        return None

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
        canvas = tk.Canvas(self.tab_stage1, highlightthickness=0, bg=SETTINGS_CANVAS_BG)
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
        
        self.stage1_vars['fdw_rft_ms'] = self._add_form_entry(
            main_settings_frame,
            "Reflection Free Time (ms):",
            "5.0",
            "Reflection Free Time (ms): Defines the fixed window length at high frequencies.",
            button_text="Calculator",
            button_command=self._open_stage1_rft_calculator
        )
        self.stage1_vars['fdw_oct_res'] = self._add_form_entry(main_settings_frame, "Octave Resolution (1/x):", "12", "Target Octave Resolution: Sets the fractional octave smoothing (e.g., 12 for 1/12th oct).")
        self.stage1_vars['fdw_max_cap_ms'] = self._add_form_entry(main_settings_frame, "Max Window Cap (ms):", "200.0", "Optional cap (ms) on the maximum window length. Will limit oct res at LF.")
        self.stage1_vars['enable_auto_gain'] = self._add_checkbutton(main_settings_frame, "Enable Auto Gain", False, "Enable global normalization across all files in the batch.")
        self.stage1_vars['target_peak_db'] = self._add_form_entry(main_settings_frame, "Target Peak (dB):", "-3.0", "Target peak level (dBFS) for the loudest file in the set.", state_var=self.stage1_vars['enable_auto_gain'])
        
        # --- Advanced Settings ---
        self.btn_stage1_advanced = ttk.Button(main_container, text="Show Advanced Settings", command=self._toggle_stage1_advanced)
        self.btn_stage1_advanced.pack(side=tk.TOP, pady=10)

        self.stage1_adv_frame = ttk.LabelFrame(main_container, text="Advanced Settings", padding="10")
        
        self.stage1_vars['smoothing_oct_res'] = self._add_form_entry(self.stage1_adv_frame, "Smoothing Octave Res (1/x):", "Auto", "Auto = octave resolution x2 so that smoothing does not reduce resolution of initial FDW.")
        self.stage1_vars['fdw_alpha_hf'] = self._add_form_entry(self.stage1_adv_frame, "Alpha HF:", "0.2", "Taper alpha for High Frequencies (0.0=Rectangular, 1.0=Hann).")
        self.stage1_vars['fdw_alpha_lf'] = self._add_form_entry(self.stage1_adv_frame, "Alpha LF:", "1.0", "Taper alpha for Low Frequencies.")
        self.stage1_vars['fdw_windows_per_oct'] = self._add_form_entry(self.stage1_adv_frame, "Windows per Octave:", "3", "Windows per octave. Interpolation is performed in complex domain between windows.")
        self.stage1_vars['peak_detect_threshold_db'] = self._add_form_entry(self.stage1_adv_frame, "Peak Detect Threshold (dB):", "-12.0", "Peak detection finds loudest peak, then searches for earlier peaks above this threshold. A reflection may be louder than the true direct sound peak.")

        stage1_debug_frame = ttk.LabelFrame(self.stage1_adv_frame, text="Debug / Inspection", padding="10")
        stage1_debug_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        self.stage1_vars['enable_smoothing'] = self._add_checkbutton(stage1_debug_frame, "Enable Smoothing", True, "Enable or disable complex smoothing.")
        self.stage1_vars['keep_raw_and_smoothed'] = self._add_checkbutton(stage1_debug_frame, "Keep Raw & Smoothed", False, "Save both files if True.")

        # --- Button ---
        self.btn_stage1_run = ttk.Button(main_container, text="Run Stage 1", command=self._action_run_stage1)
        self.btn_stage1_run.pack(side=tk.TOP, pady=20)
        stage1_folder_note = ttk.Label(
            main_container,
            text="Note: Stage 1 expects IR WAV files in project_folder/measurement_set or project_folder/recordings.",
            font=("Arial", 9, "italic"),
            justify=tk.LEFT
        )
        stage1_folder_note.pack(side=tk.TOP, anchor=tk.W, fill=tk.X, padx=10, pady=(12, 10))
        main_container.bind("<Configure>", lambda e, lbl=stage1_folder_note: lbl.configure(wraplength=max(240, e.width - 44)), add="+")
        

    def _toggle_stage1_advanced(self):
        if self.stage1_adv_frame.winfo_ismapped():
            self.stage1_adv_frame.pack_forget()
            self.btn_stage1_advanced.config(text="Show Advanced Settings")
            self.stage1_canvas.yview_moveto(0)
        else:
            self.stage1_adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_stage1_run)
            self.btn_stage1_advanced.config(text="Hide Advanced Settings")

    def _open_stage1_rft_calculator(self):
        try:
            from rft_calculator import RFTCalculatorWindow

            def apply_rft(value):
                self.stage1_vars['fdw_rft_ms'].set(value)

            RFTCalculatorWindow(
                parent=self,
                initial_rft_ms=self.stage1_vars['fdw_rft_ms'].get(),
                on_apply=apply_rft
            )
        except Exception as exc:
            messagebox.showerror("RFT Calculator", f"Could not open RFT calculator:\n{exc}")
            
    def _update_cli(self):
        self._process_stage_job_events()

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

    def _stage_run_buttons(self):
        names = (
            'btn_stage1_run', 'btn_stage2_run', 'btn_stage3_run',
            'btn_stage4_run', 'btn_stage5_live', 'btn_stage5_run'
        )
        return [getattr(self, name) for name in names if hasattr(self, name)]

    def _set_stage_run_buttons(self, state):
        for button in self._stage_run_buttons():
            button.config(state=state)

    def _start_stage_job(self, stage_name, worker, on_success=None):
        if self.active_stage_job is not None:
            messagebox.showwarning(
                "Processing Busy",
                f"{self.active_stage_job} is still running. Wait for it to finish before starting {stage_name}."
            )
            return False

        # Do not let a persistent preview pool compete with a full processing
        # stage for the same CPU cores.
        if self.stage5_live_window is not None:
            self._close_stage5_live_preview(wait_for_pool=True)

        self.active_stage_job = stage_name
        self._set_stage_run_buttons(tk.DISABLED)
        self.status_var.set(f"{stage_name} running...")
        self.cli_text.config(state=tk.NORMAL)
        self.cli_text.insert(tk.END, f"\n--- Starting {stage_name} ---\n")
        self.cli_text.config(state=tk.DISABLED)

        def run_job():
            gc_was_enabled = gc.isenabled()
            if gc_was_enabled:
                gc.disable()
            try:
                result = worker()
                self.stage_job_queue.put(("success", stage_name, result, on_success, gc_was_enabled))
            except Exception as exc:
                self.stage_job_queue.put(("error", stage_name, exc, traceback.format_exc(), gc_was_enabled))

        threading.Thread(target=run_job, name=f"{stage_name}-worker", daemon=True).start()
        return True

    def _process_stage_job_events(self):
        while True:
            try:
                event = self.stage_job_queue.get_nowait()
            except queue.Empty:
                return

            event_type, stage_name, payload, callback, gc_was_enabled = event
            try:
                if gc_was_enabled:
                    gc.enable()
                gc.collect()
                if event_type == "success":
                    if callback is not None:
                        callback(payload)
                else:
                    print(f"Error during {stage_name}: {payload}")
                    if callback:
                        print(callback)
            except Exception as exc:
                print(f"Error finalizing {stage_name} on the GUI thread: {exc}")
                print(traceback.format_exc())
            finally:
                self.active_stage_job = None
                self._set_stage_run_buttons(tk.NORMAL)
                self.status_var.set("Ready.")

    def _action_run_stage1(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Run Stage 1")
        try:
            self._save_project_if_not_exists()
            proj_dir = self.project_dir.get()
            input_dir = self._find_stage1_ir_dir(proj_dir)
            out_dir = os.path.join(proj_dir, "outputs")
            output_filename = f"{self.project_name.get()}_complex_data.npz"

            rft_ms = float(self.stage1_vars['fdw_rft_ms'].get())
            oct_res = float(self.stage1_vars['fdw_oct_res'].get())
            enable_smoothing = self.stage1_vars['enable_smoothing'].get()
            smooth_res_str = self.stage1_vars['smoothing_oct_res'].get().strip().lower()
            if enable_smoothing:
                if smooth_res_str == 'auto' or not smooth_res_str:
                    smooth_res = oct_res * 2
                else:
                    try:
                        smooth_res = float(smooth_res_str)
                    except ValueError:
                        print("Warning: Invalid smoothing resolution entered, falling back to Auto (Octave Res x2).")
                        smooth_res = oct_res * 2
            else:
                smooth_res = oct_res * 2

            settings = {
                'input_dir': input_dir,
                'out_dir': out_dir,
                'output_filename': output_filename,
                'fdw_rft_ms': rft_ms,
                'fdw_oct_res': oct_res,
                'fdw_max_cap_ms': float(self.stage1_vars['fdw_max_cap_ms'].get()),
                'enable_auto_gain': self.stage1_vars['enable_auto_gain'].get(),
                'target_peak_db': float(self.stage1_vars['target_peak_db'].get()),
                'enable_smoothing': enable_smoothing,
                'smoothing_oct_res': smooth_res,
                'keep_raw_and_smoothed': self.stage1_vars['keep_raw_and_smoothed'].get(),
                'fdw_alpha_hf': float(self.stage1_vars['fdw_alpha_hf'].get()),
                'fdw_alpha_lf': float(self.stage1_vars['fdw_alpha_lf'].get()),
                'fdw_windows_per_oct': int(self.stage1_vars['fdw_windows_per_oct'].get()),
                'peak_detect_threshold_db': float(self.stage1_vars['peak_detect_threshold_db'].get()),
                'use_process_pool': True,
            }
        except Exception as exc:
            messagebox.showerror("Stage 1 Settings", f"Could not start Stage 1:\n{exc}")
            return

        self._start_stage_job(
            "Stage 1",
            lambda: self._run_stage1_job(settings),
            self._finish_stage1_job
        )

    @staticmethod
    def _run_stage1_job(settings):
        print(f"Stage 1 using IR WAV folder: {settings['input_dir']}")
        results = fdwsmooth(
                input_dir=settings['input_dir'],
                out_dir=settings['out_dir'],
                output_filename=settings['output_filename'],
                fdw_rft_ms=settings['fdw_rft_ms'],
                fdw_oct_res=settings['fdw_oct_res'],
                fdw_max_cap_ms=settings['fdw_max_cap_ms'],
                enable_smoothing=settings['enable_smoothing'],
                smoothing_oct_res=settings['smoothing_oct_res'],
                show_plot=False,  # Prevent plotting in the background thread
                save_to_disk=True,
                fdw_alpha_hf=settings['fdw_alpha_hf'],
                fdw_alpha_lf=settings['fdw_alpha_lf'],
                fdw_windows_per_oct=settings['fdw_windows_per_oct'],
                peak_detect_threshold_db=settings['peak_detect_threshold_db'],
                enable_auto_gain=settings['enable_auto_gain'],
                target_peak_db=settings['target_peak_db'],
                keep_raw_and_smoothed=settings['keep_raw_and_smoothed'],
                use_process_pool=settings['use_process_pool']
            )
        return settings, results

    def _finish_stage1_job(self, payload):
        settings, results = payload
        if results:
            freqs, results_raw, results_smooth, meta = results
            n_fft = (len(freqs) - 1) * 2
            fs_common = int(round((freqs[1] - freqs[0]) * n_fft))
            if not settings['enable_smoothing']:
                plot_data, plot_smooth = results_raw, None
            elif settings['keep_raw_and_smoothed']:
                plot_data, plot_smooth = results_raw, results_smooth
            else:
                plot_data, plot_smooth = results_smooth, None

            from viewers import FDWViewer
            print("Opening Viewer...")
            self.stage1_ui_instance = FDWViewer(
                freqs, plot_data, meta, fs_common, settings['input_dir'], n_fft,
                data_dict_smooth=plot_smooth,
                fdw_rft_ms=settings['fdw_rft_ms']
            )
            self._seed_stage3_start_from_npz(force=True)
        print("Stage 1 completed successfully.")

    def _seed_stage3_start_from_npz(self, force=False):
        if not hasattr(self, 'stage3_vars') or 'freq_start_hz' not in self.stage3_vars:
            return
        current = self.stage3_vars['freq_start_hz'].get().strip()
        if not force and current:
            return
        npz_path = os.path.join(
            self.project_dir.get(),
            "outputs",
            f"{self.project_name.get()}_complex_data.npz"
        )
        if not os.path.exists(npz_path):
            return
        try:
            from utils import load_and_parse_npz
            parsed = load_and_parse_npz(npz_path)
            value = parsed.get('stage3_rft_lower_hz')
            if value is not None:
                import math
                rounded_hz = max(0.0, math.ceil(float(value) / 1000.0) * 1000.0)
                self.stage3_vars['freq_start_hz'].set(f"{rounded_hz:.1f}")
        except Exception as e:
            print(f"Warning: Could not seed Stage 3 start frequency from Stage 1 metadata: {e}")

    def _build_stage2_ui(self):
        canvas = tk.Canvas(self.tab_stage2, highlightthickness=0, bg=SETTINGS_CANVAS_BG)
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
        self.btn_stage2_advanced = ttk.Button(main_container, text="Show Advanced Settings", command=self._toggle_stage2_advanced)
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
            self.btn_stage2_advanced.config(text="Show Advanced Settings")
            self.stage2_canvas.yview_moveto(0)
        else:
            self.stage2_adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_stage2_run)
            self.btn_stage2_advanced.config(text="Hide Advanced Settings")

    def _action_run_stage2(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Run Stage 2")
        try:
            self._save_project_if_not_exists()
            proj_dir = self.project_dir.get()
            input_dir = os.path.join(proj_dir, "outputs")
            input_filename = f"{self.project_name.get()}_complex_data.npz"
            output_filename = f"{self.project_name.get()}_complex_data.npz"

            def parse_bounds(b_str):
                parts = b_str.split(',')
                return (float(parts[0]), float(parts[1]))

            speed_var = getattr(self, 'global_vars', {}).get('speed_of_sound')
            manual_speed_var = getattr(self, 'global_vars', {}).get('enable_manual_speed_of_sound')
            use_manual_speed = bool(manual_speed_var.get()) if manual_speed_var is not None else False
            selected_speed = float(speed_var.get()) if use_manual_speed and speed_var is not None else 343.0

            settings = {
                'input_dir': input_dir,
                'input_filename': input_filename,
                'output_filename': output_filename,
                'tweeter_coords': (
                    float(self.stage2_vars['tweeter_x'].get()),
                    float(self.stage2_vars['tweeter_y'].get()),
                    float(self.stage2_vars['tweeter_z'].get())
                ),
                'octave_resolution': 1.0 / float(self.stage2_vars['octave_resolution'].get()),
                'freq_start_hz': float(self.stage2_vars['freq_start_hz'].get()),
                'freq_end_hz': float(self.stage2_vars['freq_end_hz'].get()),
                'x_bounds': parse_bounds(self.stage2_vars['x_bounds'].get()),
                'y_bounds': parse_bounds(self.stage2_vars['y_bounds'].get()),
                'z_bounds': parse_bounds(self.stage2_vars['z_bounds'].get()),
                'grid_res_mm': float(self.stage2_vars['grid_res_mm'].get()),
                'target_n_max_origins': int(self.stage2_vars['target_n_max_origins'].get()),
                'initial_simplex_step': float(self.stage2_vars['initial_simplex_step'].get()),
                'max_iterations': int(self.stage2_vars['max_iterations'].get()),
                'plot_results': self.stage2_vars['plot_results_origins'].get(),
                'selected_speed': selected_speed,
                'use_manual_speed': use_manual_speed,
            }
        except Exception as exc:
            messagebox.showerror("Stage 2 Settings", f"Could not start Stage 2:\n{exc}")
            return

        self._start_stage_job(
            "Stage 2",
            lambda: self._run_stage2_job(settings),
            self._finish_stage2_job
        )

    @staticmethod
    def _run_stage2_job(settings):
        from stage2_centre_origin import run_origin_search

        result = run_origin_search(
                input_dir_origins=settings['input_dir'],
                input_filename_origins=settings['input_filename'],
                output_filename_origins=settings['output_filename'],
                tweeter_coords_mm=settings['tweeter_coords'],
                octave_resolution=settings['octave_resolution'],
                freq_start_hz=settings['freq_start_hz'],
                freq_end_hz=settings['freq_end_hz'],
                initial_simplex_step=settings['initial_simplex_step'],
                max_iterations=settings['max_iterations'],
                x_bounds=settings['x_bounds'],
                y_bounds=settings['y_bounds'],
                z_bounds=settings['z_bounds'],
                grid_res_mm=settings['grid_res_mm'],
                target_n_max_origins=settings['target_n_max_origins'],
                manual_order_table=None,
                save_to_disk=True,
                plot_results_origins=False,
                speed_of_sound=settings['selected_speed'],
                optimize_speed_of_sound=not settings['use_manual_speed'],
                use_process_pool=True,
                return_state=True
            )
        return settings, result

    def _finish_stage2_job(self, payload):
        settings, result = payload
        if result is None:
            return

        from stage2_centre_origin import export_interpolated_origins

        sweep_results, f_all, keys, d_dict, geom, cfg, data = result
        selected_speed = float(cfg.get('speed_of_sound', 343.0))

        def do_save(reopen_after=False):
            nonlocal data
            history_freq, history_search_x, history_search_y, history_search_z = [], [], [], []
            for f_hz in sorted(sweep_results.keys()):
                row = sweep_results[f_hz]
                if row['final_c'] is not None:
                    history_freq.append(f_hz)
                    history_search_x.append(row['final_c'][0])
                    history_search_y.append(row['final_c'][1])
                    history_search_z.append(row['final_c'][2])

            origins_full = export_interpolated_origins(
                history_freq, history_search_x, history_search_y, history_search_z,
                f_all, data, settings['input_dir'], settings['output_filename'], True,
                speed_of_sound=selected_speed
            )
            if reopen_after and origins_full is not None:
                import numpy as np
                data = np.load(
                    os.path.join(settings['input_dir'], settings['output_filename']),
                    allow_pickle=True
                )
            print("Stage 2 completed successfully.")

        if settings['plot_results']:
            do_save(reopen_after=True)
            print("\nOpening Validation UI...")
            from viewers import ValidationUI
            save_path = os.path.splitext(
                os.path.join(settings['input_dir'], settings['output_filename'])
            )[0] + "_origins.png"
            self.stage2_ui_instance = ValidationUI(
                sweep_results, f_all, keys, d_dict, geom, cfg, save_path=save_path
            )

            def wait_for_ui():
                if plt.fignum_exists(self.stage2_ui_instance.view.fig.number):
                    self.after(200, wait_for_ui)
                elif self.stage2_ui_instance.accepted:
                    print("Saving adjusted results...")
                    do_save()
                else:
                    if hasattr(data, "close"):
                        data.close()
                    print("Validation UI closed.")

            wait_for_ui()
        else:
            do_save()

    def _build_stage3_ui(self):
        canvas = tk.Canvas(self.tab_stage3, highlightthickness=0, bg=SETTINGS_CANVAS_BG)
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

        # --- Main Settings ---
        main_settings_frame = ttk.LabelFrame(main_container, text="Main Settings", padding="10")
        main_settings_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.stage3_vars['freq_end_hz'] = self._add_form_entry(
            main_settings_frame, "Upper Test Range (Hz):", "20000.0", STAGE3_UPPER_RANGE_HELP,
            button_text="?", button_command=lambda: self._show_stage3_range_help("Upper test range", STAGE3_UPPER_RANGE_HELP))

        # --- Advanced Settings ---
        self.btn_stage3_advanced = ttk.Button(main_container, text="Show Advanced Settings", command=self._toggle_stage3_advanced)
        self.btn_stage3_advanced.pack(side=tk.TOP, pady=10)

        self.stage3_adv_frame = ttk.LabelFrame(main_container, text="Advanced Settings", padding="10")

        self.stage3_vars['freq_start_hz'] = self._add_form_entry(
            self.stage3_adv_frame, "Lower Test Range (Hz):", "10000.0", STAGE3_LOWER_RANGE_HELP,
            button_text="?", button_command=lambda: self._show_stage3_range_help("Lower test range", STAGE3_LOWER_RANGE_HELP))
        
        self.stage3_vars['test_order_range'] = self._add_form_entry(self.stage3_adv_frame, "Test Order Range (min, max):", "2, 15", "Range of orders N to test.")
        self.stage3_vars['octave_resolution'] = self._add_form_entry(
            self.stage3_adv_frame, "Octave Resolution (1/x; 0 = all bins):", "12",
            "Test frequencies at 1/x-octave spacing: 12 = 1/12 octave, 24 = 1/24 octave. "
            "0 tests every positive frequency bin in the range. Finer spacing takes longer. "
            "All Stage 3 diagnostics use these frequencies.")
        self.stage3_vars['spl_sphere_points'] = self._add_form_entry(self.stage3_adv_frame, "Directivity-change sphere points:", "1000",
            "Approximately uniform directions. More points can catch narrower directivity features but take longer.")
        self.stage3_vars['spl_radius_m'] = self._add_form_entry(self.stage3_adv_frame, "Directivity-change sphere radius (m):", "1.0",
            "Fixed sphere centred on the measurement coordinate origin. It should enclose the source.")


        # --- Button ---
        self.btn_stage3_run = ttk.Button(main_container, text="Run Stage 3", command=self._action_run_stage3)
        self.btn_stage3_run.pack(side=tk.TOP, pady=20)

    def _toggle_stage3_advanced(self):
        if self.stage3_adv_frame.winfo_ismapped():
            self.stage3_adv_frame.pack_forget()
            self.btn_stage3_advanced.config(text="Show Advanced Settings")
            self.stage3_canvas.yview_moveto(0)
        else:
            self.stage3_adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_stage3_run)
            self.btn_stage3_advanced.config(text="Hide Advanced Settings")

    def _action_run_stage3(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Run Stage 3")
        try:
            self._save_project_if_not_exists()
            proj_dir = self.project_dir.get()
            input_dir = os.path.join(proj_dir, "outputs")
            input_filename = f"{self.project_name.get()}_complex_data_centered.npz"
            if not os.path.exists(os.path.join(input_dir, input_filename)):
                input_filename = f"{self.project_name.get()}_complex_data.npz"

            def parse_bounds_int(b_str):
                parts = b_str.split(',')
                return (int(parts[0]), int(parts[1]))

            settings = {
                'input_dir': input_dir,
                'input_filename': input_filename,
                'order_range': parse_bounds_int(self.stage3_vars['test_order_range'].get()),
                'octave_resolution': int(self.stage3_vars['octave_resolution'].get()),
                'freq_start_hz': float(self.stage3_vars['freq_start_hz'].get()),
                'freq_end_hz': float(self.stage3_vars['freq_end_hz'].get()),
                'spl_sphere_points': int(self.stage3_vars['spl_sphere_points'].get()),
                'spl_radius_m': float(self.stage3_vars['spl_radius_m'].get()),
            }
            if settings['octave_resolution'] < 0:
                raise ValueError("Octave resolution must be a nonnegative integer (0 = all bins).")
        except Exception as exc:
            messagebox.showerror("Stage 3 Settings", f"Could not start Stage 3:\n{exc}")
            return

        self._start_stage_job(
            "Stage 3",
            lambda: self._run_stage3_job(settings),
            self._finish_stage3_job
        )

    def _run_stage3_job(self, settings):
        from stage3_optimize_she_settings import run_open_branch_optimizer
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        import multiprocessing

        if getattr(self, '_stage3_pool', None) is None:
            self._stage3_pool = ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context('spawn'))
            print('Stage 3: created session process pool; workers will be reused.')
        else:
            print('Stage 3: reusing session process pool.')

        try:
            return run_open_branch_optimizer(
                input_dir_opti=settings['input_dir'],
                input_filename_opti=settings['input_filename'],
                test_order_range=settings['order_range'],
                octave_resolution=settings.get('octave_resolution', 12),
                spl_change_enabled=True,
                spl_sphere_points=settings.get('spl_sphere_points', 1000),
                spl_radius_m=settings.get('spl_radius_m', 1.0),
                spl_floor_db=-40.0,
                freq_start_hz=settings['freq_start_hz'],
                freq_end_hz=settings['freq_end_hz'],
                test_start_db_range=(-20.0, -60.0),
                test_lambda_range=(0.0000001, 0.01),
                test_db_transition_span=20.0,
                use_optimized_origins=True,
                speed_of_sound=343.0,
                kr_offset=2.0,
                use_process_pool=True,
                process_pool=self._stage3_pool,
            )
        except BrokenProcessPool:
            self._stage3_pool.shutdown(wait=False, cancel_futures=True)
            self._stage3_pool = None
            raise

    def _finish_stage3_job(self, optimizer_result):
        if optimizer_result.get('below_rft', optimizer_result.get('tail_only')):
            self.stage3_vars['freq_start_hz'].set(f"{optimizer_result['test_band_hz'][0]:g}")
        self._show_stage3_choice_popup(optimizer_result)
        print("Stage 3 completed successfully. Review the recommended order before sending it to Stage 4.")

    def _show_stage3_help(self, parent):
        from stage3_optimize_she_settings import STAGE3_CHOICE_HELP

        self._show_stage3_help_window(parent, "Choosing your model's level of detail",
                                     "A guide to the graphs, frequency band and order choices",
                                     STAGE3_CHOICE_HELP)

    def _show_stage3_range_help(self, title, content):
        self._show_stage3_help_window(self.tab_stage3, title,
                                     "Choosing the Stage 3 test frequency band",
                                     title + "\n\n" + content, height=420)

    def _show_stage3_help_window(self, parent, title, subtitle, content, height=790):
        previous_grab = parent.grab_current()
        help_window = tk.Toplevel(parent)
        help_window.title("Stage 3 guide - " + title)
        help_window.transient(parent.winfo_toplevel())
        help_window.configure(bg="#f4f7fb")
        height = min(height, help_window.winfo_screenheight() - 100)
        width = min(760, help_window.winfo_screenwidth() - 80)
        help_window.geometry(f"{width}x{height}")
        help_window.minsize(min(540, width), min(440, height))

        header = tk.Frame(help_window, bg="#e8f0fa", padx=24, pady=18)
        header.pack(fill=tk.X)
        tk.Label(header, text=title, font=("Segoe UI", 18, "bold"),
                 bg="#e8f0fa", fg="#203b59", anchor="w").pack(fill=tk.X)
        tk.Label(header, text=subtitle,
                 font=("Segoe UI", 10), bg="#e8f0fa", fg="#44617e", anchor="w").pack(fill=tk.X, pady=(6, 0))

        body = tk.Frame(help_window, bg="white")
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)
        scrollbar = ttk.Scrollbar(body, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text = tk.Text(body, wrap=tk.WORD, font=("Segoe UI", 11), bg="white", fg="#263445",
                       relief=tk.FLAT, borderwidth=0, padx=20, pady=16, cursor="arrow",
                       spacing1=3, spacing3=10, yscrollcommand=scrollbar.set)
        text.pack(fill=tk.BOTH, expand=True)
        scrollbar.configure(command=text.yview)
        text.tag_configure("heading", font=("Segoe UI", 13, "bold"), foreground="#203b59", spacing1=16, spacing3=8)
        text.tag_configure("choice", font=("Segoe UI", 11, "bold"), foreground="#315e91")
        text.tag_configure("tip", background="#edf5fc", foreground="#23496d", lmargin1=10, lmargin2=10,
                           rmargin=10, spacing1=10, spacing3=14)
        sections = content.strip().split("\n\n")
        for paragraph in sections[1:]:
            lines = paragraph.splitlines()
            if lines[0] in ("Choosing the test frequency band", "Top graph: Solve Stability",
                            "Bottom graph: Sound Power Discarded", "The three choices", "How we recommend an order", "Incremental SPL change graph",
                            "Below the reflection-free range"):
                text.insert(tk.END, lines[0] + "\n", "heading")
                if len(lines) > 1:
                    text.insert(tk.END, " ".join(lines[1:]) + "\n\n")
            elif paragraph.startswith(("Roll-off knee:", "Soft -25 dB tail:", "Directivity change:")):
                title, detail = " ".join(lines).split(":", 1)
                text.insert(tk.END, title + "\n", "choice")
                text.insert(tk.END, detail.strip() + "\n\n")
            else:
                text.insert(tk.END, " ".join(lines) + "\n\n", "tip" if paragraph.startswith("Select a choice") else ())
        text.configure(state=tk.DISABLED)

        footer = tk.Frame(help_window, bg="#f4f7fb", padx=20, pady=12)
        footer.pack(fill=tk.X)
        tk.Label(footer, text="Scroll to explore • Your current selection stays unchanged",
                 bg="#f4f7fb", fg="#53677b", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        def close_help():
            help_window.destroy()
            if previous_grab is not None and previous_grab.winfo_exists():
                previous_grab.grab_set()
                previous_grab.focus_set()
        ttk.Button(footer, text="Got it", command=close_help).pack(side=tk.RIGHT)
        help_window.protocol("WM_DELETE_WINDOW", close_help)
        help_window.bind("<Escape>", lambda event: close_help())
        help_window.grab_set()
        text.focus_set()

    def _show_stage3_choice_popup(self, optimizer_result):
        if isinstance(optimizer_result, tuple):
            target_n_max, noise_floor_start_db, noise_floor_max_db, max_lambda = optimizer_result
            options = {
                "best_sfs": {
                    "label": "Recommended Order N",
                    "n": target_n_max,
                    "st": noise_floor_start_db,
                    "mx": noise_floor_max_db,
                    "lam": max_lambda,
                    "ratio": None,
                    "err": None,
                    "warning": "",
                    "reason": "Legacy Stage 3 result."
                }
            }
        else:
            options = optimizer_result.get("options", {})

        top = tk.Toplevel(self)
        top.title("Stage 3 Recommended Order N")
        top.geometry("1280x960")
        top.transient(self)
        top.grab_set()

        footer = ttk.Frame(top, padding=12)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        header = ttk.Frame(top, padding=(12, 8))
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text='Stage 3 - Recommended Max Order for Stage 4 Solve',
                  font=('Arial', 14, 'bold')).pack(anchor=tk.CENTER, pady=(0, 8))
        band = optimizer_result.get('spl_change', {}).get('test_band_hz') if isinstance(optimizer_result, dict) and optimizer_result.get('spl_change') else None
        if not band and isinstance(optimizer_result, dict):
            frequencies = optimizer_result.get('step1', {}).get('sample_frequencies_hz', [])
            if len(frequencies):
                band = (min(frequencies), max(frequencies))
        if band:
            ttk.Label(header, text=f"Test range: {band[0]:g} - {band[1]:g} Hz",
                      font=('Arial', 12, 'bold')).pack(anchor=tk.CENTER)
        ttk.Label(header, text='Higher orders describe more detail; too high can introduce errors or spurious detail.',
                  wraplength=1000, justify=tk.CENTER).pack(anchor=tk.CENTER, pady=(6, 0))
        content = ttk.Frame(top)
        content.pack(fill=tk.BOTH, expand=True)
        scroll = ScrollableFrame(content)
        scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame = scroll.scrollable_frame
        graph_area = ttk.Frame(frame)
        graph_area.columnconfigure(0, weight=1, uniform='graphs')
        graph_area.columnconfigure(1, weight=1, uniform='graphs')
        left_graphs = ttk.Frame(graph_area)
        left_graphs.grid(row=0, column=0, sticky='nsew')
        sidebar = ttk.Frame(graph_area, padding=(8, 0, 0, 0))
        sidebar.grid(row=0, column=1, sticky='nsew')

        plot_fig = None
        spl_fig = None
        step1 = optimizer_result.get("step1", {}) if isinstance(optimizer_result, dict) else {}
        tail_only = optimizer_result.get('tail_only', False) if isinstance(optimizer_result, dict) else False
        plot_path = optimizer_result.get("plot_path", "") if isinstance(optimizer_result, dict) else ""
        recommended_key = optimizer_result.get("recommended_key", "recommended") if isinstance(optimizer_result, dict) else "best_sfs"
        if recommended_key not in options:
            recommended_key = "recommended" if "recommended" in options else next(iter(options), None)
        recommended = options.get(recommended_key, {'n': None})
        from stage3_optimize_she_settings import (STAGE3_CHOICE_HELP, STAGE3_CHOICE_STYLES,
                                                  stage3_order_choices, highlight_stage3_choices,
                                                  recommended_stage3_choice, format_tail_power, format_stage3_ratio_axis,
                                                  format_stage3_order_axis)
        options = dict(options)
        choice_var = tk.StringVar(value=next((key for key in STAGE3_CHOICE_STYLES
                                             if key in options and options[key]['n'] == recommended['n']), recommended_key))
        recommendation_note = optimizer_result.get("recommendation_note", recommended.get("reason", "")) if isinstance(optimizer_result, dict) else recommended.get("reason", "")
        sfs_rule_db = optimizer_result.get("sfs_ratio_rule_db", 20.0) if isinstance(optimizer_result, dict) else 20.0
        orders = step1.get("orders", [])
        ratios = step1.get("ratios", [])
        degree_power = step1.get("internal_degree_power_db")
        tail_power = step1.get("internal_tail_power_db")
        if tail_only:
            tail_power = degree_power = None
        active_tail = dict(power=tail_power, n=step1.get('tail_reference', {}).get('n'))
        spl_change = optimizer_result.get('spl_change') if isinstance(optimizer_result, dict) else None
        choice_artists = []
        if len(orders) > 0 and len(ratios) > 0:
            has_power = tail_only or tail_power is not None or degree_power is not None
            count = 1 + int(has_power)
            plot_fig, axes = plt.subplots(count, 1, figsize=(6, 3.8 * count), squeeze=False)
            ax_ratio = axes[0, 0]
            ax_power = axes[1, 0] if has_power else None
            if tail_power is not None:
                from stage3_optimize_she_settings import plot_internal_tail_power
                plot_internal_tail_power(ax_power, orders, tail_power, step1['tail_reference'])
                references = step1.get('tail_by_reference', {})
                if references:
                    reference_frame = ttk.Frame(frame)
                    reference_frame.pack(fill=tk.X, pady=(0, 6))
                    ttk.Label(reference_frame, text="Sound power reference:").pack(side=tk.LEFT)
                    default_n = step1['tail_reference']['n']
                    reference_choices = {}
                    for key, ref in references.items():
                        status = "automatic" if ref['n'] == default_n else "manual"
                        if not tail_only and ref['ratio'] <= float(sfs_rule_db):
                            status += "; below >20 dB threshold"
                        label = f"N={ref['n']} | Int/Ext {ref['ratio']:.2f} dB | {status}"
                        if tail_only:
                            status = step1['tail_reference'].get('label', 'provisional SPL reference, backed off one order') if ref['n'] == default_n else 'manual'
                            label = f"N={ref['n']} | {status} | Int/Ext not used"
                        reference_choices[label] = ref
                    default_label = next((label for label, ref in reference_choices.items() if ref['n'] == default_n),
                                         'No automatic reference - select manually')
                    reference_var = tk.StringVar(value=default_label)
                    reference_combo = ttk.Combobox(reference_frame, textvariable=reference_var,
                                                   values=list(reference_choices), state="readonly",
                                                   width=max(map(len, reference_choices)) + 2)
                    reference_combo.pack(side=tk.LEFT, padx=(8, 0))

                    def change_tail_reference(event=None):
                        selected = reference_choices[reference_var.get()]
                        active_tail.update(power=selected['power_db'], n=selected['n'])
                        metadata = dict(selected, fallback=step1['tail_reference']['fallback'],
                                        manual=selected['n'] != default_n, tail_only=tail_only,
                                        label=step1['tail_reference'].get('label', 'provisional SPL reference, backed off one order'),
                                        ratio_note=optimizer_result.get('ratio_note', 'Int/Ext shown for inspection; not used for selection.'))
                        ax_power.clear()
                        plot_internal_tail_power(ax_power, orders, selected['power_db'], metadata)
                        updated = stage3_order_choices(orders, ratios, step1['residuals'],
                                                      step1.get('rolloff_knee'), selected['power_db'], selected['n'], tail_only=tail_only)
                        for key in ('knee', 'tail'):
                            options.pop(key, None)
                        options.update(updated)
                        choice_var.set(recommended_stage3_choice(options) or '')
                        for artist in choice_artists:
                            artist.remove()
                        if ax_ratio is not None:
                            choice_artists[:] = highlight_stage3_choices(ax_ratio, options)
                        highlight_stage3_choices(ax_power, options, tail=True)
                        if ax_ratio is not None:
                            ax_ratio.legend(loc='best', fontsize=8)
                        ax_power.legend(loc='best', fontsize=8)
                        refresh_choices()
                        plot_fig.tight_layout()
                        canvas_plot.draw_idle()

                    reference_combo.bind('<<ComboboxSelected>>', change_tail_reference)
            elif degree_power is not None:
                from stage3_optimize_she_settings import plot_internal_degree_power
                plot_internal_degree_power(ax_power, orders, degree_power)
            elif tail_only:
                ax_power.set_title('Sound Power Discarded', fontsize=12, fontweight='bold')
                format_stage3_order_axis(ax_power)
                ax_power.set_ylabel('Discarded internal power (dB)')
                ax_power.set_xticks(orders)
                ax_power.set_xlim(min(orders) - .5, max(orders) + .5)
                ax_power.set_ylim(-40, 0)
                ax_power.grid(True, linestyle='--', alpha=.2)
            if ax_ratio is not None:
                ax_ratio.plot(orders, ratios, marker="o", linewidth=1.4, color="#4c78a8", label="_nolegend_")
                ax_ratio.axhline(
                    float(sfs_rule_db),
                    color="#59a14f",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                    label=f"{sfs_rule_db:.0f} dB quality threshold",
                )
                ax_ratio.set_xlabel("Order N")
                ax_ratio.set_ylabel("Int/Ext ratio (dB)")
                format_stage3_ratio_axis(ax_ratio, inspection_only=tail_only)
                ax_ratio.set_xticks(orders)
                ax_ratio.set_xlim(min(orders) - .5, max(orders) + .5)
                ax_ratio.grid(True, linestyle="--", alpha=0.35)

                marker_styles = STAGE3_CHOICE_STYLES if any(key in options for key in STAGE3_CHOICE_STYLES) else {
                    recommended_key: ("#e45756", "o", "Recommended")}
                choice_artists = []
                for key, (color, marker, label) in marker_styles.items():
                    if key != "knee":
                        continue
                    opt = options.get(key)
                    if opt and opt.get("n") is not None and opt.get("ratio") is not None:
                        choice_artists.append(ax_ratio.scatter(
                            [float(opt["n"])],
                            [float(opt["ratio"])],
                            s=95,
                            color=color,
                            marker=marker,
                            edgecolors="white",
                            linewidths=1.1,
                            zorder=5,
                            label=label,
                        ))

            if tail_power is not None:
                highlight_stage3_choices(ax_power, options, tail=True)
                ax_power.legend(loc='best', fontsize=8)

            if ax_ratio is not None:
                handles, labels = ax_ratio.get_legend_handles_labels()
                ax_ratio.legend(handles, labels, loc="best", fontsize=8)
            plot_fig.tight_layout()

            if spl_change is not None:
                from stage3_spl_change import plot_spl_changes
                displayed_spl = dict(spl_change, order_choices={
                    key: opt for key, opt in options.items() if key == 'spl'})
                spl_fig, ax_spl = plt.subplots(figsize=(6, 3.8))
                plot_spl_changes(ax_spl, displayed_spl)
                spl_fig.tight_layout()
                spl_canvas = FigureCanvasTkAgg(spl_fig, master=sidebar)
                spl_canvas.draw()
                spl_canvas.get_tk_widget().pack(fill=tk.X)
            if tail_only:
                unavailable_reason = (
                    'The upper test range is below the reflection-free (RFT) range.'
                    if optimizer_result.get('below_rft') else
                    'No tested order provides Int/Ext separation greater than 20 dB.')
                for ax in (ax_ratio, ax_power):
                    ax.set_facecolor('#eeeeee')
                    for artist in [*ax.lines, *ax.collections]:
                        artist.set_color('#999999')
                        artist.set_alpha(.3)
                    ax.tick_params(colors='#999999')
                    for spine in ax.spines.values():
                        spine.set_color('#bbbbbb')
                    for text in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]:
                        text.set_color('#999999')
                    if ax.get_legend() is not None:
                        ax.get_legend().set_visible(False)
                    ax.text(.5, .5, 'Unavailable for order selection\n' + unavailable_reason,
                            transform=ax.transAxes, ha='center', va='center', fontsize=10,
                            fontweight='bold', color='#666666', zorder=20,
                            bbox=dict(boxstyle='round,pad=.8', facecolor='#f3f3f3',
                                      edgecolor='#bbbbbb', alpha=.95))
            plot_fig.tight_layout()
            graph_area.pack(fill=tk.X)
            if has_power:
                from matplotlib.patches import Rectangle
                # Match the panel gutter between the two columns.
                rgb = tuple(v / 65535 for v in top.winfo_rgb(SETTINGS_CANVAS_BG))
                plot_fig.add_artist(Rectangle((0, .495), 1, .01,
                    transform=plot_fig.transFigure, facecolor=rgb, edgecolor='none', zorder=10))
            canvas_plot = FigureCanvasTkAgg(plot_fig, master=left_graphs)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(fill=tk.X, pady=(0, 8))

        ratio_text = "n/a" if recommended.get("ratio") is None else f"{recommended['ratio']:.2f} dB"
        result_text = (
            f"{recommended.get('label', recommended_key)}: N={recommended['n']}, Int/Ext={ratio_text}\n"
            f"{recommendation_note}\n\n"
            f"Rule of thumb: an Int/Ext SFS ratio greater than {float(sfs_rule_db):.0f} dB has been found to produce acceptable results."
        )
        if recommended.get("warning"):
            result_text += f"\n\nWarning: {recommended['warning']}"
        choice_frame = ttk.LabelFrame(sidebar, text="Recommended Order N:", padding=6)
        choice_frame.pack(fill=tk.X, pady=(12, 6))

        use_button = None

        def refresh_choices():
            for child in choice_frame.winfo_children():
                child.destroy()
            active_recommendation = recommended_stage3_choice(options)
            if choice_var.get() not in options:
                choice_var.set(active_recommendation or '')
            grouped = {}
            for key in ('knee', 'tail', 'spl'):
                label = {'knee': 'Internal to External Ratio', 'tail': 'Sound Power Discarded',
                         'spl': 'Directivity Change'}[key]
                opt = options.get(key)
                if not opt:
                    reason = ('unavailable without usable separation' if tail_only and key != 'spl'
                              else 'no eligible candidate found')
                    if key == 'spl':
                        reason = optimizer_result.get('spl_candidate_note') or 'No valid directivity-change comparisons are available.'
                    ttk.Label(choice_frame, text=f"{label}: {reason}", wraplength=480,
                              foreground='#687078').pack(anchor=tk.W, pady=6)
                    continue
                grouped[key] = opt
                discarded = opt.get('tail_db', float('nan'))
                if (key == 'spl' and active_tail['power'] is not None
                        and active_tail['n'] is not None and opt['n'] < active_tail['n']
                        and opt['n'] in orders):
                    discarded = active_tail['power'][list(orders).index(opt['n'])]
                ratio = opt.get('ratio')
                row = ttk.Frame(choice_frame)
                row.pack(fill=tk.X)
                radio = ttk.Radiobutton(row, variable=choice_var, value=key, state='normal')
                radio.pack(side=tk.LEFT, anchor=tk.N, pady=8)
                details = ttk.Frame(row)
                details.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=8)
                choice_label = ttk.Label(details, text=f"Recommended by {label} (N={opt['n']})",
                                         font=('Segoe UI', 10, 'bold'), wraplength=450)
                choice_label.pack(anchor=tk.W)
                choice_label.bind('<Button-1>', lambda event, selected=key: choice_var.set(selected))
                specs = []
                if ratio is not None and ratio == ratio and not tail_only:
                    specs.append(f"Int / Ext Ratio: {ratio:.2f} dB")
                if discarded is not None and discarded == discarded and not tail_only:
                    specs.append(f"Sound Power Discarded: {discarded:.2f} dB")
                if specs:
                    metrics = ttk.Label(details, text='\n'.join(specs), wraplength=450)
                    metrics.pack(anchor=tk.W, padx=(12, 0), pady=(3, 0))
                    metrics.bind('<Button-1>', lambda event, selected=key: choice_var.set(selected))
            if not grouped:
                ttk.Label(choice_frame, text=("No valid directivity-change comparisons are available. Review the test band and order range." if tail_only else "No eligible choice. Check the test frequency range, windowing, measurement quality and grid coverage; re-measure if needed."),
                          wraplength=800).pack(anchor=tk.W)
            if use_button is not None:
                use_button.configure(state='normal' if grouped else 'disabled')

        refresh_choices()

        help_row = ttk.Frame(sidebar)
        help_row.pack(fill=tk.X, pady=10)
        ttk.Label(help_row, text='Help').pack(side=tk.LEFT)
        SpkrScannerApp._make_help_icon(self, help_row, lambda: self._show_stage3_help(top)).pack(side=tk.LEFT, padx=6)
        if plot_path:
            ttk.Label(sidebar, text=f'Plots saved to {os.path.dirname(os.path.abspath(plot_path))}',
                      wraplength=480).pack(anchor=tk.W, pady=4)

        note = optimizer_result.get("warning", "") if isinstance(optimizer_result, dict) else ""
        if note:
            ttk.Label(frame, text=note, foreground="orange", wraplength=720).pack(anchor=tk.W, pady=(8, 0))

        def send_choice():
            if choice_var.get() not in options:
                return
            opt = options[choice_var.get()]
            if 'target_n_max' in self.stage4_vars:
                self.stage4_vars['target_n_max'].set(str(opt['n']))
            print(f"Sent {opt.get('label', recommended_key)} (N={opt['n']}) to Stage 4.")
            close_popup()

        btn_frame = ttk.Frame(footer)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        use_button = ttk.Button(btn_frame, text="Use in Stage 4", command=send_choice)
        use_button.pack(side=tk.RIGHT, padx=5)
        use_button.configure(state='normal' if options else 'disabled')

        def close_popup():
            if plot_fig is not None:
                plt.close(plot_fig)
            if spl_fig is not None:
                plt.close(spl_fig)
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", close_popup)
        ttk.Button(btn_frame, text="Cancel", command=close_popup).pack(side=tk.RIGHT, padx=5)
        top.update_idletasks()
        required_height = frame.winfo_reqheight() + header.winfo_reqheight() + footer.winfo_reqheight() + 24
        available_height = top.winfo_screenheight() - 80
        top.geometry(f"{min(1280, top.winfo_screenwidth() - 40)}x{min(required_height, available_height)}")
        scrollbar = next(w for w in scroll.winfo_children() if isinstance(w, ttk.Scrollbar))

        def update_results_scrollbar(event=None):
            needs_scroll = frame.winfo_reqheight() > scroll.canvas.winfo_height()
            if needs_scroll and not scrollbar.winfo_manager():
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y, before=scroll.canvas)
            elif not needs_scroll and scrollbar.winfo_manager():
                scrollbar.pack_forget()
                scroll.canvas.yview_moveto(0)

        scroll.canvas.bind('<Configure>', update_results_scrollbar, add='+')
        top.after_idle(update_results_scrollbar)

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
        self.btn_stage4_advanced = ttk.Button(main_container, text="Show Advanced Settings", command=self._toggle_stage4_advanced)
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
        
        self.stage4_vars['noise_floor_start_db'] = self._add_form_entry(self.stage4_adv_frame, "Noise Floor Start (dB):", "-30.0", "Fixed default damping start.")
        self.stage4_vars['noise_floor_max_db'] = self._add_form_entry(self.stage4_adv_frame, "Noise Floor Max (dB):", "-40.0", "Fixed default point where damping hits MAX_LAMBDA.")
        self.stage4_vars['max_lambda'] = self._add_form_entry(self.stage4_adv_frame, "Max Lambda:", "0.00000100", "Fixed default maximum penalty applied to modes.")
        self.stage4_vars['use_optimized_origins'] = self._add_checkbutton(self.stage4_adv_frame, "Use Optimized Origins", True, "Essential for best fit.")
        
        # --- Button ---
        self.btn_stage4_run = ttk.Button(main_container, text="Run Stage 4", command=self._action_run_stage4)
        self.btn_stage4_run.pack(side=tk.TOP, pady=20)

    def _toggle_stage4_advanced(self):
        if self.stage4_adv_frame.winfo_ismapped():
            self.stage4_adv_frame.pack_forget()
            self.btn_stage4_advanced.config(text="Show Advanced Settings")
            self.stage4_canvas.yview_moveto(0)
        else:
            self.stage4_adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_stage4_run)
            self.btn_stage4_advanced.config(text="Hide Advanced Settings")

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
        try:
            self._save_project_if_not_exists()
            proj_dir = self.project_dir.get()
            project_name = self.project_name.get()
            input_dir = os.path.join(proj_dir, "outputs")
            input_filename = f"{project_name}_complex_data.npz"
            output_dir = os.path.join(proj_dir, "outputs", "coefficients")
            settings = {
                'project_name': project_name,
                'input_dir': input_dir,
                'input_filename': input_filename,
                'output_dir': output_dir,
                'output_filename': f"{project_name}_coefficients.h5",
                'target_n_max': int(self.stage4_vars['target_n_max'].get()),
                'kr_offset': float(self.stage4_vars['kr_offset'].get()),
                'use_manual_table': self.stage4_vars['use_manual_table'].get(),
                'noise_floor_start_db': float(self.stage4_vars['noise_floor_start_db'].get()),
                'noise_floor_max_db': float(self.stage4_vars['noise_floor_max_db'].get()),
                'max_lambda': float(self.stage4_vars['max_lambda'].get()),
                'use_optimized_origins': self.stage4_vars['use_optimized_origins'].get(),
                'manual_table': {
                    float(k): int(v)
                    for k, v in getattr(self, 'stage4_manual_table', {}).items()
                },
            }
        except Exception as exc:
            messagebox.showerror("Stage 4 Settings", f"Could not start Stage 4:\n{exc}")
            return

        self._start_stage_job(
            "Stage 4",
            lambda: self._run_stage4_job(settings),
            self._finish_stage4_job
        )

    @staticmethod
    def _run_stage4_job(settings):
        from stage4_run_she_solve import run_she_solve

        results = run_she_solve(
                input_filename_she=settings['input_filename'],
                output_filename_she=settings['output_filename'],
                input_dir_she=settings['input_dir'],
                output_dir_she=settings['output_dir'],
                target_n_max=settings['target_n_max'],
                use_manual_table=settings['use_manual_table'],
                manual_order_table=settings['manual_table'],
                noise_floor_start_db=settings['noise_floor_start_db'],
                noise_floor_max_db=settings['noise_floor_max_db'],
                max_lambda=settings['max_lambda'],
                condition_metrics=True,
                use_optimized_origins=settings['use_optimized_origins'],
                save_to_disk=True,
                speed_of_sound=343.0,
                kr_offset=settings['kr_offset'],
                jobs=None,
                show_plot=False,
                use_process_pool=True
            )
        return settings, results

    def _finish_stage4_job(self, payload):
        settings, results = payload
        if results:
            from viewers import plot_she_results

            save_prefix = os.path.join(
                settings['input_dir'], f"{settings['project_name']}_coefficients"
            )
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
                c_sound=results.get("speed_of_sound_mps", 343.0)
            )
        print("Stage 4 completed successfully.")

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
        if GUI_TOOLTIPS.get('output_dir'): ToolTip(out_dir_frame, GUI_TOOLTIPS['output_dir'])

        self.stage5_vars['frd_prefix'] = self._add_form_entry(main_settings_frame, "FRD Prefix:", self.project_name.get(), GUI_TOOLTIPS.get('frd_prefix'))
        self.project_name.trace_add("write", lambda *args: self.stage5_vars['frd_prefix'].set(self.project_name.get()))

        self.stage5_vars['frd_db_offset'] = self._add_form_entry(main_settings_frame, "FRD dB Offset:", "0.0", GUI_TOOLTIPS.get('frd_db_offset'))

        self.stage5_vars['generate_ir_files'] = self._add_checkbutton(
            main_settings_frame, "Generate IR Files (.wav)", False, GUI_TOOLTIPS.get('generate_ir_files')
        )

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
        self.stage5_vars['apply_mic_cal'] = self._add_checkbutton(chk_frame, "Apply Microphone Calibration", False, GUI_TOOLTIPS.get('apply_mic_cal'))
        
        combo_frame = ttk.Frame(cal_top_frame)
        combo_frame.pack(side=tk.LEFT, fill=tk.X)
        ttk.Label(combo_frame, text="Mode:").pack(side=tk.LEFT, padx=(10, 5))
        self.stage5_vars['mic_cal_mode'] = tk.StringVar(value="subtract")
        cb_mode = ttk.Combobox(combo_frame, textvariable=self.stage5_vars['mic_cal_mode'], values=["subtract", "add"], state="readonly", width=10)
        cb_mode.pack(side=tk.LEFT)
        if GUI_TOOLTIPS.get('mic_cal_mode'): ToolTip(cb_mode, GUI_TOOLTIPS['mic_cal_mode'])

        self.lbl_mic_cal_fallback = ttk.Label(combo_frame, text="")
        self.lbl_mic_cal_fallback.pack(side=tk.LEFT, padx=(10, 0))
        self.stage5_vars['mic_cal_file'].trace_add("write", self._check_mic_cal_status)

        # --- Reference Axis & Phase ---
        ref_axis_frame = ttk.LabelFrame(main_container, text="Reference Axis & Phase", padding="10")
        ref_axis_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        reference_geometry_frame = ttk.Frame(ref_axis_frame)
        reference_geometry_frame.pack(side=tk.TOP, fill=tk.X)

        offset_frame = ttk.Frame(reference_geometry_frame)
        offset_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(offset_frame, text="Mic Offset (mm):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.stage5_offset_entries = {}
        for column, axis in enumerate(('x', 'y', 'z'), start=1):
            axis_frame = ttk.Frame(offset_frame)
            axis_frame.grid(row=0, column=column, sticky=tk.N, padx=8)
            label = ttk.Label(axis_frame, text=f"{axis.upper()}:")
            label.grid(row=0, column=0, sticky=tk.E, padx=(0, 3))
            var_key = f'offset_mic_{axis}'
            value_var = tk.StringVar(value="0.0")
            self.stage5_vars[var_key] = value_var
            entry = ttk.Entry(axis_frame, textvariable=value_var, width=7)
            entry.grid(row=0, column=1, sticky=tk.W)
            entry.bind("<Up>", lambda event, key=var_key: self._jog_stage5_offset(key, 1))
            entry.bind("<Down>", lambda event, key=var_key: self._jog_stage5_offset(key, -1))
            entry.bind("<FocusOut>", self._schedule_update_stage5_preview)
            entry.bind("<Return>", self._schedule_update_stage5_preview)
            self.stage5_offset_entries[axis] = entry
            help_text = GUI_TOOLTIPS.get(var_key)
            if help_text:
                ToolTip(label, help_text)
                ToolTip(entry, help_text)

        step_frame = ttk.Frame(offset_frame)
        step_frame.grid(row=0, column=4, sticky=tk.N, padx=(18, 0))
        ttk.Label(step_frame, text="Jog step:").pack(side=tk.LEFT, padx=(0, 5))
        self.stage5_vars['offset_step_mm'] = tk.StringVar(value="10")
        step_combo = ttk.Combobox(
            step_frame,
            textvariable=self.stage5_vars['offset_step_mm'],
            values=("1", "10", "25", "50", "100"),
            state="readonly",
            width=6,
        )
        step_combo.pack(side=tk.LEFT)
        ttk.Label(step_frame, text="mm").pack(side=tk.LEFT, padx=(3, 0))

        jog_tip = ttk.Label(
            offset_frame,
            text="Tip: focus an X, Y, or Z value and press ↑ / ↓ to jog it.",
            font=("Arial", 8, "italic"),
            anchor=tk.W,
            justify=tk.LEFT,
        )
        jog_tip.grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(4, 0))

        zero_angle_frame = ttk.Frame(reference_geometry_frame)
        zero_angle_frame.pack(side=tk.TOP, fill=tk.X, pady=(7, 2))
        ttk.Label(zero_angle_frame, text="Zero Angle (deg):").pack(side=tk.LEFT, padx=(0, 10))
        self.stage5_vars['zero_theta_deg'] = self._add_labeled_entry(zero_angle_frame, "Theta:", "90.0", 5, GUI_TOOLTIPS.get('zero_theta_deg'))
        self.stage5_vars['zero_phi_deg'] = self._add_labeled_entry(zero_angle_frame, "Phi:", "0.0", 5, GUI_TOOLTIPS.get('zero_phi_deg'))

        phase_frame = ttk.Frame(ref_axis_frame)
        phase_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        ttk.Label(phase_frame, text="Subtract TOF Phase:").pack(side=tk.LEFT, padx=(0, 5))
        self.stage5_vars['subtract_tof'] = tk.StringVar(value="Ref Origin")
        cb_tof = ttk.Combobox(
            phase_frame,
            textvariable=self.stage5_vars['subtract_tof'],
            values=["Off", "Ref Origin", "Min Phase Ref", "IR Peak"],
            state="readonly",
            width=15,
        )
        cb_tof.pack(side=tk.LEFT)
        if GUI_TOOLTIPS.get('subtract_tof'):
            ToolTip(cb_tof, GUI_TOOLTIPS['subtract_tof'])
        self.stage5_tof_distance_var = tk.StringVar(value="")
        ttk.Label(
            phase_frame,
            textvariable=self.stage5_tof_distance_var,
            anchor=tk.W,
            font=("Arial", 8),
        ).pack(side=tk.LEFT, padx=(10, 6))

        # --- Evaluation Mode ---
        eval_frame = ttk.LabelFrame(main_container, text="Evaluation Mode", padding="10")
        eval_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.stage5_dist_frame = ttk.Frame(eval_frame, padding=(0, 0, 0, 5))
        self.stage5_dist_frame.pack(side=tk.TOP, fill=tk.X)
        self.stage5_vars['dist_mic'] = self._add_form_entry(self.stage5_dist_frame, "Distance (m):", "1.0", GUI_TOOLTIPS.get('dist_mic'))

        # --- CTA-2034 Mode ---
        self.stage5_cta_frame = ttk.Frame(eval_frame, padding=(0, 5, 0, 5))
        self.stage5_cta_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Label(self.stage5_cta_frame, text="CTA-2034:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.stage5_vars['cta_mode'] = self._add_checkbutton(self.stage5_cta_frame, "Generate CTA-2034 graphs", False, GUI_TOOLTIPS.get('cta_mode'))

        # --- Arc Sweep Mode ---
        self.stage5_arc_frame = ttk.Frame(eval_frame, padding=(0, 5, 0, 5))
        self.stage5_arc_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Label(self.stage5_arc_frame, text="Measurement Arc Sweep:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.stage5_vars['range_deg'] = self._add_form_entry(self.stage5_arc_frame, "Range (+/- deg):", "90", GUI_TOOLTIPS.get('range_deg'))
        self.stage5_vars['increment_deg'] = self._add_form_entry(self.stage5_arc_frame, "Increment (deg):", "10", GUI_TOOLTIPS.get('increment_deg'))
        self.stage5_vars['direction'] = self._add_combobox(self.stage5_arc_frame, "Sweep Direction:", ["horizontal", "vertical", "hor_vert"], "horizontal", GUI_TOOLTIPS.get('direction'))

        # --- Manual List Mode ---
        self.stage5_manual_frame = ttk.Frame(eval_frame, padding=(0, 5, 0, 5))
        self.stage5_manual_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Label(self.stage5_manual_frame, text="Manual Coordinate List:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        manual_inner_frame = ttk.Frame(self.stage5_manual_frame)
        manual_inner_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.stage5_vars['manual_list_mode'] = tk.BooleanVar(value=False)
        cb_manual = ttk.Checkbutton(manual_inner_frame, text="Use Manual Coordinate List", variable=self.stage5_vars['manual_list_mode'])
        cb_manual.pack(side=tk.LEFT, padx=(0,5))
        if GUI_TOOLTIPS.get('manual_list_mode'): ToolTip(cb_manual, GUI_TOOLTIPS['manual_list_mode'])
        
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
            self._set_widget_state(reference_geometry_frame, tk.DISABLED if is_manual else tk.NORMAL)

        # Add traces to all relevant vars to trigger live preview update
        for key in ['cta_mode', 'manual_list_mode', 'subtract_tof', 'generate_ir_files', 'direction']:
            self.stage5_vars[key].trace_add("write", self._schedule_update_stage5_preview)
        
        # For entry boxes, the update is handled by FocusOut/Return bindings in _add_form_entry/_add_labeled_entry
        # But we still need to trace the mode changes here
        self.stage5_vars['cta_mode'].trace_add("write", lambda *args: [ _on_cta_changed(), self._schedule_update_stage5_preview()])
        self.stage5_vars['manual_list_mode'].trace_add("write", lambda *args: [ _on_manual_changed(), self._schedule_update_stage5_preview()])

        # Baffle width/height/position come from the project baffle waypoints.
        default_dut_depth = self._get_project_baffle_width_m()
        self.stage5_vars['dut_depth_x'] = tk.StringVar(value=f"{default_dut_depth * 1000.0:.3f}" if default_dut_depth else "0.0")
        self.stage5_vars['show_stage2_origin'] = tk.BooleanVar(value=False)
        self.stage5_vars['stage2_origin_frequency_hz'] = tk.StringVar(value="1000")

        # Bind updates for entries that don't use the trace system
        for var_key in ['dist_mic', 'range_deg', 'increment_deg', 'offset_mic_x', 'offset_mic_y', 'offset_mic_z', 'zero_theta_deg', 'zero_phi_deg', 'dut_depth_x']:
            # This is a bit of a hack to get the widget from the var, assuming it was the last one created
            self.stage5_vars[var_key].trace_add('write', self._schedule_update_stage5_preview)

        _sync_eval_ui()

        # --- Live Preview ---
        preview_frame = ttk.LabelFrame(main_container, text="Preview", padding="10")
        preview_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        preview_controls = ttk.Frame(preview_frame)
        preview_controls.pack(side=tk.TOP, fill=tk.X)
        self.btn_stage5_prev = ttk.Button(
            preview_controls, text="Previous", command=lambda: self._step_stage5_live_point(-1), state=tk.DISABLED
        )
        self.btn_stage5_prev.pack(side=tk.LEFT)
        self.btn_stage5_live = ttk.Button(preview_controls, text="Live Preview", command=self._open_stage5_live_preview)
        self.btn_stage5_live.pack(side=tk.LEFT, expand=True, padx=10)
        self.btn_stage5_next = ttk.Button(
            preview_controls, text="Next", command=lambda: self._step_stage5_live_point(1), state=tk.DISABLED
        )
        self.btn_stage5_next.pack(side=tk.RIGHT)
        self.stage5_live_point_var = tk.StringVar(value="Open Live Preview to inspect an observation point.")
        ttk.Label(preview_frame, textvariable=self.stage5_live_point_var, anchor=tk.CENTER).pack(
            side=tk.TOP, fill=tk.X, pady=(7, 0)
        )

        # --- Advanced Settings ---
        self.btn_stage5_advanced = ttk.Button(main_container, text="Show Advanced Settings", command=self._toggle_stage5_advanced)
        self.btn_stage5_advanced.pack(side=tk.TOP, pady=10)

        self.stage5_adv_frame = ttk.LabelFrame(main_container, text="Advanced Settings", padding="10")
        
        self.stage5_vars['obs_mode'] = self._add_combobox(self.stage5_adv_frame, "Observation Mode:", ["Internal", "External", "Full"], "Internal", GUI_TOOLTIPS.get('obs_mode'))
        self.stage5_vars['mic_cal_fade_octaves'] = self._add_form_entry(self.stage5_adv_frame, "Mic Cal Fade Octaves:", "1.0", GUI_TOOLTIPS.get('mic_cal_fade_octaves'))
        origins_frame = ttk.Frame(self.stage5_adv_frame)
        origins_frame.pack(anchor=tk.W, fill=tk.X, pady=(5, 0))
        self.stage5_vars['use_optimized_origins'] = self._add_checkbutton(
            origins_frame,
            "Use Optimized Origins",
            True,
            GUI_TOOLTIPS.get('use_optimized_origins_stage5')
        )

        try:
            from extract_pressures_core import IR_CAPTURE_PADDING_SAMPLES
            default_ir_padding = str(IR_CAPTURE_PADDING_SAMPLES)
        except Exception:
            default_ir_padding = "50"
        ir_padding_frame = ttk.Frame(self.stage5_adv_frame)
        ir_padding_frame.pack(anchor=tk.W, fill=tk.X, pady=(5, 0))
        self.stage5_vars['manual_ir_capture_padding'] = self._add_checkbutton(
            ir_padding_frame,
            "Edit IR Capture Padding",
            False,
            GUI_TOOLTIPS.get('manual_ir_capture_padding')
        )
        self.stage5_vars['ir_capture_padding_samples'] = self._add_form_entry(
            self.stage5_adv_frame,
            "IR Capture Padding Samples:",
            default_ir_padding,
            GUI_TOOLTIPS.get('ir_capture_padding_samples'),
            state_var=self.stage5_vars['manual_ir_capture_padding']
        )

        # Any setting that changes preview pressure should refresh an open window.
        for var_key in [
            'frd_db_offset', 'apply_mic_cal', 'mic_cal_file', 'mic_cal_mode',
            'obs_mode', 'mic_cal_fade_octaves', 'use_optimized_origins',
            'manual_ir_capture_padding', 'ir_capture_padding_samples',
        ]:
            self.stage5_vars[var_key].trace_add('write', self._schedule_stage5_live_preview)

        self._invalidate_stage5_tof_distance()
        self.btn_stage5_run = ttk.Button(main_container, text="Run Stage 5", command=self._action_run_stage5)
        self.btn_stage5_run.pack(side=tk.TOP, pady=(10, 20))
        
    def _toggle_stage5_advanced(self):
        if self.stage5_adv_frame.winfo_ismapped():
            self.stage5_adv_frame.pack_forget()
            self.btn_stage5_advanced.config(text="Show Advanced Settings")
            self.stage5_scroll_canvas.yview_moveto(0)
        else:
            self.stage5_adv_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=self.btn_stage5_run)
            self.btn_stage5_advanced.config(text="Hide Advanced Settings")

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
            self._invalidate_stage5_tof_distance()
            self._schedule_update_stage5_preview()
            
        ttk.Button(frame, text="Close", command=close_and_update).pack(side=tk.BOTTOM, pady=5)

    def _schedule_update_stage5_preview(self, *args):
        if args and hasattr(self, 'stage5_tof_distance_var'):
            self._invalidate_stage5_tof_distance()
        if self.stage5_update_job:
            self.after_cancel(self.stage5_update_job)
        self.stage5_update_job = self.after(100, self._update_stage5_preview)
        self._schedule_stage5_live_preview()

    def _invalidate_stage5_tof_distance(self):
        mode = self.stage5_vars.get('subtract_tof')
        mode = mode.get() if mode is not None else "Off"
        if mode == "Ref Origin":
            try:
                geometry = self._stage5_evaluation_geometry()
                reference_index = geometry['reference_index']
                if self.stage5_vars['manual_list_mode'].get():
                    distance = geometry['reference_distance']
                else:
                    import numpy as np

                    reference_mic = geometry['final_xyz'][reference_index]
                    reference_origin = np.asarray(geometry['offset_xyz'], dtype=float)
                    distance = float(np.linalg.norm(reference_mic - reference_origin))
                self.stage5_tof_distance_var.set(f"TOF distance: {distance:.4f} m")
            except (KeyError, TypeError, ValueError):
                self.stage5_tof_distance_var.set("TOF distance: unavailable")
        elif mode in ("Min Phase Ref", "IR Peak"):
            self.stage5_tof_distance_var.set("TOF distance: open Live Preview to calculate")
        else:
            self.stage5_tof_distance_var.set("")

    def _jog_stage5_offset(self, var_key, direction):
        try:
            step = float(self.stage5_vars['offset_step_mm'].get())
            current = float(self.stage5_vars[var_key].get())
            updated = current + float(direction) * step
            self.stage5_vars[var_key].set(f"{updated:.3f}")
            self._schedule_update_stage5_preview()
        except (KeyError, TypeError, ValueError):
            self.bell()
        return "break"

    def _stage5_evaluation_geometry(self):
        import math
        import numpy as np

        offset_xyz = np.asarray(self._stage5_offset_m(), dtype=float)
        zero_theta = float(self.stage5_vars['zero_theta_deg'].get())
        zero_phi = float(self.stage5_vars['zero_phi_deg'].get())
        distance = float(self.stage5_vars['dist_mic'].get())
        manual_mode = self.stage5_vars['manual_list_mode'].get()
        base_spherical = []
        base_xyz = []

        if manual_mode:
            for row in getattr(self, 'stage5_manual_coords', []):
                theta, phi = float(row[0]), float(row[1])
                radius = float(row[2]) if len(row) > 2 else distance
                th_rad, ph_rad = math.radians(theta), math.radians(phi)
                xyz = np.array([
                    radius * math.sin(th_rad) * math.cos(ph_rad),
                    radius * math.sin(th_rad) * math.sin(ph_rad),
                    radius * math.cos(th_rad),
                ])
                base_spherical.append((theta, phi, radius))
                base_xyz.append(xyz)
        else:
            th_rad = math.radians(zero_theta)
            ph_rad = math.radians(zero_phi)
            forward = np.array([math.sin(th_rad) * math.cos(ph_rad), math.sin(th_rad) * math.sin(ph_rad), math.cos(th_rad)])
            right = np.array([-math.sin(ph_rad), math.cos(ph_rad), 0.0])
            up = np.cross(forward, right)
            rotation = np.array([forward, right, up]).T

            candidate_xyz = []
            if self.stage5_vars['cta_mode'].get():
                for angle in range(0, 360, 10):
                    rad = math.radians(angle)
                    candidate_xyz.append(rotation @ np.array([distance * math.cos(rad), distance * math.sin(rad), 0.0]))
                for angle in range(0, 360, 10):
                    rad = math.radians(angle)
                    candidate_xyz.append(rotation @ np.array([distance * math.cos(rad), 0.0, distance * math.sin(rad)]))
            else:
                sweep_range = int(self.stage5_vars['range_deg'].get())
                increment = int(self.stage5_vars['increment_deg'].get())
                if increment <= 0:
                    raise ValueError("Stage 5 increment must be greater than zero.")
                direction = self.stage5_vars['direction'].get().lower()
                # Number/navigate the preview clockwise from on-axis. The bulk
                # exporter can retain its centre-out calculation order without
                # making Previous/Next alternate between the two sides.
                from stage5_pressure_utils import preview_sweep_sequence
                for arc, angle in preview_sweep_sequence(sweep_range, increment, direction):
                    rad = math.radians(angle)
                    if arc == "horizontal":
                        candidate_xyz.append(rotation @ np.array([distance * math.cos(rad), distance * math.sin(rad), 0.0]))
                    else:
                        candidate_xyz.append(rotation @ np.array([distance * math.cos(rad), 0.0, distance * math.sin(rad)]))

            # The two arcs share both their front and rear crossings. Compare
            # Cartesian positions because spherical +180/-180 representations
            # can describe the same rear point with different angle pairs.
            from stage5_pressure_utils import unique_cartesian_points
            for xyz in unique_cartesian_points(candidate_xyz):
                radius = float(np.linalg.norm(xyz))
                theta = math.degrees(math.acos(np.clip(xyz[2] / max(radius, 1e-12), -1.0, 1.0)))
                phi = math.degrees(math.atan2(xyz[1], xyz[0]))
                rounded_theta_deg = round(theta, 2)
                rounded_phi_deg = round(phi, 2)
                base_spherical.append((rounded_theta_deg, rounded_phi_deg, radius))
                rounded_theta = math.radians(rounded_theta_deg)
                rounded_phi = math.radians(rounded_phi_deg)
                base_xyz.append(np.array([
                    radius * math.sin(rounded_theta) * math.cos(rounded_phi),
                    radius * math.sin(rounded_theta) * math.sin(rounded_phi),
                    radius * math.cos(rounded_theta),
                ]))

        if not base_xyz:
            raise ValueError("No Stage 5 observation points are configured.")

        base_xyz = np.asarray(base_xyz, dtype=float)
        final_xyz = base_xyz.copy() if manual_mode else base_xyz + offset_xyz
        final_spherical = []
        for x, y, z in final_xyz:
            radius = math.sqrt(x * x + y * y + z * z)
            theta = math.degrees(math.acos(np.clip(z / max(radius, 1e-12), -1.0, 1.0)))
            phi = math.degrees(math.atan2(y, x))
            final_spherical.append((theta, phi, radius))

        if manual_mode:
            reference_index = int(np.argmin([point[2] for point in base_spherical]))
        else:
            target = base_xyz[0] * 0.0
            target[:] = distance * np.array([
                math.sin(math.radians(zero_theta)) * math.cos(math.radians(zero_phi)),
                math.sin(math.radians(zero_theta)) * math.sin(math.radians(zero_phi)),
                math.cos(math.radians(zero_theta)),
            ])
            reference_index = int(np.argmin(np.linalg.norm(base_xyz - target, axis=1)))

        return {
            'base_spherical': base_spherical,
            'final_spherical': final_spherical,
            'final_xyz': final_xyz,
            'reference_index': reference_index,
            'reference_distance': float(np.min([point[2] for point in base_spherical])),
            'offset_xyz': tuple(offset_xyz),
            'zero_theta': zero_theta,
            'zero_phi': zero_phi,
        }

    def _update_stage5_preview(self):
        if self.stage5_viewer is None:
            return
        try:
            box_dims, box_center, box_vertices = self._get_project_baffle_box_m()
            metadata_mode = self.main_notebook.index(self.main_notebook.select()) == 0
            if metadata_mode:
                # At project-open time this is a geometry/metadata preview only.
                # Stage 5 evaluation points and its reference axis are not relevant yet.
                mic_coords = None
                active_index = None
                ref_origin = (0.0, 0.0, 0.0)
                zero_theta = 90.0
                zero_phi = 0.0
            else:
                geometry = self._stage5_evaluation_geometry()
                mic_coords = geometry['final_xyz']
                active_index = self.stage5_live_point_index if self.stage5_live_window is not None else None
                ref_origin = geometry['offset_xyz']
                zero_theta = geometry['zero_theta']
                zero_phi = geometry['zero_phi']
            named_points = self._get_project_named_points_m()
            stage2_origin = self._get_stage2_acoustic_origin_point()
            if stage2_origin is not None:
                named_points.append(stage2_origin)

            self.stage5_viewer.update_view(
                box_dims=box_dims, 
                mic_coords_xyz=mic_coords,
                active_mic_index=active_index,
                ref_origin=ref_origin,
                zero_theta_deg=zero_theta,
                zero_phi_deg=zero_phi,
                show_reference_axis=not metadata_mode,
                named_points_xyz=named_points,
                z_center=self._get_project_z_center_m(),
                box_center=box_center,
                box_vertices=box_vertices
            )
            
        except ValueError:
            pass # Silence float conversion errors from empty fields
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not generate preview:\n{str(e)}")

    def _open_stage5_live_preview(self):
        if self.stage5_live_window is not None:
            try:
                self.stage5_live_window.deiconify()
                self.stage5_live_window.lift()
                self.stage5_live_window.focus_force()
                return
            except tk.TclError:
                self.stage5_live_window = None

        try:
            initial_request = self._collect_stage5_live_request()
            self._ensure_stage5_live_evaluator(initial_request['she_input'])
        except Exception as exc:
            messagebox.showerror("Live Preview", f"Could not open Live Preview:\n{exc}")
            return

        window = tk.Toplevel(self)
        window.title("Stage 5 Live Preview")
        window.geometry("900x800")
        window.minsize(650, 600)
        window.transient(self)
        window.protocol("WM_DELETE_WINDOW", self._close_stage5_live_preview)
        self.stage5_live_window = window
        self.btn_stage5_prev.config(state=tk.NORMAL)
        self.btn_stage5_next.config(state=tk.NORMAL)

        preview_toolbar = ttk.Frame(window, padding=(6, 4))
        preview_toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(preview_toolbar, text="FR/phase smoothing:").pack(side=tk.LEFT)
        self.stage5_live_smoothing_var = tk.StringVar(value="Off")
        smoothing_combo = ttk.Combobox(
            preview_toolbar,
            textvariable=self.stage5_live_smoothing_var,
            values=("Off", "1/3", "1/6", "1/12", "1/24", "1/48"),
            state="readonly",
            width=7,
        )
        smoothing_combo.pack(side=tk.LEFT, padx=(5, 0))
        smoothing_combo.bind("<<ComboboxSelected>>", self._redraw_stage5_live_cached)
        ttk.Label(
            preview_toolbar,
            text="Left drag: zoom   Right drag: pan   Double-click: reset",
            font=("Arial", 8, "italic"),
        ).pack(side=tk.RIGHT)

        self.stage5_live_figure, (
            self.stage5_live_mag_ax,
            self.stage5_live_phase_ax,
            self.stage5_live_ir_ax,
        ) = plt.subplots(3, 1, figsize=(9, 7), gridspec_kw={'height_ratios': (2.0, 1.5, 1.5)})
        self.stage5_live_figure.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.08, hspace=0.28)
        plot_host = ttk.Frame(window)
        plot_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.stage5_live_canvas = FigureCanvasTkAgg(self.stage5_live_figure, master=plot_host)
        self.stage5_live_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._connect_stage5_live_plot_interactions()
        self.stage5_live_status_var = tk.StringVar(value="Preparing preview…")
        ttk.Label(window, textvariable=self.stage5_live_status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2)).pack(side=tk.BOTTOM, fill=tk.X)

        self.stage5_live_point_index = 0
        self.stage5_live_last_result = None
        self.stage5_live_plot_limits = {"magnitude": None, "phase": None, "ir": None}
        self.stage5_live_plot_drag = None
        self._update_stage5_preview()
        self._schedule_stage5_live_preview(immediate=True)
        self.after(50, self._poll_stage5_live_results)

    def _ensure_stage5_live_evaluator(self, coefficient_path):
        resolved_path = os.path.abspath(coefficient_path)
        if self.stage5_live_evaluator is not None and self.stage5_live_evaluator_path == resolved_path:
            return self.stage5_live_evaluator

        old_evaluator = self.stage5_live_evaluator
        self.stage5_live_evaluator = None
        self.stage5_live_evaluator_path = None
        self.stage5_live_last_result = None
        self.stage5_live_plot_limits = {}
        self.stage5_live_plot_drag = None
        if old_evaluator is not None:
            old_evaluator.close()

        from extract_pressures_core import PressureEvaluationSession

        evaluator = PressureEvaluationSession(resolved_path, use_process_pool=True)
        self.stage5_live_evaluator = evaluator
        self.stage5_live_evaluator_path = resolved_path
        return evaluator

    def _raise_stage5_live_with_main(self, event=None):
        """Keep the owned preview with the application when Windows activates it."""
        if event is not None and event.widget is not self:
            return
        window = self.stage5_live_window
        if window is None:
            return
        try:
            if window.state() != "withdrawn":
                self.after_idle(window.lift)
        except tk.TclError:
            self.stage5_live_window = None

    def _close_stage5_live_preview(self, wait_for_pool=False):
        self.stage5_live_generation += 1
        if self.stage5_live_update_job is not None:
            try:
                self.after_cancel(self.stage5_live_update_job)
            except Exception:
                pass
            self.stage5_live_update_job = None
        window = self.stage5_live_window
        self.stage5_live_window = None
        evaluator = self.stage5_live_evaluator
        self.stage5_live_evaluator = None
        self.stage5_live_evaluator_path = None
        self.stage5_live_last_result = None
        self.stage5_live_plot_limits = {}
        self.stage5_live_plot_drag = None
        if evaluator is not None:
            if wait_for_pool:
                evaluator.close()
            else:
                threading.Thread(
                    target=evaluator.close,
                    daemon=True,
                    name="stage5-preview-pool-close",
                ).start()
        if hasattr(self, 'btn_stage5_prev'):
            self.btn_stage5_prev.config(state=tk.DISABLED)
            self.btn_stage5_next.config(state=tk.DISABLED)
        if hasattr(self, 'stage5_live_point_var'):
            self.stage5_live_point_var.set("Open Live Preview to inspect an observation point.")
        if hasattr(self, 'stage5_live_figure'):
            plt.close(self.stage5_live_figure)
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass
        self._schedule_update_stage5_preview()

    def _redraw_stage5_live_cached(self, _event=None):
        if self.stage5_live_window is not None and self.stage5_live_last_result is not None:
            self._draw_stage5_live_result(self.stage5_live_last_result)

    def _stage5_live_plot_axis(self, plot_name):
        return {
            "magnitude": self.stage5_live_mag_ax,
            "phase": self.stage5_live_phase_ax,
            "ir": self.stage5_live_ir_ax,
        }[plot_name]

    def _stage5_live_plot_name(self, axis):
        for plot_name in ("magnitude", "phase", "ir"):
            if axis is self._stage5_live_plot_axis(plot_name):
                return plot_name
        return None

    def _connect_stage5_live_plot_interactions(self):
        canvas = self.stage5_live_canvas
        canvas.mpl_connect("button_press_event", self._on_stage5_live_plot_press)
        canvas.mpl_connect("motion_notify_event", self._on_stage5_live_plot_motion)
        canvas.mpl_connect("button_release_event", self._on_stage5_live_plot_release)

    def _set_stage5_live_plot_limits(self, plot_name, x_limits, y_limits):
        """Apply limits, linking the magnitude/phase frequency axes."""
        axis = self._stage5_live_plot_axis(plot_name)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        self.stage5_live_plot_limits[plot_name] = (x_limits, y_limits)

        if plot_name in ("magnitude", "phase"):
            linked_name = "phase" if plot_name == "magnitude" else "magnitude"
            linked_axis = self._stage5_live_plot_axis(linked_name)
            linked_axis.set_xlim(*x_limits)
            linked_limits = self.stage5_live_plot_limits.get(linked_name)
            linked_y_limits = linked_limits[1] if linked_limits is not None else linked_axis.get_ylim()
            self.stage5_live_plot_limits[linked_name] = (x_limits, linked_y_limits)

    def _on_stage5_live_plot_press(self, event):
        plot_name = self._stage5_live_plot_name(event.inaxes)
        if plot_name is None or event.xdata is None or event.ydata is None:
            return

        if event.dblclick:
            self.stage5_live_plot_drag = None
            self._reset_stage5_live_plot_zoom(plot_name)
            return

        if event.button == 1:
            from matplotlib.patches import Rectangle

            rectangle = Rectangle(
                (event.xdata, event.ydata), 0.0, 0.0,
                facecolor="#1f77b4", edgecolor="#1f77b4",
                linewidth=0.8, alpha=0.18,
            )
            event.inaxes.add_patch(rectangle)
            self.stage5_live_plot_drag = {
                "mode": "zoom",
                "plot_name": plot_name,
                "axis": event.inaxes,
                "start": (event.xdata, event.ydata),
                "start_pixel": (event.x, event.y),
                "rectangle": rectangle,
            }
            self.stage5_live_canvas.draw_idle()
        elif event.button == 3:
            self.stage5_live_plot_drag = {
                "mode": "pan",
                "plot_name": plot_name,
                "axis": event.inaxes,
                "start": (event.xdata, event.ydata),
                "start_pixel": (event.x, event.y),
                "xlim": event.inaxes.get_xlim(),
                "ylim": event.inaxes.get_ylim(),
            }

    def _on_stage5_live_plot_motion(self, event):
        drag = self.stage5_live_plot_drag
        if drag is None or event.x is None or event.y is None:
            return

        axis = drag["axis"]
        start_x, start_y = drag["start"]
        if drag["mode"] == "zoom":
            # Continue a selection outside the axes in the conventional way:
            # pin the moving corner to the nearest visible plot boundary.
            clamped_x = min(max(event.x, axis.bbox.x0), axis.bbox.x1)
            clamped_y = min(max(event.y, axis.bbox.y0), axis.bbox.y1)
            current_x, current_y = axis.transData.inverted().transform((clamped_x, clamped_y))
            rectangle = drag["rectangle"]
            rectangle.set_x(min(start_x, current_x))
            rectangle.set_y(min(start_y, current_y))
            rectangle.set_width(abs(current_x - start_x))
            rectangle.set_height(abs(current_y - start_y))
        else:
            delta_x_pixels = event.x - drag["start_pixel"][0]
            delta_y_pixels = event.y - drag["start_pixel"][1]
            if axis.get_xscale() == "log":
                import math

                log_limits = tuple(math.log(value) for value in drag["xlim"])
                shift = delta_x_pixels / axis.bbox.width * (log_limits[1] - log_limits[0])
                x_limits = tuple(math.exp(value - shift) for value in log_limits)
            else:
                shift = delta_x_pixels / axis.bbox.width * (drag["xlim"][1] - drag["xlim"][0])
                x_limits = tuple(value - shift for value in drag["xlim"])
            y_shift = delta_y_pixels / axis.bbox.height * (drag["ylim"][1] - drag["ylim"][0])
            y_limits = tuple(value - y_shift for value in drag["ylim"])
            self._set_stage5_live_plot_limits(drag["plot_name"], x_limits, y_limits)
        self.stage5_live_canvas.draw_idle()

    def _on_stage5_live_plot_release(self, event):
        drag = self.stage5_live_plot_drag
        self.stage5_live_plot_drag = None
        if drag is None:
            return

        axis = drag["axis"]
        if drag["mode"] == "zoom":
            rectangle = drag["rectangle"]
            try:
                rectangle.remove()
            except ValueError:
                pass
            if event.x is not None and event.y is not None:
                clamped_x = min(max(event.x, axis.bbox.x0), axis.bbox.x1)
                clamped_y = min(max(event.y, axis.bbox.y0), axis.bbox.y1)
                current_x, current_y = axis.transData.inverted().transform((clamped_x, clamped_y))
                pixel_distance = (
                    abs(clamped_x - drag["start_pixel"][0])
                    + abs(clamped_y - drag["start_pixel"][1])
                )
                start_x, start_y = drag["start"]
                x_limits = tuple(sorted((start_x, current_x)))
                y_limits = tuple(sorted((start_y, current_y)))
                if pixel_distance >= 6 and x_limits[0] != x_limits[1] and y_limits[0] != y_limits[1]:
                    if axis.get_xscale() != "log" or x_limits[0] > 0.0:
                        self._set_stage5_live_plot_limits(drag["plot_name"], x_limits, y_limits)
        else:
            self._set_stage5_live_plot_limits(
                drag["plot_name"], axis.get_xlim(), axis.get_ylim()
            )
        self.stage5_live_canvas.draw_idle()

    def _reset_stage5_live_plot_zoom(self, plot_name):
        if self.stage5_live_window is None:
            return
        linked_name = None
        linked_y_limits = None
        if plot_name in ("magnitude", "phase"):
            linked_name = "phase" if plot_name == "magnitude" else "magnitude"
            linked_y_limits = self._stage5_live_plot_axis(linked_name).get_ylim()
            self.stage5_live_plot_limits[linked_name] = None
        self.stage5_live_plot_limits[plot_name] = None
        self._redraw_stage5_live_cached()
        if linked_name is not None:
            axis = self._stage5_live_plot_axis(plot_name)
            linked_axis = self._stage5_live_plot_axis(linked_name)
            shared_x_limits = axis.get_xlim()
            linked_axis.set_xlim(*shared_x_limits)
            linked_axis.set_ylim(*linked_y_limits)
            self.stage5_live_plot_limits[plot_name] = (shared_x_limits, axis.get_ylim())
            self.stage5_live_plot_limits[linked_name] = (shared_x_limits, linked_y_limits)
            self.stage5_live_canvas.draw_idle()

    def _step_stage5_live_point(self, delta):
        try:
            count = len(self._stage5_evaluation_geometry()['final_spherical'])
        except Exception:
            self.bell()
            return
        self.stage5_live_point_index = (self.stage5_live_point_index + int(delta)) % count
        self._update_stage5_preview()
        self._schedule_stage5_live_preview(immediate=True)

    def _schedule_stage5_live_preview(self, *trace_args, immediate=False):
        # StringVar/BooleanVar traces supply (variable_name, index, operation).
        # Accept and ignore those values while retaining the keyword used by
        # direct callers that need an immediate refresh.
        if trace_args and hasattr(self, 'stage5_tof_distance_var'):
            self._invalidate_stage5_tof_distance()
        if self.stage5_live_window is None:
            return
        self.stage5_live_generation += 1
        if self.stage5_live_update_job is not None:
            try:
                self.after_cancel(self.stage5_live_update_job)
            except Exception:
                pass
        delay = 0 if immediate else 100
        self.stage5_live_update_job = self.after(delay, self._start_stage5_live_calculation)

    def _resolve_stage5_preview_cal_file(self):
        if not self.stage5_vars['apply_mic_cal'].get():
            return None
        configured = self.stage5_vars['mic_cal_file'].get().strip()
        if not configured:
            raise ValueError("Microphone calibration is enabled but no calibration file is selected.")
        resolved = configured if os.path.isabs(configured) else os.path.join(self.project_dir.get(), configured)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Microphone calibration file not found: {resolved}")
        return resolved

    def _collect_stage5_live_request(self):
        geometry = self._stage5_evaluation_geometry()
        count = len(geometry['final_spherical'])
        self.stage5_live_point_index %= count
        project_dir = self.project_dir.get()
        project_name = self.project_name.get()
        coeff_path = os.path.join(project_dir, "outputs", "coefficients", f"{project_name}_coefficients.h5")
        if not os.path.exists(coeff_path):
            raise FileNotFoundError(f"Coefficient file not found: {coeff_path}")
        manual_padding = self.stage5_vars['manual_ir_capture_padding'].get()
        padding = int(self.stage5_vars['ir_capture_padding_samples'].get()) if manual_padding else None
        if padding is not None and padding < 0:
            raise ValueError("IR Capture Padding Samples must be zero or greater.")
        index = self.stage5_live_point_index
        xyz = geometry['final_xyz'][index]
        reference_index = geometry['reference_index']
        return {
            'coord_sph': geometry['final_spherical'][index],
            'reference_coord_sph': geometry['final_spherical'][reference_index],
            'reference_distance': geometry['reference_distance'],
            'cartesian': tuple(float(value) for value in xyz),
            'point_index': index,
            'point_count': count,
            'she_input': coeff_path,
            'obs_mode': self.stage5_vars['obs_mode'].get(),
            'use_optimized_origins': self.stage5_vars['use_optimized_origins'].get(),
            'ir_capture_padding_samples': padding,
            'subtract_tof': self.stage5_vars['subtract_tof'].get(),
            'apply_mic_cal': self.stage5_vars['apply_mic_cal'].get(),
            'mic_cal_file': self._resolve_stage5_preview_cal_file(),
            'mic_cal_mode': self.stage5_vars['mic_cal_mode'].get(),
            'mic_cal_fade_octaves': float(self.stage5_vars['mic_cal_fade_octaves'].get()),
            'frd_db_offset': float(self.stage5_vars['frd_db_offset'].get()),
        }

    def _start_stage5_live_calculation(self):
        self.stage5_live_update_job = None
        if self.stage5_live_window is None:
            return
        if self.stage5_live_running:
            return
        generation = self.stage5_live_generation
        try:
            request = self._collect_stage5_live_request()
            evaluator = self._ensure_stage5_live_evaluator(request['she_input'])
        except Exception as exc:
            self.stage5_live_status_var.set(f"Preview unavailable: {exc}")
            return
        x, y, z = request['cartesian']
        self.stage5_live_point_var.set(
            f"Point {request['point_index'] + 1} of {request['point_count']}   "
            f"X {x:.4f} m   Y {y:.4f} m   Z {z:.4f} m"
        )
        self.stage5_live_status_var.set("Calculating full-resolution preview…")
        if request['subtract_tof'] in ("Min Phase Ref", "IR Peak"):
            self.stage5_tof_distance_var.set("TOF distance: calculating…")
        self.stage5_live_running = True

        def worker():
            try:
                if evaluator is None:
                    raise RuntimeError("The live preview evaluation pool is not available.")
                result = evaluator.evaluate_preview_response(**{
                    key: value for key, value in request.items()
                    if key not in ('cartesian', 'point_index', 'point_count', 'she_input')
                })
                self.stage5_live_queue.put((generation, result, None))
            except Exception as exc:
                self.stage5_live_queue.put((generation, None, exc))

        threading.Thread(target=worker, daemon=True, name="stage5-live-preview").start()

    def _poll_stage5_live_results(self):
        if self.stage5_live_window is None:
            return
        newest = None
        try:
            while True:
                newest = self.stage5_live_queue.get_nowait()
        except queue.Empty:
            pass
        if newest is not None:
            generation, result, error = newest
            self.stage5_live_running = False
            if generation == self.stage5_live_generation:
                if error is not None:
                    self.stage5_live_status_var.set(f"Preview failed: {error}")
                    mode = self.stage5_vars['subtract_tof'].get()
                    if mode in ("Min Phase Ref", "IR Peak"):
                        self.stage5_tof_distance_var.set("TOF distance: unavailable")
                else:
                    self._draw_stage5_live_result(result)
            else:
                self._schedule_stage5_live_preview(immediate=True)
        self.after(50, self._poll_stage5_live_results)

    def _draw_stage5_live_result(self, result):
        import matplotlib.ticker as ticker
        import numpy as np

        self.stage5_live_last_result = result
        freqs = result['freqs']
        magnitude = np.asarray(result['magnitude'], dtype=float)
        phase = np.asarray(result['phase'], dtype=float)
        smoothing_label = self.stage5_live_smoothing_var.get()
        if smoothing_label != "Off":
            from stage5_pressure_utils import smooth_fractional_octave_response

            denominator = int(smoothing_label.split("/", 1)[1])
            magnitude, phase = smooth_fractional_octave_response(
                freqs, result['complex'], denominator
            )
            magnitude += float(result.get('frd_db_offset', 0.0))

        mag_ax = self.stage5_live_mag_ax
        phase_ax = self.stage5_live_phase_ax
        ir_ax = self.stage5_live_ir_ax
        # clear() resets limits to (0, 1). Switch each frequency axis back to
        # linear first so Matplotlib does not reject that temporary range.
        mag_ax.set_xscale('linear')
        phase_ax.set_xscale('linear')
        mag_ax.clear()
        phase_ax.clear()
        ir_ax.clear()
        mag_ax.semilogx(freqs, magnitude, color='#1f77b4', linewidth=1.4)
        phase_ax.semilogx(freqs, phase, color='#d62728', linewidth=1.1)
        mag_ax.set_ylabel("Magnitude (dB)")
        phase_ax.set_ylabel("Phase (deg)")
        phase_ax.set_xlabel("Frequency (Hz)")
        phase_ax.set_ylim(-180.0, 180.0)
        phase_ax.set_yticks((-180, -90, 0, 90, 180))
        audio_ticks = [20, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 800,
                       1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000]
        visible_ticks = [tick for tick in audio_ticks if freqs[0] <= tick <= freqs[-1]]
        for axis in (mag_ax, phase_ax):
            axis.set_xlim(freqs[0], freqs[-1])
            axis.set_xscale('log')
            axis.set_xticks(visible_ticks)
            axis.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            axis.grid(True, which='major', alpha=0.35)
            axis.grid(True, which='minor', alpha=0.12)
        phase_ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda value, _pos: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"
        ))
        mag_ax.tick_params(labelbottom=False)

        ir = np.asarray(result['ir'], dtype=float)
        ir_times_ms = np.asarray(result['ir_times_s'], dtype=float) * 1000.0
        ir_peak = float(np.max(np.abs(ir))) if ir.size else 0.0
        ir_display = ir / ir_peak if ir_peak > 0.0 else ir
        ir_ax.plot(ir_times_ms, ir_display, color='#2ca02c', linewidth=1.0)
        ir_ax.axhline(0.0, color='black', linewidth=0.6, alpha=0.4)
        ir_ax.set_ylabel("IR (normalized)")
        ir_ax.set_xlabel("Time (ms)")
        ir_ax.grid(True, alpha=0.25)
        ir_ax.set_title("Selected-point IR (before physical TOF subtraction)", fontsize=9)

        reference_time = result.get('tof_reference_time_s')
        reference_time_ms = None if reference_time is None else float(reference_time) * 1000.0
        if reference_time_ms is not None:
            ir_ax.axvline(
                reference_time_ms,
                color='#d62728',
                linestyle='--',
                linewidth=1.4,
                label=f"FRD t=0 from on-axis {result.get('tof_mode', 'reference')} ({reference_time_ms:.3f} ms)",
            )
            ir_ax.legend(loc='upper right', fontsize='small', frameon=False)

        if ir_times_ms.size:
            selected_peak_ms = float(ir_times_ms[int(np.argmax(np.abs(ir)))]) if ir.size else 0.0
            display_end_ms = max(20.0, selected_peak_ms + 10.0)
            if reference_time_ms is not None:
                display_end_ms = max(display_end_ms, reference_time_ms + 10.0)
            ir_ax.set_xlim(0.0, min(float(ir_times_ms[-1]), display_end_ms))

        for plot_name, axis in (("magnitude", mag_ax), ("phase", phase_ax), ("ir", ir_ax)):
            saved_limits = self.stage5_live_plot_limits.get(plot_name)
            if saved_limits is None:
                self.stage5_live_plot_limits[plot_name] = (axis.get_xlim(), axis.get_ylim())
            else:
                axis.set_xlim(*saved_limits[0])
                axis.set_ylim(*saved_limits[1])

        self.stage5_live_canvas.draw_idle()
        bin_count = len(freqs)
        distance = result.get('tof_reference_distance')
        tof_mode = result.get('tof_mode')
        if tof_mode in ("Ref Origin", "Min Phase Ref", "IR Peak") and distance is not None:
            self.stage5_tof_distance_var.set(f"TOF distance: {distance:.4f} m")
        else:
            self.stage5_tof_distance_var.set("")
        suffix = f"; TOF reference {distance:.4f} m" if distance is not None else ""
        smooth_suffix = "" if smoothing_label == "Off" else f"; {smoothing_label}-oct smoothing"
        self.stage5_live_status_var.set(
            f"Ready — {bin_count} full-resolution frequency bins{smooth_suffix}{suffix}"
        )

    def _action_run_stage5(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Run Stage 5")
        try:
            self._save_project_if_not_exists()
            proj_dir = self.project_dir.get()
            proj_name = self.project_name.get()
            coeff_path = os.path.join(proj_dir, "outputs", "coefficients", f"{proj_name}_coefficients.h5")
            if not os.path.exists(coeff_path):
                raise FileNotFoundError(f"Coefficient file not found: {coeff_path}")

            manual_mode = self.stage5_vars['manual_list_mode'].get()
            manual_coords = list(getattr(self, 'stage5_manual_coords', []))
            if manual_mode and not manual_coords:
                raise ValueError("Manual Coordinate List mode is active, but the list is empty.")

            frd_offset_var = self.stage5_vars.get('frd_db_offset')
            manual_ir_padding = self.stage5_vars['manual_ir_capture_padding'].get()
            ir_padding_samples = None
            if manual_ir_padding:
                ir_padding_samples = int(self.stage5_vars['ir_capture_padding_samples'].get())
                if ir_padding_samples < 0:
                    raise ValueError("IR Capture Padding Samples must be zero or greater.")
            settings = {
                'project_dir': proj_dir,
                'coeff_path': coeff_path,
                'output_dir': os.path.join(proj_dir, self.stage5_vars['output_dir'].get()),
                'offset_xyz': tuple(self._stage5_offset_m()),
                'apply_mic_cal': self.stage5_vars['apply_mic_cal'].get(),
                'mic_cal_file': self.stage5_vars['mic_cal_file'].get(),
                'mic_cal_fallback_content': getattr(self, 'mic_cal_fallback_content', ""),
                'mic_cal_mode': self.stage5_vars['mic_cal_mode'].get(),
                'obs_mode': self.stage5_vars['obs_mode'].get(),
                'mic_cal_fade_octaves': float(self.stage5_vars['mic_cal_fade_octaves'].get()),
                'use_optimized_origins': self.stage5_vars['use_optimized_origins'].get(),
                'frd_db_offset': float(frd_offset_var.get()) if frd_offset_var is not None else 0.0,
                'ir_capture_padding_samples': ir_padding_samples,
                'manual_mode': manual_mode,
                'manual_coords': manual_coords,
                'cta_mode': self.stage5_vars['cta_mode'].get(),
                'zero_theta': float(self.stage5_vars['zero_theta_deg'].get()),
                'zero_phi': float(self.stage5_vars['zero_phi_deg'].get()),
                'dist_mic': float(self.stage5_vars['dist_mic'].get()),
                'subtract_tof': self.stage5_vars['subtract_tof'].get(),
                'frd_prefix': self.stage5_vars['frd_prefix'].get(),
                'generate_ir_files': self.stage5_vars['generate_ir_files'].get(),
                'direction': self.stage5_vars['direction'].get(),
                'range_deg': int(self.stage5_vars['range_deg'].get()),
                'increment_deg': int(self.stage5_vars['increment_deg'].get()),
            }
        except Exception as exc:
            messagebox.showerror("Stage 5 Settings", f"Could not start Stage 5:\n{exc}")
            return

        self._start_stage_job(
            "Stage 5",
            lambda: self._run_stage5_job(settings),
            self._finish_stage5_job
        )

    @staticmethod
    def _run_stage5_job(settings):
        from stage5_extract_pressures import run_cta2034_extraction, run_sweep_extraction

        mic_cal_file = ""
        if settings['apply_mic_cal'] and settings['mic_cal_file']:
            cal_path = settings['mic_cal_file']
            full_path = cal_path if os.path.isabs(cal_path) else os.path.join(settings['project_dir'], cal_path)
            if os.path.exists(full_path):
                mic_cal_file = full_path
            elif settings['mic_cal_fallback_content']:
                fallback_path = os.path.join(settings['project_dir'], "fallback_mic_cal.txt")
                try:
                    with open(fallback_path, "w") as file_obj:
                        file_obj.write(settings['mic_cal_fallback_content'])
                    mic_cal_file = fallback_path
                    print("Original mic cal file not found. Using fallback data from project save.")
                except Exception as exc:
                    print(f"Error writing fallback mic cal file: {exc}")
                    mic_cal_file = full_path
            else:
                print("Warning: Mic cal file not found and no fallback data available.")
                mic_cal_file = full_path

        common = {
            'coeff_path': settings['coeff_path'],
            'output_dir': settings['output_dir'],
            'zero_theta': settings['zero_theta'],
            'zero_phi': settings['zero_phi'],
            'dist_mic': settings['dist_mic'],
            'offset_xyz': settings['offset_xyz'],
            'subtract_tof': settings['subtract_tof'],
            'apply_mic_cal': settings['apply_mic_cal'],
            'mic_cal_file': mic_cal_file,
            'mic_cal_mode': settings['mic_cal_mode'],
            'obs_mode': settings['obs_mode'],
            'mic_cal_fade_octaves': settings['mic_cal_fade_octaves'],
            'use_optimized_origins': settings['use_optimized_origins'],
            'frd_db_offset': settings['frd_db_offset'],
            'ir_capture_padding_samples': settings['ir_capture_padding_samples'],
            'use_process_pool': True,
        }

        if settings['manual_mode']:
            run_sweep_extraction(
                **common,
                use_coord_list=True,
                coord_list=settings['manual_coords'],
                frd_prefix=settings['frd_prefix'],
                generate_ir_files=settings['generate_ir_files']
            )
        elif settings['cta_mode']:
            run_cta2034_extraction(**common)
        else:
            run_sweep_extraction(
                **common,
                use_coord_list=False,
                direction=settings['direction'],
                range_deg=settings['range_deg'],
                increment_deg=settings['increment_deg'],
                frd_prefix=settings['frd_prefix'],
                generate_ir_files=settings['generate_ir_files']
            )

    @staticmethod
    def _finish_stage5_job(_result):
        print("Stage 5 completed successfully.")

    def on_closing(self):
        if DEBUG_MODE:
            print("[DEBUG] Action: Application closing")
        pool = getattr(self, '_stage3_pool', None)
        if pool is not None:
            self._stage3_pool = None
            terminate = getattr(pool, 'terminate_workers', None)
            if terminate is not None:
                terminate()
            else:
                pool.shutdown(wait=True, cancel_futures=True)
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
        
    # Make all LabelFrame titles bold (must be applied after the theme change)
    style.configure('TLabelframe.Label', font=('Arial', 10, 'bold'))

    app.mainloop()


if __name__ == "__main__":
    main()

