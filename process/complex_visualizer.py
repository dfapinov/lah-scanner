import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import scrolledtext, font as tkfont
import threading
import platform
import re
import ctypes

# ---------------------------------------------------------
# 1. Setup Parameters
# ---------------------------------------------------------
init_real = 0.0
init_imag = -3.0
init_freq = 1.0 

t_max = 4 * np.pi
num_points = 600
t_vals = np.linspace(0, t_max, num_points)

# ---------------------------------------------------------
# 2. Setup the MAIN INTERACTIVE FIGURE (Matplotlib)
# ---------------------------------------------------------
fig, (ax_phasor, ax_wave) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Interactive Complex Frequency Visualization", fontsize=16)
plt.subplots_adjust(bottom=0.25, top=0.92, wspace=0.3)

# --- Left Plot: Complex Plane ---
ax_phasor.set_title("1. The Phasor (Rotating Vector)", fontsize=12)
ax_phasor.set_xlabel("Real Axis (Cosine Component)")
ax_phasor.set_ylabel("Imaginary Axis (Sine Component)")
ax_phasor.set_xlim(-6, 6)
ax_phasor.set_ylim(-6, 6)
ax_phasor.grid(True)
ax_phasor.set_aspect('equal')
ax_phasor.axhline(0, color='black', lw=1)
ax_phasor.axvline(0, color='black', lw=1)

# --- Right Plot: Time Domain ---
ax_wave.set_title("2. The Waveform (Projection)", fontsize=12)
ax_wave.set_xlabel("Time (t)")
ax_wave.set_ylabel("Amplitude")
ax_wave.set_xlim(0, t_max)
ax_wave.set_ylim(-6, 6)
ax_wave.grid(True)

# ---------------------------------------------------------
# 3. Graphics Objects
# ---------------------------------------------------------
phasor_arrow, = ax_phasor.plot([], [], color='blue', lw=3, marker='o', label='Rotating Phasor')
static_vector, = ax_phasor.plot([], [], color='red', linestyle='--', alpha=0.5, lw=2, label='Complex Coeff')
orbit_circle, = ax_phasor.plot([], [], color='green', linestyle=':', alpha=0.3)

ref_wave_line, = ax_wave.plot(t_vals, 3.0 * np.sin(init_freq * t_vals), color='lightgrey', 
                              lw=2, linestyle='--', label='Ref (Pure Sine, Mag 3)')

wave_line, = ax_wave.plot([], [], color='blue', lw=2, label='Current Wave')
wave_dot, = ax_wave.plot([], [], color='red', marker='o')

math_text = ax_wave.text(0.05, 0.85, "", transform=ax_wave.transAxes, 
                         bbox=dict(facecolor='white', alpha=0.9), animated=True)

ax_wave.legend(loc='upper right', fontsize='small')

# ---------------------------------------------------------
# 4. Calculation Logic
# ---------------------------------------------------------
def update(frame):
    r = s_real.val
    i = s_imag.val
    f = s_freq.val
    
    coef = r + 1j * i
    
    phasors_array = coef * np.exp(1j * f * t_vals)
    wave_y = np.real(phasors_array)
    ref_y = 3.0 * np.sin(f * t_vals)
    
    idx = frame % num_points
    current_phasor = phasors_array[idx]
    
    phasor_arrow.set_data([0, np.real(current_phasor)], [0, np.imag(current_phasor)])
    static_vector.set_data([0, r], [0, i])
    
    mag = np.abs(coef)
    theta = np.linspace(0, 2*np.pi, 100)
    orbit_circle.set_data(mag*np.cos(theta), mag*np.sin(theta))
    
    wave_line.set_data(t_vals[:idx], wave_y[:idx])
    wave_dot.set_data([t_vals[idx]], [wave_y[idx]])
    ref_wave_line.set_ydata(ref_y)
    
    phase_rad = np.angle(coef)
    math_text.set_text(
        f"Complex: {r:.1f} + {i:.1f}i\n"
        f"Mag: {mag:.2f}\n"
        f"Freq: {f:.1f}\n"
        f"Phase: {np.degrees(phase_rad):.1f}°"
    )
    
    return phasor_arrow, static_vector, orbit_circle, wave_line, wave_dot, ref_wave_line, math_text

# ---------------------------------------------------------
# 5. Sliders
# ---------------------------------------------------------
ax_freq = plt.axes([0.25, 0.15, 0.5, 0.03])
ax_real = plt.axes([0.25, 0.10, 0.5, 0.03])
ax_imag = plt.axes([0.25, 0.05, 0.5, 0.03])

s_freq = Slider(ax_freq, 'Freq (Speed)', 0.1, 5.0, valinit=init_freq)
s_real = Slider(ax_real, 'Real (Cosine)', -5.0, 5.0, valinit=init_real)
s_imag = Slider(ax_imag, 'Imag (Sine)', -5.0, 5.0, valinit=init_imag)

ani = FuncAnimation(fig, update, frames=range(2000), interval=20, blit=True)

# ---------------------------------------------------------
# 6. Separate Rich Text Window (High DPI Aware)
# ---------------------------------------------------------
desc_text_content = """ Lesson:

1. THE COMPLEX COEFFICIENT (LEFT):
The plot on the left is the Complex Plane — a 4-quadrant, 2D plane where the horizontal (x) axis is the +/- valued Real coefficient and the vertical (y) axis is the +/- valued Imaginary coefficient.

The Real and Imaginary sliders define a static point on the complex plane (Red Dashed Line).

Coordinate Systems: Note that the same point on the plane can also be defined as an angle and a radius. This is the same vector described in two different coordinate systems — Cartesian and Polar.

Audio DSP: The two components (Sine and Cosine) are inherently 90 degrees separated in phase. The sine and cosine waves can be combined to form a single 'sine' wave where the magnitude and phase are controlled by the mix of the two ingredients.

Note: The waveform magnitude relates to the radius. The phase relates to the angle.

2. THE PHASOR (BLUE ARROW):
The Blue rotating line is the 'Phasor'. It takes that static point and rotates it continuously by a specified speed (frequency).

Rotation Mechanism:** This is multiplication by i. Any point (complex number) multiplied by i will rotate 90 degrees counter-clockwise. The Phasor's rotation is enabled by applying that multiplication continuously over time.

3. THE WAVEFORM (RIGHT):
The graph on the right tracks the Phasor's 'Real' value (horizontal position) over time, generating the waveform.

The Blue wave shows the current output. Compare it to the Grey Reference (Pure Sine). Notice how the phase and magnitude relate to the mix of sine and cosine components.

----------------------------------------------------------------------
EXAMPLE EXPERIMENTS (Try These):

* Example 1: Pure Sine. Set Imaginary (sine) part = -3. Set Real (cosine) part = 0. The vector is exactly on the **lower** vertical axis (-90 deg). The result is a positive sine wave of magnitude 3 (matching the reference).

* Example 2: Pure Cosine. Set Imaginary (sine) part = 0. Set Real (cosine) part = 3. The vector is exactly on the right horizontal axis (0 deg). The result is a pure cosine wave of magnitude 3, which starts at its peak.

* Example 3: Reverse Sweep. Set Imaginary part to 3. Set the Real part to -5. Continuously increase the Real value while observing the phasor's rotation and the waveform. Notice that the phase rotation is effectively reversed by the increasing Real value.

Note: This constant change in the complex coefficient is the same effect as applying phase rotation in the complex frequency domain and can be used to compensate for a time delay.
"""

def set_dpi_awareness():
    """Forces Windows to render the window at full native resolution (crisp text)."""
    try:
        # Windows 10/11
        ctypes.windll.shcore.SetProcessDpiAwareness(1) 
    except Exception:
        try:
            # Windows 8/Vista
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass # Not on Windows or already handled

def launch_guide_window():
    set_dpi_awareness() # <--- CRITICAL FIX FOR BLURRY FONTS
    
    root = tk.Tk()
    root.title("Guide")
    
    # 1. Select High-Quality System Font
    sys_os = platform.system()
    if sys_os == "Windows":
        base_font = "Segoe UI"
    elif sys_os == "Darwin": 
        base_font = "Helvetica Neue"
    else:
        base_font = "Liberation Sans"
    
    # Set window size (larger for readability)
    root.geometry("640x800+950+50")

    # 2. Setup Font Styles
    font_normal = tkfont.Font(family=base_font, size=11)
    font_bold = tkfont.Font(family=base_font, size=11, weight="bold")
    font_header = tkfont.Font(family=base_font, size=13, weight="bold")
    
    # 3. Create Text Widget
    txt = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=font_normal, 
                                    bd=0, padx=25, pady=25) 
    txt.pack(expand=True, fill='both')

    # 4. Define formatting tags
    txt.tag_config("bold", font=font_bold)
    txt.tag_config("header", font=font_header, spacing3=12, foreground="#222222")
    txt.tag_config("divider", justify='center', foreground="#999999")

    # 5. Insert Text
    txt.insert(tk.INSERT, desc_text_content)
    
    # 6. Apply Markdown Styling using Regex
    content = txt.get("1.0", tk.END)

    # A. Bold (**text**)
    # Use regex to find all **...** matches
    for match in re.finditer(r"\*\*(.*?)\*\*", content):
        start = f"1.0 + {match.start()} chars"
        end = f"1.0 + {match.end()} chars"
        
        # Apply bold tag to the inner text
        # We target the group(1) which is inside the asterisks
        inner_start = f"1.0 + {match.start() + 2} chars"
        inner_end = f"1.0 + {match.end() - 2} chars"
        
        txt.tag_add("bold", inner_start, inner_end)
        
        # Mark asterisks for deletion later (to not mess up indices during loop)
        # Note: In a real editor we'd hide them, but deleting is simpler for read-only
        pass 

    # Cleanup asterisks (simple pass)
    # We do this backwards to avoid index shifting
    count = tk.IntVar()
    while True:
        pos = txt.search("**", "1.0", stopindex=tk.END, count=count)
        if not pos: break
        txt.delete(pos, f"{pos} + 2 chars")

    # B. Headers (Lines ending in ':')
    # Loop through lines to find headers
    num_lines = int(txt.index('end-1c').split('.')[0])
    for i in range(1, num_lines + 1):
        line_text = txt.get(f"{i}.0", f"{i}.end")
        if (line_text.strip().endswith(":") and line_text[0].isdigit()) or "HOW TO READ THIS" in line_text:
             txt.tag_add("header", f"{i}.0", f"{i}.end")
        
        # Style the divider line
        if "----" in line_text:
             txt.tag_add("divider", f"{i}.0", f"{i}.end")

    txt.configure(state='disabled') # Read-only
    root.mainloop()

thread = threading.Thread(target=launch_guide_window)
thread.daemon = True 
thread.start()

plt.show()