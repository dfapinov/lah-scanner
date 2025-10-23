#!/usr/bin/env python3
"""
Specular first-reflection geometry (equal heights over a flat boundary)

Usage:
  python crossover_check_util.py --bound 1.0 --source 0.5
  python crossover_check_util.py --bound 1.0 --source 0.5 --save plot.png

Args:
  --bound   : Perpendicular distance to boundary from source and mic [m]
  --source  : Direct source–mic distance D [m]
  --save    : Optional path to save the plot (PNG).

Hard-coded:
  c = 343.0 m/s   (speed of sound)

Outputs:
  - Direct & reflected path lengths and TOFs
  - Extra delay (refl - direct)  Δt = (R - D)/c
  - Recommended short gate T_short = Δt
  - Recommended 4× wavelength crossover frequency f_4λ = 4 / T_short
  - Plot: source (red), mic (blue), boundary (near bottom), and specular reflection path (dashed)

 Calculation Overview
 --------------------
1. Geometry setup:
      Source and mic both at height h above the boundary, separated by distance D.
      Boundary assumed flat (e.g., floor), so reflection is specular.

2. Path lengths:
      Direct path:     L_direct = D
      Reflected path:  L_refl = sqrt(D^2 + (2h)^2)

3. Time of flight (TOF):
      Direct:  t_direct = L_direct / c
      Reflected: t_refl = L_refl / c

4. Reflection delay:
      Δt = t_refl - t_direct
      (This is the time gap between direct and reflected arrivals.)

5. Gate duration:
      T_short = Δt
     (Gate should end just before the reflection arrives.)

6. Crossover frequency (4× wavelength rule):
      f_4λ = 4 / T_short
      (Lowest frequency that fits 4 complete wave cycles within the short gate. Multiple cycles needed for fft resolution.)

7. Outputs:
      • Path lengths and TOFs
      • Reflection delay and short gate time
      • Recommended crossover frequency (f_4λ)
      • Plot of geometry showing source, mic, boundary, and reflection path

"""

import argparse
import math
import sys
import matplotlib.pyplot as plt

# Hard-coded physical parameter
C_SOUND = 343.0  # m/s

def main():
    ap = argparse.ArgumentParser(description="Compute first-reflection timing and 4×-wavelength crossover.")
    ap.add_argument("--bound", type=float, required=True,
                    help="Perpendicular distance to boundary for both source and mic [m].")
    ap.add_argument("--source", type=float, required=True,
                    help="Direct source–mic distance D [m].")
    ap.add_argument("--save", type=str, default=None,
                    help="Optional file path to save the plot (e.g., plot.png).")
    args = ap.parse_args()

    h = args.bound
    D = args.source

    # Sanity checks
    if h <= 0 or D <= 0:
        print("Error: --bound and --source must be positive.", file=sys.stderr)
        sys.exit(1)

    # Positions (equal heights): source at (0,h), mic at (D,h); boundary is y=0
    src_xy = (0.0, h)
    mic_xy = (D, h)
    direct_len = D

    # Specular reflection geometry (equal heights → reflection point at x = D/2)
    refl_len = math.hypot(D, 2*h)           # sqrt(D^2 + (2h)^2)
    direct_tof = direct_len / C_SOUND
    refl_tof = refl_len / C_SOUND
    extra_delay = refl_tof - direct_tof     # Δt (seconds)
    T_short = extra_delay                    # recommended short gate (s), no extra alpha

    # 4× wavelength rule: need ~4 periods inside the gate
    f_4lambda = 4.0 / T_short if T_short > 0 else float("inf")

    # Reflection point on boundary (y=0)
    x_ref = D / 2.0
    ref_xy = (x_ref, 0.0)

    # Print results
    print("\n=== First-reflection timing (equal heights) ===")
    print(f"Boundary distance (h):          {h:.4f} m")
    print(f"Source–mic distance (D):        {D:.4f} m")
    print(f"Speed of sound (c):             {C_SOUND:.2f} m/s")
    print(f"Direct path length:             {direct_len:.4f} m")
    print(f"Reflection path length:         {refl_len:.4f} m")
    print(f"Direct TOF:                     {direct_tof*1e3:.3f} ms")
    print(f"Reflection TOF:                 {refl_tof*1e3:.3f} ms")
    print(f"Extra delay (refl - direct):    {extra_delay*1e3:.3f} ms")
    print(f"Short gate (T_short = Δt):      {T_short*1e3:.3f} ms")
    print(f"4× wavelength crossover freq:   {f_4lambda:.1f} Hz\n")

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Move boundary near the bottom (about one division above)
    pad_x = max(0.2, 0.1 * D)
    pad_y = 0.1 * h
    y_min = -0.25 * h - pad_y   # a little below the boundary
    y_max = 1.75 * h + pad_y    # space above for the geometry
    ax.set_ylim(y_min, y_max)

    # Boundary line (y=0)
    ax.axhline(0.0, color="black", linewidth=2, label="Boundary")

    # Points
    ax.plot(src_xy[0], src_xy[1], 'o', color='red', label='Source')
    ax.plot(mic_xy[0], mic_xy[1], 'o', color='blue', label='Mic')

    # Direct path (light gray)
    ax.plot([src_xy[0], mic_xy[0]], [src_xy[1], mic_xy[1]],
            '-', color='gray', alpha=0.5, label='Direct path')

    # Specular reflection path (dashed, two segments)
    ax.plot([src_xy[0], ref_xy[0]], [src_xy[1], ref_xy[1]],
            '--', color='black', linewidth=2)
    ax.plot([ref_xy[0], mic_xy[0]], [ref_xy[1], mic_xy[1]],
            '--', color='black', linewidth=2, label='Reflection path')

    # Reflection point marker & label
    ax.plot(ref_xy[0], ref_xy[1], 'x', color='black', markersize=8)
    ax.text(ref_xy[0], y_min + 0.06*(y_max - y_min), "reflection point",
            ha='center', va='bottom', fontsize=8)

    # Axes limits and styling
    ax.set_xlim(-pad_x, D + pad_x)
    ax.set_title("Specular Reflection Geometry (Equal Heights)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m] (boundary at y=0)")
    ax.grid(True, linestyle=':', alpha=0.6)

    # Legend on the LEFT
    ax.legend(loc="upper left", framealpha=0.9)

    # Stats textbox on the RIGHT
    text = (
        f"D = {D:.3f} m\n"
        f"h = {h:.3f} m\n"
        f"Direct TOF = {direct_tof*1e3:.2f} ms\n"
        f"Refl TOF = {refl_tof*1e3:.2f} ms\n"
        f"Δt = {extra_delay*1e3:.2f} ms\n"
        f"T_short = {T_short*1e3:.2f} ms\n"
        f"f_4λ ≈ {f_4lambda:.0f} Hz"
    )
    ax.text(0.98, 0.98, text, transform=ax.transAxes, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=160)
        print(f"Saved plot to: {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

r"""
       ____  __  __ ___ _____ ______   __   
      |  _ \|  \/  |_ _|_   _|  _ \ \ / /   
      | | | | |\/| || |  | | | |_) \ V /    
      | |_| | |  | || |  | | |  _ < | |     
     _|____/|_| _|_|___| |_| |_|_\_\|_|   __
    |  ___/ \  |  _ \_ _| \ | |/ _ \ \   / /
    | |_ / _ \ | |_) | ||  \| | | | \ \ / / 
    |  _/ ___ \|  __/| || |\  | |_| |\ V /  
    |_|/_/   \_\_|  |___|_| \_|\___/  \_/   
                       
                     ███                 
                   █████         ███     
                 ███████         ████    
               █████████    ██    ████   
     ███████████████████    ████   ████  
    ████████████████████     ███   ████  
    ████████████████████      ██    ████ 
    ████████████████████      ███    ████ 
    ████████████████████      ███    ████ 
    ████████████████████     ███    ████  
     ███████████████████    ████   ████  
              ██████████    ███   ████   
                ████████    █    ████    
                  ██████         ███     
                     ███    
"""