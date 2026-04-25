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

def plan_path(
    input_data,
    cap_tol_mm,
    output_path=None,
    delta_theta_deg=7.5,
    side_snake_start="up",
    show_replay=False
):
    """
    Plans the measurement path using a θ-binned snake strategy.
    """
    # ── Load data ────────────────────────────────────────────────────────────────
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, dict):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data must be a file path (str), dictionary, or pandas DataFrame.")

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

    # Cap detection
    z_min, z_max = float(z.min()), float(z.max())
    cap_tol = cap_tol_mm / 1000.0
    on_top_cap    = np.isclose(z, z_max, atol=cap_tol)
    on_bottom_cap = np.isclose(z, z_min, atol=cap_tol)
    on_side       = ~(on_top_cap | on_bottom_cap)

    # ── Binning Logic (-180 to 180) ──────────────────────────────────────────────
    # Shift +180 to create 0-360 scale for binning only
    bin_width = float(delta_theta_deg)
    num_bins  = int(np.ceil(360.0 / bin_width))

    phi_shifted = phi_d + 180.0
    theta_bins = np.floor(phi_shifted / bin_width).astype(int)
    theta_bins = np.clip(theta_bins, 0, num_bins - 1)

    # ── Build Order ──────────────────────────────────────────────────────────────
    order_indices = []
    snake_up = (side_snake_start.lower() == "up")

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

    if output_path is not None:
        out.to_csv(output_path, index=False)
        
    if show_replay:
        import tkinter as tk
        from coord_viewer_gui import TkinterCoordApp
        root = tk.Tk()
        app = TkinterCoordApp(root)
        app.engine.load_data(out)
        root.mainloop()
        
    return out.to_dict(orient='list')

if __name__ == "__main__":
    from config_capture import (
        INPUT_PATH_PLAN,
        OUTPUT_PATH_PLAN,
        DELTA_THETA_DEG,
        CAP_TOL_MM,
        SIDE_SNAKE_START
    )
    
    plan_path(
        input_data=INPUT_PATH_PLAN,
        output_path=OUTPUT_PATH_PLAN,
        cap_tol_mm=CAP_TOL_MM,
        delta_theta_deg=DELTA_THETA_DEG,
        side_snake_start=SIDE_SNAKE_START,
        show_replay=True
    )