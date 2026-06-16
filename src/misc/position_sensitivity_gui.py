#!/usr/bin/env python3
"""
Microphone position sensitivity sweeps for the SHE pipeline.

The experiment works by copying an existing Stage 1 NPZ dataset and rewriting
the coordinate-bearing keys in its complex-data dictionary. The acoustic data is
unchanged; only the coordinates reported by the robot are perturbed. Optionally,
Stage 2 is rerun for each perturbation before Stage 3/4.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "process"))
if PROCESS_DIR not in sys.path:
    sys.path.insert(0, PROCESS_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import schema
from stage3_optimize_she_settings import run_open_branch_optimizer
from stage4_run_she_solve import run_she_solve
from utils import parse_coords_from_filename, spherical_to_cartesian


_COORD_RE = re.compile(
    r"^(?P<prefix>.*?)_r(?P<rmm>-?\d+(?:p\d+)*)_ph(?P<ph>-?\d+(?:p\d+)*)_z(?P<zmm>-?\d+(?:p\d+)*)(?P<suffix>(?:_.*)?(?:\.[^.]+)?)$",
    re.IGNORECASE,
)


@dataclass
class PerturbationCase:
    label: str
    fixed_r_mm: float = 0.0
    fixed_phi_deg: float = 0.0
    fixed_z_mm: float = 0.0
    jitter_r_mm: float = 0.0
    jitter_phi_deg: float = 0.0
    jitter_z_mm: float = 0.0
    speed_of_sound: Optional[float] = None
    seed: Optional[int] = None


def _decode_number(value: str) -> float:
    return float(value.replace("p", ".").replace("n", "-"))


def _encode_coord_number(value: float, decimals: int = 3) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text.replace(".", "p")


def _format_coord_filename(original_name: str, r_cyl_mm: float, phi_deg: float, z_mm: float) -> str:
    match = _COORD_RE.match(original_name)
    if not match:
        raise ValueError(f"Filename does not match coordinate pattern: {original_name}")
    return (
        f"{match.group('prefix')}"
        f"_r{_encode_coord_number(r_cyl_mm)}"
        f"_ph{_encode_coord_number(phi_deg)}"
        f"_z{_encode_coord_number(z_mm)}"
        f"{match.group('suffix')}"
    )


def _dedupe_filename(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    base, ext = os.path.splitext(name)
    idx = 1
    while True:
        candidate = f"{base}_sens{idx}{ext}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def _copy_npz_with_perturbed_coordinates(source_npz: str, dest_npz: str, case: PerturbationCase) -> Dict:
    loaded = np.load(source_npz, allow_pickle=True)
    try:
        data_dict = loaded[schema.COMPLEX_DATA].item()
        rng = np.random.default_rng(case.seed)
        used_names: set[str] = set()
        new_data = {}
        rows = []

        for old_name, values in data_dict.items():
            r_sph, theta, phi = parse_coords_from_filename(old_name)
            x, y, z = spherical_to_cartesian(r_sph, theta, phi)
            r_cyl_mm = math.hypot(x, y) * 1000.0
            phi_deg = math.degrees(math.atan2(y, x))
            z_mm = z * 1000.0

            r_cyl_mm += case.fixed_r_mm
            phi_deg += case.fixed_phi_deg
            z_mm += case.fixed_z_mm
            if case.jitter_r_mm:
                r_cyl_mm += rng.uniform(-case.jitter_r_mm, case.jitter_r_mm)
            if case.jitter_phi_deg:
                phi_deg += rng.uniform(-case.jitter_phi_deg, case.jitter_phi_deg)
            if case.jitter_z_mm:
                z_mm += rng.uniform(-case.jitter_z_mm, case.jitter_z_mm)
            r_cyl_mm = max(0.001, r_cyl_mm)

            if (
                case.fixed_r_mm == 0.0 and case.fixed_phi_deg == 0.0 and case.fixed_z_mm == 0.0
                and case.jitter_r_mm == 0.0 and case.jitter_phi_deg == 0.0 and case.jitter_z_mm == 0.0
            ):
                new_name = _dedupe_filename(old_name, used_names)
            else:
                new_name = _dedupe_filename(_format_coord_filename(old_name, r_cyl_mm, phi_deg, z_mm), used_names)
            parse_coords_from_filename(new_name)
            new_data[new_name] = values
            rows.append((old_name, new_name))

        save_dict = {key: loaded[key] for key in loaded.files if key != schema.COMPLEX_DATA}
        save_dict[schema.COMPLEX_DATA] = new_data
        os.makedirs(os.path.dirname(dest_npz), exist_ok=True)
        np.savez(dest_npz, **save_dict)
        return {"renamed": rows, "count": len(rows)}
    finally:
        if hasattr(loaded, "close"):
            loaded.close()


def _parse_csv_numbers(text: str) -> List[float]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def _range_values(start: str, stop: str, step: str) -> List[float]:
    start_val = float(start)
    stop_val = float(stop)
    step_mag = abs(float(step))
    if step_mag <= 0:
        raise ValueError("Range step must be non-zero.")
    if start_val == stop_val:
        return [round(start_val, 10)]

    values = []
    current = start_val
    eps = step_mag * 1e-9
    step_val = step_mag if stop_val > start_val else -step_mag
    if step_val > 0:
        while current <= stop_val + eps:
            values.append(round(current, 10))
            current += step_val
    else:
        while current >= stop_val - eps:
            values.append(round(current, 10))
            current += step_val
    return values


def build_cases(
    fixed_r_start_mm: str = "0",
    fixed_r_stop_mm: str = "0",
    fixed_r_step_mm: str = "1",
    fixed_phi_start_deg: str = "0",
    fixed_phi_stop_deg: str = "0",
    fixed_phi_step_deg: str = "1",
    fixed_z_start_mm: str = "0",
    fixed_z_stop_mm: str = "0",
    fixed_z_step_mm: str = "1",
    jitter_r_mm: str = "0",
    jitter_phi_deg: str = "0",
    jitter_z_mm: str = "0",
    speed_start_mps: str = "343",
    speed_stop_mps: str = "343",
    speed_step_mps: str = "1",
    baseline_speed_mps: float = 343.0,
    random_trials: int = 3,
    seed: int = 12345,
) -> List[PerturbationCase]:
    cases = [PerturbationCase(label="baseline")]

    def add_fixed(label: str, **kwargs):
        active = False
        for value in kwargs.values():
            arr = np.asarray(value, dtype=float)
            if np.any(np.abs(arr) > 0):
                active = True
                break
        if active:
            cases.append(PerturbationCase(label=label, **kwargs))

    for value in _range_values(fixed_r_start_mm, fixed_r_stop_mm, fixed_r_step_mm):
        add_fixed(f"fixed_r_{value:g}mm", fixed_r_mm=value)
    for value in _range_values(fixed_phi_start_deg, fixed_phi_stop_deg, fixed_phi_step_deg):
        add_fixed(f"fixed_phi_{value:g}deg", fixed_phi_deg=value)
    for value in _range_values(fixed_z_start_mm, fixed_z_stop_mm, fixed_z_step_mm):
        add_fixed(f"fixed_z_{value:g}mm", fixed_z_mm=value)

    for value in _range_values(speed_start_mps, speed_stop_mps, speed_step_mps):
        if not math.isclose(value, baseline_speed_mps, rel_tol=0.0, abs_tol=1e-9):
            cases.append(PerturbationCase(label=f"speed_{_encode_coord_number(value)}mps", speed_of_sound=value))

    jitter_specs = [
        ("jitter_r", "jitter_r_mm", jitter_r_mm, None),
        ("jitter_phi", "jitter_phi_deg", jitter_phi_deg, None),
        ("jitter_z", "jitter_z_mm", jitter_z_mm, None),
    ]
    for prefix, field, csv_text, axis in jitter_specs:
        for value in _parse_csv_numbers(csv_text):
            if abs(value) <= 0:
                continue
            for trial in range(max(1, random_trials)):
                kwargs = {"seed": seed + len(cases) * 1009 + trial}
                kwargs[field] = value
                cases.append(PerturbationCase(label=f"{prefix}_{value:g}_trial{trial + 1}", **kwargs))

    return cases


def calc_stage3_ratio(option: Optional[dict]) -> float:
    if not option:
        return float("nan")
    return float(option.get("ratio", float("nan")))


def run_position_sensitivity(
    source_npz: str,
    output_root: str,
    cases: Iterable[PerturbationCase],
    stage2_settings: Optional[dict],
    stage3_order_range: tuple[int, int],
    rerun_stage3_each_case: bool,
    stage4_settings: dict,
    project_name: str = "project",
    speed_of_sound: float = 343.0,
    stage23_kr_offset: float = 2.0,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(output_root, f"position_sensitivity_{timestamp}")
    data_dir = os.path.join(run_root, "datasets")
    coeff_dir = os.path.join(run_root, "coefficients")
    plot_dir = os.path.join(run_root, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(coeff_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Sensitivity source NPZ: {source_npz}")
    print(f"Stage 2/3 KR offset: {stage23_kr_offset}")
    print(f"Stage 4 KR offset: {stage4_settings.get('kr_offset', 2.0)}")

    summary = []
    residual_curves = []
    stage2_error_curves = []
    stage2_origin_curves = []
    stage3_options = None
    stage3_step1 = None
    stage3_input_filename = None

    cases = list(cases)
    fixed_stage4_n = int(stage4_settings["target_n_max"])
    for idx, case in enumerate(cases, start=1):
        case_speed_of_sound = case.speed_of_sound if case.speed_of_sound is not None else speed_of_sound
        print(f"\n--- Position sensitivity case {idx}/{len(cases)}: {case.label} ---")
        print(f"Speed of sound for this case: {case_speed_of_sound:g} m/s")
        case_npz = os.path.join(data_dir, f"{project_name}_{case.label}.npz")
        copy_info = _copy_npz_with_perturbed_coordinates(source_npz, case_npz, case)
        if case.label == "baseline":
            _print_baseline_coordinate_audit(source_npz, case_npz, copy_info["renamed"])

        input_dir = os.path.dirname(case_npz)
        input_filename = os.path.basename(case_npz)

        if not stage2_settings:
            raise ValueError("Stage 2 settings are required for position sensitivity sweeps.")
        print("Running Stage 2 origin search on perturbed coordinates.")
        stage2_result = _run_stage2_for_case(
            input_dir=input_dir,
            input_filename=input_filename,
            stage2_settings=stage2_settings,
            speed_of_sound=case_speed_of_sound,
            kr_offset=stage23_kr_offset,
        )
        stage2_error_curves.append((case.label, stage2_result["target_freqs"], stage2_result["target_errors"]))
        stage2_origin_curves.append((case.label, stage2_result["freqs"], stage2_result["origins_mm"]))
        case_plot_dir = os.path.join(plot_dir, case.label)
        _plot_case_stage2(
            target_freqs=stage2_result["target_freqs"],
            target_errors=stage2_result["target_errors"],
            freqs=stage2_result["freqs"],
            origins_mm=stage2_result["origins_mm"],
            case_label=case.label,
            save_dir=case_plot_dir,
        )

        if rerun_stage3_each_case or stage3_options is None:
            stage3_result = run_open_branch_optimizer(
                input_dir_opti=input_dir,
                input_filename_opti=input_filename,
                test_order_range=stage3_order_range,
                test_start_db_range=(-20.0, -60.0),
                test_lambda_range=(0.0000001, 0.01),
                test_db_transition_span=20.0,
                use_optimized_origins=True,
                speed_of_sound=case_speed_of_sound,
                kr_offset=stage23_kr_offset,
            )
            stage3_options = stage3_result.get("options", {}) if isinstance(stage3_result, dict) else {}
            stage3_step1 = stage3_result.get("step1", {}) if isinstance(stage3_result, dict) else {}
            stage3_input_filename = input_filename
        else:
            print(f"Reusing Stage 3 selection from {stage3_input_filename}.")

        best = stage3_options.get("best_sfs") or stage3_options.get("balanced") or {}
        selected_n = fixed_stage4_n
        stage3_ratio = calc_stage3_ratio(best)
        stage3_residual = float(best.get("err", float("nan"))) if best else float("nan")
        _plot_case_stage3(stage3_step1, stage3_options, case.label, case_plot_dir)
        _write_case_stage3_report(stage3_step1, stage3_options, case.label, case_plot_dir, case_speed_of_sound)

        case_coeff_dir = os.path.join(coeff_dir, case.label)
        results = run_she_solve(
            input_filename_she=input_filename,
            output_filename_she=f"{project_name}_{case.label}_coefficients.h5",
            input_dir_she=input_dir,
            output_dir_she=case_coeff_dir,
            target_n_max=selected_n,
            use_manual_table=stage4_settings.get("use_manual_table", False),
            manual_order_table=stage4_settings.get("manual_order_table", {}),
            noise_floor_start_db=stage4_settings.get("noise_floor_start_db", -30.0),
            noise_floor_max_db=stage4_settings.get("noise_floor_max_db", -40.0),
            max_lambda=stage4_settings.get("max_lambda", 0.000001),
            condition_metrics=True,
            use_optimized_origins=True,
            save_to_disk=True,
            speed_of_sound=case_speed_of_sound,
            kr_offset=stage4_settings.get("kr_offset", 2.0),
            jobs=stage4_settings.get("jobs"),
            show_plot=False,
        )

        freqs = results["freqs"]
        pct_error = results["pct_error"]
        residual_curves.append((case.label, freqs, pct_error))
        _plot_case_stage4(
            freqs=freqs,
            pct_error=pct_error,
            cond=results["cond"],
            n_used=results["N_used"],
            case_label=case.label,
            save_dir=case_plot_dir,
        )
        summary.append(
            {
                "case": case.label,
                "speed_of_sound_mps": float(case_speed_of_sound),
                "stage2_mean_origin_error_pct": float(np.mean(stage2_result["target_errors"])),
                "stage2_median_origin_error_pct": float(np.median(stage2_result["target_errors"])),
                "stage2_max_origin_error_pct": float(np.max(stage2_result["target_errors"])),
                "stage2_origin_x_mean_mm": float(np.mean(stage2_result["origins_mm"][:, 0])),
                "stage2_origin_y_mean_mm": float(np.mean(stage2_result["origins_mm"][:, 1])),
                "stage2_origin_z_mean_mm": float(np.mean(stage2_result["origins_mm"][:, 2])),
                "stage3_best_sfs_n": int(best.get("n", -1)) if best else -1,
                "stage3_int_ext_db": stage3_ratio,
                "stage3_residual_pct": stage3_residual,
                "stage4_fixed_target_n": selected_n,
                "stage4_mean_residual_pct": float(np.mean(pct_error)),
                "stage4_median_residual_pct": float(np.median(pct_error)),
                "stage4_p95_residual_pct": float(np.percentile(pct_error, 95)),
                "stage4_max_residual_pct": float(np.max(pct_error)),
            }
        )

    summary_csv = os.path.join(run_root, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    _plot_stage4_residuals(residual_curves, os.path.join(plot_dir, "stage4_residual_comparison.png"))
    _plot_stage2_errors(stage2_error_curves, os.path.join(plot_dir, "stage2_origin_error_comparison.png"))
    _plot_stage2_origins(stage2_origin_curves, os.path.join(plot_dir, "stage2_origin_xyz_comparison.png"))
    _plot_stage3_ratios(summary, os.path.join(plot_dir, "stage3_int_ext_comparison.png"))
    print(f"\nPosition sensitivity sweep complete: {run_root}")
    return {"run_root": run_root, "summary_csv": summary_csv, "summary": summary}


def _print_baseline_coordinate_audit(source_npz, case_npz, renamed_rows):
    from utils import load_and_parse_npz

    src = load_and_parse_npz(source_npz)
    dst = load_and_parse_npz(case_npz)
    if len(src["r_arr"]) != len(dst["r_arr"]):
        print(f"Baseline coordinate audit: point count changed {len(src['r_arr'])} -> {len(dst['r_arr'])}")
        return
    dr_mm = np.max(np.abs(src["r_arr"] - dst["r_arr"])) * 1000.0
    dth_deg = np.max(np.abs(np.degrees(src["th_arr"] - dst["th_arr"])))
    dph = np.angle(np.exp(1j * (src["ph_arr"] - dst["ph_arr"])))
    dph_deg = np.max(np.abs(np.degrees(dph)))
    print(
        "Baseline coordinate audit: "
        f"max spherical delta r={dr_mm:.6f} mm, theta={dth_deg:.6f} deg, phi={dph_deg:.6f} deg"
    )
    if src["origins_mm"] is not None:
        print("Baseline source already contained origins_mm; sensitivity Stage 2 will overwrite them.")


def _run_stage2_for_case(input_dir, input_filename, stage2_settings, speed_of_sound, kr_offset):
    from stage2_centre_origin import export_interpolated_origins, run_origin_search

    res = run_origin_search(
        input_dir_origins=input_dir,
        input_filename_origins=input_filename,
        output_filename_origins=input_filename,
        tweeter_coords_mm=stage2_settings["tweeter_coords_mm"],
        octave_resolution=stage2_settings["octave_resolution"],
        freq_start_hz=stage2_settings["freq_start_hz"],
        freq_end_hz=stage2_settings["freq_end_hz"],
        initial_simplex_step=stage2_settings["initial_simplex_step"],
        max_iterations=stage2_settings["max_iterations"],
        x_bounds=stage2_settings["x_bounds"],
        y_bounds=stage2_settings["y_bounds"],
        z_bounds=stage2_settings["z_bounds"],
        grid_res_mm=stage2_settings["grid_res_mm"],
        target_n_max_origins=stage2_settings["target_n_max_origins"],
        manual_order_table=None,
        save_to_disk=False,
        plot_results_origins=False,
        speed_of_sound=speed_of_sound,
        enable_full_grid_scan=False,
        kr_offset=kr_offset,
        return_state=True,
    )

    sweep_results, f_all, _keys, _d_dict, _geom, _cfg, data = res
    history_freq = []
    history_x = []
    history_y = []
    history_z = []
    target_errors = []

    for f_hz in sorted(sweep_results.keys()):
        row = sweep_results[f_hz]
        if row.get("final_c") is None:
            continue
        history_freq.append(float(f_hz))
        history_x.append(float(row["final_c"][0]))
        history_y.append(float(row["final_c"][1]))
        history_z.append(float(row["final_c"][2]))
        target_errors.append(float(row.get("error", float("nan"))))

    origins_full = export_interpolated_origins(
        history_freq,
        history_x,
        history_y,
        history_z,
        f_all,
        data,
        input_dir,
        input_filename,
        True,
    )
    if origins_full is None:
        raise RuntimeError("Stage 2 did not produce any optimized origins.")

    saved_path = os.path.join(input_dir, input_filename)
    with np.load(saved_path, allow_pickle=True) as saved:
        if schema.ORIGINS_MM not in saved:
            raise RuntimeError(f"Stage 2 origins were not saved to {saved_path}")
        saved_origins = saved[schema.ORIGINS_MM]
        if saved_origins.shape != origins_full.shape or not np.all(np.isfinite(saved_origins)):
            raise RuntimeError(f"Stage 2 origins in {saved_path} are missing or invalid.")

    means = np.mean(origins_full, axis=0)
    spans = np.ptp(origins_full, axis=0)
    print(
        "Verified Stage 2 origins saved: "
        f"mean=({means[0]:.2f}, {means[1]:.2f}, {means[2]:.2f}) mm, "
        f"span=({spans[0]:.2f}, {spans[1]:.2f}, {spans[2]:.2f}) mm"
    )

    return {
        "target_freqs": np.array(history_freq, dtype=float),
        "target_errors": np.array(target_errors, dtype=float),
        "freqs": np.array(f_all, dtype=float),
        "origins_mm": np.array(origins_full, dtype=float),
    }


def _plot_stage4_residuals(curves, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, freqs, pct_error in curves:
        ax.semilogx(freqs, pct_error, linewidth=1.3, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Stage 4 residual fit error (%)")
    ax.set_title("Stage 4 Residual Error vs Position Perturbation")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_stage2_errors(curves, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, freqs, errors in curves:
        ax.semilogx(freqs, errors, marker="o", linewidth=1.2, markersize=3, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Stage 2 origin-search residual error (%)")
    ax.set_title("Stage 2 Origin Search Error vs Position Perturbation")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_stage2_origins(curves, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axis_labels = ["X origin (mm)", "Y origin (mm)", "Z origin (mm)"]
    titles = ["Stage 2 Acoustic Origin X", "Stage 2 Acoustic Origin Y", "Stage 2 Acoustic Origin Z"]

    for axis_idx, ax in enumerate(axes):
        for label, freqs, origins_mm in curves:
            ax.semilogx(freqs, origins_mm[:, axis_idx], linewidth=1.1, label=label)
        ax.set_ylabel(axis_labels[axis_idx])
        ax.set_title(titles[axis_idx])
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_case_stage2(target_freqs, target_errors, freqs, origins_mm, case_label, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(target_freqs, target_errors, marker="o", linewidth=1.2, markersize=3, color="#1f77b4")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Origin-search residual error (%)")
    ax.set_title(f"Stage 2 Origin Search Error: {case_label}")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "stage2_origin_error.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axis_labels = ["X origin (mm)", "Y origin (mm)", "Z origin (mm)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for axis_idx, ax in enumerate(axes):
        ax.semilogx(freqs, origins_mm[:, axis_idx], linewidth=1.2, color=colors[axis_idx])
        ax.set_ylabel(axis_labels[axis_idx])
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
    axes[0].set_title(f"Stage 2 Acoustic Origin XYZ: {case_label}")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "stage2_origin_xyz.png"), dpi=150)
    plt.close(fig)


def _plot_case_stage3(step1, options, case_label, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    orders = np.asarray(step1.get("orders", []), dtype=float)
    ratios = np.asarray(step1.get("ratios", []), dtype=float)
    residuals = np.asarray(step1.get("residuals", []), dtype=float)
    if orders.size == 0:
        return

    fig, ax_ratio = plt.subplots(figsize=(10, 5))
    ax_ratio.plot(orders, ratios, marker="o", linewidth=1.3, color="#4c78a8", label="Int/Ext ratio")
    ax_ratio.axhline(15.0, color="#d62728", linestyle="--", linewidth=1.0, label="15 dB reference")
    ax_ratio.set_xlabel("Order N")
    ax_ratio.set_ylabel("Int/Ext ratio (dB)")
    ax_ratio.grid(True, linestyle="--", alpha=0.35)

    ax_resid = ax_ratio.twinx()
    ax_resid.plot(orders, residuals, marker="s", linewidth=1.2, color="#f58518", label="Residual")
    ax_resid.set_ylabel("Residual (%)")

    for option in options.values():
        n_val = option.get("n")
        if n_val is not None:
            ax_ratio.axvline(float(n_val), color="#54a24b", linestyle=":", linewidth=1.0, alpha=0.8)

    lines = ax_ratio.get_lines() + ax_resid.get_lines()
    labels = [line.get_label() for line in lines]
    ax_ratio.legend(lines, labels, loc="best")
    ax_ratio.set_title(f"Stage 3 Order Sweep: {case_label}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "stage3_order_sweep.png"), dpi=150)
    plt.close(fig)


def _write_case_stage3_report(step1, options, case_label, save_dir, speed_of_sound=None):
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, "stage3_results.txt")
    orders = step1.get("orders", [])
    ratios = step1.get("ratios", [])
    residuals = step1.get("residuals", [])
    deltas = step1.get("delta_ratios", [])

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"Stage 3 results for position sensitivity case: {case_label}\n")
        if speed_of_sound is not None:
            fh.write(f"Speed of sound: {float(speed_of_sound):g} m/s\n")
        fh.write("=" * 72 + "\n\n")
        fh.write("Step 1 order sweep\n")
        fh.write(f"{'Order N':>8}  {'Int/Ext Ratio (dB)':>20}  {'Residual (%)':>14}  {'Delta Ratio':>12}\n")
        fh.write("-" * 72 + "\n")
        for n, ratio, residual, delta in zip(orders, ratios, residuals, deltas):
            fh.write(f"{int(n):>8}  {float(ratio):>20.6f}  {float(residual):>14.6f}  {float(delta):>12.6f}\n")

        fh.write("\nSuggested options\n")
        fh.write("-" * 72 + "\n")
        for key, option in options.items():
            fh.write(f"{key}: {option.get('label', '')}\n")
            fh.write(f"  order_n: {option.get('n')}\n")
            fh.write(f"  int_ext_ratio_db: {option.get('ratio')}\n")
            fh.write(f"  residual_pct: {option.get('err')}\n")
            fh.write(f"  noise_floor_start_db: {option.get('st')}\n")
            fh.write(f"  noise_floor_max_db: {option.get('mx')}\n")
            fh.write(f"  max_lambda: {option.get('lam')}\n")
            warning = option.get("warning")
            if warning:
                fh.write(f"  warning: {warning}\n")
            fh.write("\n")


def _plot_case_stage4(freqs, pct_error, cond, n_used, case_label, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(freqs, pct_error, color="#1f77b4", linewidth=1.4)
    ax.axhline(10.0, color="#d62728", linestyle="--", linewidth=1.0, label="10% reference")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residual fit error (%)")
    ax.set_title(f"Stage 4 Residual Error: {case_label}")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "stage4_residual_error.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(freqs, cond, color="#2ca02c", linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Condition number")
    ax.set_title(f"Stage 4 Condition Number: {case_label}")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "stage4_condition_number.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.semilogx(freqs, n_used, color="#9467bd", linewidth=1.4)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Order N")
    ax.set_title(f"Stage 4 Order N: {case_label}")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "stage4_order_n.png"), dpi=150)
    plt.close(fig)


def _plot_stage3_ratios(summary, save_path):
    labels = [row["case"] for row in summary]
    ratios = [row["stage3_int_ext_db"] for row in summary]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.65), 5))
    ax.bar(range(len(labels)), ratios, color="#4c78a8")
    ax.axhline(15.0, color="#d62728", linestyle="--", linewidth=1.0, label="15 dB reference")
    ax.set_ylabel("Stage 3 Int/Ext ratio (dB)")
    ax.set_title("Stage 3 Sound Field Separation vs Position Perturbation")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


class QueueRedirector:
    def __init__(self, queue_obj):
        self.queue = queue_obj

    def write(self, text):
        if text:
            self.queue.put(text)

    def flush(self):
        pass


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.inner = ttk.Frame(self.canvas)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)


class PositionSensitivityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HALS Position Sensitivity")
        self.geometry("1180x760")
        self.project_settings = {}
        self.manual_order_table = {}
        self.vars = {}
        self.cli_queue = queue.Queue()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = QueueRedirector(self.cli_queue)
        sys.stderr = QueueRedirector(self.cli_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._load_default_project_if_found()
        self.after(40, self._drain_cli_queue)

    def _build_ui(self):
        root = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True)

        left = ScrollableFrame(root)
        right = ttk.Frame(root)
        root.add(left, weight=3)
        root.add(right, weight=2)

        container = left.inner
        self._build_project_frame(container)
        self._build_offset_frame(container)
        self._build_jitter_frame(container)
        self._build_speed_frame(container)
        self._build_stage2_frame(container)
        self._build_stage3_frame(container)
        self._build_stage4_frame(container)

        actions = ttk.Frame(container, padding=(10, 8))
        actions.pack(fill=tk.X)
        self.preview_button = ttk.Button(actions, text="Preview Cases", command=self._preview_cases)
        self.preview_button.pack(side=tk.LEFT)
        self.run_button = ttk.Button(actions, text="Run Sweep", command=self._run_sweep)
        self.run_button.pack(side=tk.RIGHT)

        ttk.Label(right, text="Run Log").pack(anchor=tk.W, padx=8, pady=(8, 0))
        self.log_text = tk.Text(right, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _build_project_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Project", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=6)
        frame.columnconfigure(1, weight=1)

        self.vars["project_json"] = tk.StringVar(value="")
        self.vars["project_dir"] = tk.StringVar(value=os.getcwd())
        self.vars["project_name"] = tk.StringVar(value="MySpeaker")
        self.vars["source_npz"] = tk.StringVar(value="")
        self.vars["output_root"] = tk.StringVar(value="")

        self._grid_entry(frame, "Project JSON:", "project_json", 0, browse=self._browse_project_json)
        self._grid_entry(frame, "Project Dir:", "project_dir", 1, browse=self._browse_project_dir)
        self._grid_entry(frame, "Project Name:", "project_name", 2)
        self._grid_entry(frame, "Source NPZ:", "source_npz", 3, browse=self._browse_source_npz)
        self._grid_entry(frame, "Output Root:", "output_root", 4, browse=self._browse_output_root)

    def _build_offset_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Fixed Coordinate Offsets", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=6)
        for row, (label, prefix, unit) in enumerate([
            ("R offset", "fixed_r", "mm"),
            ("Phi offset", "fixed_phi", "deg"),
            ("Z offset", "fixed_z", "mm"),
        ]):
            ttk.Label(frame, text=f"{label} ({unit})").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=3)
            self._mini_entry(frame, f"{prefix}_start", "0", row, 1, "Start")
            self._mini_entry(frame, f"{prefix}_stop", "0", row, 3, "Stop")
            self._mini_entry(frame, f"{prefix}_step", "1", row, 5, "Step")

    def _build_jitter_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Random Position Jitter", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=6)
        self._grid_entry(frame, "R jitter ranges (+/- mm):", "jitter_r_mm", 0, default="0")
        self._grid_entry(frame, "Phi jitter ranges (+/- deg):", "jitter_phi_deg", 1, default="0")
        self._grid_entry(frame, "Z jitter ranges (+/- mm):", "jitter_z_mm", 2, default="0")
        self._grid_entry(frame, "Trials per jitter level:", "random_trials", 3, default="3")
        self._grid_entry(frame, "Random seed:", "random_seed", 4, default="12345")

    def _build_speed_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Speed of Sound Sweep", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=6)
        self._mini_entry(frame, "speed_start_mps", "343", 0, 1, "Start (m/s)")
        self._mini_entry(frame, "speed_stop_mps", "343", 0, 3, "Stop (m/s)")
        self._mini_entry(frame, "speed_step_mps", "1", 0, 5, "Step")

    def _build_stage2_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Stage 2 Origin Search", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=6)
        for idx, (label, key, default) in enumerate([
            ("Tweeter X (mm):", "stage2_tweeter_x", "0.0"),
            ("Tweeter Y (mm):", "stage2_tweeter_y", "0.0"),
            ("Tweeter Z (mm):", "stage2_tweeter_z", "0.0"),
            ("Octave Resolution (1/x):", "stage2_octave_resolution", "6"),
            ("Start Frequency (Hz):", "stage2_freq_start_hz", "20.0"),
            ("End Frequency (Hz):", "stage2_freq_end_hz", "20000.0"),
            ("Initial Simplex Step (mm):", "stage2_initial_simplex_step", "15.0"),
            ("Max Iterations:", "stage2_max_iterations", "50"),
            ("X Bounds:", "stage2_x_bounds", "-500.0, 500.0"),
            ("Y Bounds:", "stage2_y_bounds", "-500.0, 500.0"),
            ("Z Bounds:", "stage2_z_bounds", "-500.0, 500.0"),
            ("Grid Res (mm):", "stage2_grid_res_mm", "15.0"),
            ("Max Harmonic Order (N):", "stage2_target_n_max_origins", "4"),
        ]):
            self._grid_entry(frame, label, key, idx, default=default)

    def _build_stage3_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Stage 3 Order Test", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=6)
        self._grid_entry(frame, "Order range (min, max):", "stage3_order_range", 0, default="2, 15")
        self.vars["rerun_stage3_each_case"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Run Stage 3 for every case", variable=self.vars["rerun_stage3_each_case"]).grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=3
        )

    def _build_stage4_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Stage 4 SHE Solve", padding=10)
        frame.pack(fill=tk.X, padx=8, pady=6)
        for idx, (label, key, default) in enumerate([
            ("Target N Max:", "stage4_target_n_max", "8"),
            ("KR Offset:", "stage4_kr_offset", "2.0"),
            ("Noise Floor Start (dB):", "stage4_noise_floor_start_db", "-30.0"),
            ("Noise Floor Max (dB):", "stage4_noise_floor_max_db", "-40.0"),
            ("Max Lambda:", "stage4_max_lambda", "0.000001"),
        ]):
            self._grid_entry(frame, label, key, idx, default=default)
        self.vars["stage4_use_manual_table"] = tk.BooleanVar(value=False)
        self.vars["stage4_use_optimized_origins"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Use manual order table from project JSON", variable=self.vars["stage4_use_manual_table"]).grid(
            row=5, column=0, columnspan=3, sticky=tk.W, pady=3
        )
        ttk.Checkbutton(frame, text="Use optimized origins", variable=self.vars["stage4_use_optimized_origins"]).grid(
            row=6, column=0, columnspan=3, sticky=tk.W, pady=3
        )

    def _grid_entry(self, parent, label, key, row, default="", browse=None):
        if key not in self.vars:
            self.vars[key] = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        entry = ttk.Entry(parent, textvariable=self.vars[key])
        entry.grid(row=row, column=1, sticky=tk.EW, pady=3)
        parent.columnconfigure(1, weight=1)
        if browse:
            ttk.Button(parent, text="Browse", command=browse).grid(row=row, column=2, sticky=tk.W, padx=(6, 0), pady=3)
        return entry

    def _mini_entry(self, parent, key, default, row, column, label):
        if key not in self.vars:
            self.vars[key] = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=row, column=column - 1, sticky=tk.W, padx=(0, 4), pady=3)
        ttk.Entry(parent, textvariable=self.vars[key], width=10).grid(row=row, column=column, sticky=tk.W, padx=(0, 10), pady=3)

    def _browse_project_json(self):
        path = filedialog.askopenfilename(
            initialdir=self.vars["project_dir"].get() or os.getcwd(),
            title="Select HALS Project JSON",
            filetypes=[("HALS project", "*_project.json"), ("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.vars["project_json"].set(path)
            self._load_project_json(path)

    def _browse_project_dir(self):
        path = filedialog.askdirectory(initialdir=self.vars["project_dir"].get() or os.getcwd())
        if path:
            self.vars["project_dir"].set(path)
            self._refresh_default_paths()

    def _browse_source_npz(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.join(self.vars["project_dir"].get(), "outputs"),
            title="Select Source NPZ",
            filetypes=[("NPZ", "*.npz"), ("All files", "*.*")]
        )
        if path:
            self.vars["source_npz"].set(os.path.relpath(path, self.vars["project_dir"].get()))

    def _browse_output_root(self):
        path = filedialog.askdirectory(initialdir=self.vars["project_dir"].get() or os.getcwd())
        if path:
            self.vars["output_root"].set(path)

    def _load_default_project_if_found(self):
        cwd = os.getcwd()
        candidates = [os.path.join(cwd, name) for name in os.listdir(cwd) if name.endswith("_project.json")]
        if candidates:
            self.vars["project_json"].set(candidates[0])
            self._load_project_json(candidates[0])
        else:
            self._refresh_default_paths()

    def _load_project_json(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            settings = json.load(fh)
        self.project_settings = settings
        project_dir = os.path.dirname(os.path.abspath(path))
        self.vars["project_dir"].set(project_dir)
        self.vars["project_name"].set(str(settings.get("project_name") or os.path.basename(path).replace("_project.json", "")))
        self.manual_order_table = {float(k): int(v) for k, v in settings.get("stage4_manual_table", {}).items()}

        self._apply_project_section("sensitivity_vars", {
            "source_npz": "source_npz",
            "fixed_r_start": "fixed_r_start",
            "fixed_r_stop": "fixed_r_stop",
            "fixed_r_step": "fixed_r_step",
            "fixed_phi_start": "fixed_phi_start",
            "fixed_phi_stop": "fixed_phi_stop",
            "fixed_phi_step": "fixed_phi_step",
            "fixed_z_start": "fixed_z_start",
            "fixed_z_stop": "fixed_z_stop",
            "fixed_z_step": "fixed_z_step",
            "jitter_r_mm": "jitter_r_mm",
            "jitter_phi_deg": "jitter_phi_deg",
            "jitter_z_mm": "jitter_z_mm",
            "random_trials": "random_trials",
            "random_seed": "random_seed",
            "stage3_order_range": "stage3_order_range",
            "speed_start_mps": "speed_start_mps",
            "speed_stop_mps": "speed_stop_mps",
            "speed_step_mps": "speed_step_mps",
        })
        self._apply_project_section("stage2_vars", {
            "tweeter_x": "stage2_tweeter_x",
            "tweeter_y": "stage2_tweeter_y",
            "tweeter_z": "stage2_tweeter_z",
            "octave_resolution": "stage2_octave_resolution",
            "freq_start_hz": "stage2_freq_start_hz",
            "freq_end_hz": "stage2_freq_end_hz",
            "initial_simplex_step": "stage2_initial_simplex_step",
            "max_iterations": "stage2_max_iterations",
            "x_bounds": "stage2_x_bounds",
            "y_bounds": "stage2_y_bounds",
            "z_bounds": "stage2_z_bounds",
            "grid_res_mm": "stage2_grid_res_mm",
            "target_n_max_origins": "stage2_target_n_max_origins",
        })
        self._apply_project_section("stage4_vars", {
            "target_n_max": "stage4_target_n_max",
            "kr_offset": "stage4_kr_offset",
            "noise_floor_start_db": "stage4_noise_floor_start_db",
            "noise_floor_max_db": "stage4_noise_floor_max_db",
            "max_lambda": "stage4_max_lambda",
        })
        stage4_vars = settings.get("stage4_vars", {})
        self.vars["stage4_use_manual_table"].set(bool(stage4_vars.get("use_manual_table", False)))
        self.vars["stage4_use_optimized_origins"].set(bool(stage4_vars.get("use_optimized_origins", True)))
        self._refresh_default_paths()
        print(f"Loaded project: {path}")

    def _apply_project_section(self, section_name, mapping):
        section = self.project_settings.get(section_name, {})
        for source_key, var_key in mapping.items():
            if source_key in section and var_key in self.vars:
                self.vars[var_key].set(str(section[source_key]))

    def _refresh_default_paths(self):
        project_dir = self.vars["project_dir"].get() or os.getcwd()
        project_name = self.vars["project_name"].get().strip() or "project"
        if not self.vars["source_npz"].get().strip():
            self.vars["source_npz"].set(os.path.join("outputs", f"{project_name}_complex_data.npz"))
        if not self.vars["output_root"].get().strip():
            self.vars["output_root"].set(os.path.join(project_dir, "outputs", "position_sensitivity"))

    def _build_cases_from_ui(self):
        return build_cases(
            fixed_r_start_mm=self.vars["fixed_r_start"].get(),
            fixed_r_stop_mm=self.vars["fixed_r_stop"].get(),
            fixed_r_step_mm=self.vars["fixed_r_step"].get(),
            fixed_phi_start_deg=self.vars["fixed_phi_start"].get(),
            fixed_phi_stop_deg=self.vars["fixed_phi_stop"].get(),
            fixed_phi_step_deg=self.vars["fixed_phi_step"].get(),
            fixed_z_start_mm=self.vars["fixed_z_start"].get(),
            fixed_z_stop_mm=self.vars["fixed_z_stop"].get(),
            fixed_z_step_mm=self.vars["fixed_z_step"].get(),
            jitter_r_mm=self.vars["jitter_r_mm"].get(),
            jitter_phi_deg=self.vars["jitter_phi_deg"].get(),
            jitter_z_mm=self.vars["jitter_z_mm"].get(),
            speed_start_mps=self.vars["speed_start_mps"].get(),
            speed_stop_mps=self.vars["speed_stop_mps"].get(),
            speed_step_mps=self.vars["speed_step_mps"].get(),
            baseline_speed_mps=343.0,
            random_trials=int(self.vars["random_trials"].get()),
            seed=int(self.vars["random_seed"].get()),
        )

    def _preview_cases(self):
        try:
            cases = self._build_cases_from_ui()
            print(f"\nConfigured cases: {len(cases)}")
            for case in cases:
                suffix = f"  c={case.speed_of_sound:g} m/s" if case.speed_of_sound is not None else ""
                print(f"  {case.label}{suffix}")
        except Exception as exc:
            messagebox.showerror("Case Preview", str(exc))

    def _run_sweep(self):
        self.run_button.config(state=tk.DISABLED)
        threading.Thread(target=self._run_sweep_thread, daemon=True).start()

    def _run_sweep_thread(self):
        try:
            project_dir = self.vars["project_dir"].get().strip() or os.getcwd()
            source_npz = self.vars["source_npz"].get().strip()
            if not os.path.isabs(source_npz):
                source_npz = os.path.join(project_dir, source_npz)
            if not os.path.exists(source_npz):
                raise FileNotFoundError(f"Source NPZ not found: {source_npz}")

            cases = self._build_cases_from_ui()
            if len(cases) <= 1:
                print("Only the baseline case is configured.")

            result = run_position_sensitivity(
                source_npz=source_npz,
                output_root=self.vars["output_root"].get().strip(),
                cases=cases,
                stage2_settings=self._stage2_settings(),
                stage3_order_range=self._parse_int_bounds(self.vars["stage3_order_range"].get()),
                rerun_stage3_each_case=self.vars["rerun_stage3_each_case"].get(),
                stage4_settings=self._stage4_settings(),
                project_name=self.vars["project_name"].get().strip() or "project",
                speed_of_sound=343.0,
                stage23_kr_offset=2.0,
            )
            print(f"Summary CSV: {result['summary_csv']}")
        except Exception as exc:
            print(f"Error during position sensitivity sweep: {exc}")
        finally:
            self.after(0, lambda: self.run_button.config(state=tk.NORMAL))

    def _stage2_settings(self):
        return {
            "tweeter_coords_mm": (
                float(self.vars["stage2_tweeter_x"].get()),
                float(self.vars["stage2_tweeter_y"].get()),
                float(self.vars["stage2_tweeter_z"].get()),
            ),
            "octave_resolution": 1.0 / float(self.vars["stage2_octave_resolution"].get()),
            "freq_start_hz": float(self.vars["stage2_freq_start_hz"].get()),
            "freq_end_hz": float(self.vars["stage2_freq_end_hz"].get()),
            "initial_simplex_step": float(self.vars["stage2_initial_simplex_step"].get()),
            "max_iterations": int(self.vars["stage2_max_iterations"].get()),
            "x_bounds": self._parse_float_bounds(self.vars["stage2_x_bounds"].get()),
            "y_bounds": self._parse_float_bounds(self.vars["stage2_y_bounds"].get()),
            "z_bounds": self._parse_float_bounds(self.vars["stage2_z_bounds"].get()),
            "grid_res_mm": float(self.vars["stage2_grid_res_mm"].get()),
            "target_n_max_origins": int(self.vars["stage2_target_n_max_origins"].get()),
        }

    def _stage4_settings(self):
        return {
            "target_n_max": int(self.vars["stage4_target_n_max"].get()),
            "kr_offset": float(self.vars["stage4_kr_offset"].get()),
            "use_manual_table": self.vars["stage4_use_manual_table"].get(),
            "manual_order_table": self.manual_order_table,
            "noise_floor_start_db": float(self.vars["stage4_noise_floor_start_db"].get()),
            "noise_floor_max_db": float(self.vars["stage4_noise_floor_max_db"].get()),
            "max_lambda": float(self.vars["stage4_max_lambda"].get()),
            "use_optimized_origins": self.vars["stage4_use_optimized_origins"].get(),
        }

    def _parse_float_bounds(self, text):
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected two comma-separated values: {text}")
        return (float(parts[0]), float(parts[1]))

    def _parse_int_bounds(self, text):
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected two comma-separated values: {text}")
        return (int(parts[0]), int(parts[1]))

    def _drain_cli_queue(self):
        try:
            while True:
                text = self.cli_queue.get_nowait()
                self.log_text.insert(tk.END, text)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(40, self._drain_cli_queue)

    def _on_close(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.destroy()


if __name__ == "__main__":
    app = PositionSensitivityApp()
    app.mainloop()
