#!/usr/bin/env python3
"""
Duplicate a HALS complex-data NPZ with perturbed coordinate filenames.

This script changes only the coordinate metadata encoded in the keys of the NPZ
complex-data dictionary. The complex pressure data itself is copied unchanged.

Example:
    python src/misc/perturb_npz_coordinates.py ^
        project/outputs/MySpeaker_complex_data.npz ^
        project/outputs/MySpeaker_complex_data_Rplus1mm.npz ^
        --r-offset-mm 1
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROCESS_DIR = SCRIPT_DIR.parent / "process"
sys.path.insert(0, str(PROCESS_DIR))

import schema  # noqa: E402
from utils import parse_coords_from_filename, spherical_to_cartesian  # noqa: E402


COORD_RE = re.compile(
    r"^(?P<prefix>.*?)_r(?P<rmm>-?\d+(?:p\d+)*)_ph(?P<ph>-?\d+(?:p\d+)*)_z(?P<zmm>-?\d+(?:p\d+)*)(?P<suffix>(?:_.*)?(?:\.[^.]+)?)$",
    re.IGNORECASE,
)


def encode_coord_number(value: float, decimals: int = 3) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text.replace(".", "p")


def format_coord_filename(original_name: str, r_cyl_mm: float, phi_deg: float, z_mm: float) -> str:
    match = COORD_RE.match(original_name)
    if not match:
        raise ValueError(f"Filename does not match coordinate pattern: {original_name}")
    return (
        f"{match.group('prefix')}"
        f"_r{encode_coord_number(r_cyl_mm)}"
        f"_ph{encode_coord_number(phi_deg)}"
        f"_z{encode_coord_number(z_mm)}"
        f"{match.group('suffix')}"
    )


def dedupe_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    base, ext = os.path.splitext(name)
    idx = 1
    while True:
        candidate = f"{base}_pert{idx}{ext}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def perturb_npz_coordinates(
    input_npz: Path,
    output_npz: Path,
    r_offset_mm: float,
    phi_offset_deg: float,
    z_offset_mm: float,
    r_jitter_mm: float,
    phi_jitter_deg: float,
    z_jitter_mm: float,
    seed: int,
    keep_origins: bool,
) -> dict:
    rng = np.random.default_rng(seed)
    loaded = np.load(input_npz, allow_pickle=True)
    try:
        if schema.COMPLEX_DATA not in loaded:
            raise KeyError(f"Input NPZ does not contain '{schema.COMPLEX_DATA}'.")

        data_dict = loaded[schema.COMPLEX_DATA].item()
        new_data = {}
        used_names: set[str] = set()

        max_abs_r_delta = 0.0
        max_abs_phi_delta = 0.0
        max_abs_z_delta = 0.0
        changed_count = 0

        for old_name, values in data_dict.items():
            r_sph, theta, phi = parse_coords_from_filename(old_name)
            x, y, z = spherical_to_cartesian(r_sph, theta, phi)

            old_r_mm = math.hypot(x, y) * 1000.0
            old_phi_deg = math.degrees(math.atan2(y, x))
            old_z_mm = z * 1000.0

            r_delta = r_offset_mm + (rng.uniform(-r_jitter_mm, r_jitter_mm) if r_jitter_mm else 0.0)
            phi_delta = phi_offset_deg + (rng.uniform(-phi_jitter_deg, phi_jitter_deg) if phi_jitter_deg else 0.0)
            z_delta = z_offset_mm + (rng.uniform(-z_jitter_mm, z_jitter_mm) if z_jitter_mm else 0.0)

            new_r_mm = max(0.001, old_r_mm + r_delta)
            new_phi_deg = old_phi_deg + phi_delta
            new_z_mm = old_z_mm + z_delta

            if r_delta == 0.0 and phi_delta == 0.0 and z_delta == 0.0:
                new_name = dedupe_name(old_name, used_names)
            else:
                new_name = dedupe_name(format_coord_filename(old_name, new_r_mm, new_phi_deg, new_z_mm), used_names)
            parse_coords_from_filename(new_name)
            new_data[new_name] = values

            if new_name != old_name:
                changed_count += 1
            max_abs_r_delta = max(max_abs_r_delta, abs(new_r_mm - old_r_mm))
            max_abs_phi_delta = max(max_abs_phi_delta, abs(phi_delta))
            max_abs_z_delta = max(max_abs_z_delta, abs(new_z_mm - old_z_mm))

        save_dict = {}
        for key in loaded.files:
            if key == schema.COMPLEX_DATA:
                continue
            if key == schema.ORIGINS_MM and not keep_origins:
                continue
            save_dict[key] = loaded[key]
        save_dict[schema.COMPLEX_DATA] = new_data

        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_npz, **save_dict)

        return {
            "points": len(data_dict),
            "changed_names": changed_count,
            "dropped_origins": (schema.ORIGINS_MM in loaded.files and not keep_origins),
            "max_abs_r_delta_mm": max_abs_r_delta,
            "max_abs_phi_delta_deg": max_abs_phi_delta,
            "max_abs_z_delta_mm": max_abs_z_delta,
        }
    finally:
        if hasattr(loaded, "close"):
            loaded.close()


class PerturbNpzGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perturb NPZ Coordinates")
        self.geometry("680x430")
        self.minsize(620, 390)

        self.vars = {
            "input_npz": tk.StringVar(value=""),
            "output_npz": tk.StringVar(value=""),
            "r_offset_mm": tk.StringVar(value="0.0"),
            "phi_offset_deg": tk.StringVar(value="0.0"),
            "z_offset_mm": tk.StringVar(value="0.0"),
            "r_jitter_mm": tk.StringVar(value="0.0"),
            "phi_jitter_deg": tk.StringVar(value="0.0"),
            "z_jitter_mm": tk.StringVar(value="0.0"),
            "seed": tk.StringVar(value="12345"),
            "keep_origins": tk.BooleanVar(value=False),
            "status": tk.StringVar(value="Ready."),
        }

        self._build_ui()

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)

        files = ttk.LabelFrame(root, text="Files", padding=10)
        files.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Input NPZ:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(files, textvariable=self.vars["input_npz"]).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(files, text="Browse", command=self._browse_input).grid(row=0, column=2, padx=(6, 0), pady=3)

        ttk.Label(files, text="Output NPZ:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(files, textvariable=self.vars["output_npz"]).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(files, text="Browse", command=self._browse_output).grid(row=1, column=2, padx=(6, 0), pady=3)

        offsets = ttk.LabelFrame(root, text="Fixed Cylindrical Offsets", padding=10)
        offsets.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for col in range(3):
            offsets.columnconfigure(col, weight=1)
        self._add_value(offsets, 0, 0, "R offset (mm)", "r_offset_mm")
        self._add_value(offsets, 0, 1, "Phi offset (deg)", "phi_offset_deg")
        self._add_value(offsets, 0, 2, "Z offset (mm)", "z_offset_mm")

        jitter = ttk.LabelFrame(root, text="Random Jitter Per Point", padding=10)
        jitter.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        for col in range(4):
            jitter.columnconfigure(col, weight=1)
        self._add_value(jitter, 0, 0, "+/- R jitter (mm)", "r_jitter_mm")
        self._add_value(jitter, 0, 1, "+/- Phi jitter (deg)", "phi_jitter_deg")
        self._add_value(jitter, 0, 2, "+/- Z jitter (mm)", "z_jitter_mm")
        self._add_value(jitter, 0, 3, "Seed", "seed")

        options = ttk.Frame(root)
        options.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ttk.Checkbutton(
            options,
            text="Keep existing origins_mm in output",
            variable=self.vars["keep_origins"],
        ).pack(side=tk.LEFT)
        ttk.Label(
            options,
            text="Default is to remove origins so Stage 2 starts clean.",
            font=("Arial", 8, "italic"),
        ).pack(side=tk.LEFT, padx=(12, 0))

        actions = ttk.Frame(root)
        actions.grid(row=4, column=0, sticky="ew")
        ttk.Button(actions, text="Create Perturbed NPZ", command=self._run).pack(side=tk.RIGHT)
        ttk.Button(actions, text="Quit", command=self.destroy).pack(side=tk.RIGHT, padx=(0, 8))

        status = ttk.Label(root, textvariable=self.vars["status"], relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status.grid(row=5, column=0, sticky="ew", pady=(12, 0))

    def _add_value(self, parent, row, col, label, key):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, sticky="ew", padx=4)
        ttk.Label(frame, text=label).pack(anchor=tk.W)
        ttk.Entry(frame, textvariable=self.vars[key], width=14).pack(fill=tk.X)

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input NPZ",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return
        self.vars["input_npz"].set(path)
        if not self.vars["output_npz"].get().strip():
            input_path = Path(path)
            self.vars["output_npz"].set(str(input_path.with_name(f"{input_path.stem}_perturbed{input_path.suffix}")))

    def _browse_output(self):
        initial = self.vars["output_npz"].get().strip()
        initial_dir = str(Path(initial).parent) if initial else ""
        path = filedialog.asksaveasfilename(
            title="Save output NPZ",
            initialdir=initial_dir or None,
            defaultextension=".npz",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
        )
        if path:
            self.vars["output_npz"].set(path)

    def _float(self, key):
        return float(self.vars[key].get().strip())

    def _run(self):
        try:
            input_npz = Path(self.vars["input_npz"].get().strip())
            output_npz = Path(self.vars["output_npz"].get().strip())
            if not input_npz.exists():
                raise FileNotFoundError(f"Input NPZ not found: {input_npz}")
            if not output_npz:
                raise ValueError("Choose an output NPZ path.")

            summary = perturb_npz_coordinates(
                input_npz=input_npz,
                output_npz=output_npz,
                r_offset_mm=self._float("r_offset_mm"),
                phi_offset_deg=self._float("phi_offset_deg"),
                z_offset_mm=self._float("z_offset_mm"),
                r_jitter_mm=self._float("r_jitter_mm"),
                phi_jitter_deg=self._float("phi_jitter_deg"),
                z_jitter_mm=self._float("z_jitter_mm"),
                seed=int(self.vars["seed"].get().strip()),
                keep_origins=self.vars["keep_origins"].get(),
            )
            msg = (
                f"Wrote {output_npz}\n\n"
                f"Points copied: {summary['points']}\n"
                f"Coordinate keys changed: {summary['changed_names']}\n"
                f"Dropped existing origins_mm: {summary['dropped_origins']}\n"
                "Max absolute perturbation:\n"
                f"R={summary['max_abs_r_delta_mm']:.6f} mm, "
                f"Phi={summary['max_abs_phi_delta_deg']:.6f} deg, "
                f"Z={summary['max_abs_z_delta_mm']:.6f} mm"
            )
            self.vars["status"].set(f"Wrote {output_npz}")
            messagebox.showinfo("Perturb NPZ Coordinates", msg)
        except Exception as exc:
            self.vars["status"].set(f"Error: {exc}")
            messagebox.showerror("Perturb NPZ Coordinates", str(exc))


def run_gui():
    app = PerturbNpzGui()
    app.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a HALS complex-data NPZ while perturbing reported cylindrical mic coordinates."
    )
    parser.add_argument("input_npz", type=Path, nargs="?", help="Input Stage 1 complex_data NPZ.")
    parser.add_argument("output_npz", type=Path, nargs="?", help="Output NPZ to create.")
    parser.add_argument("--r-offset-mm", type=float, default=0.0, help="Fixed cylindrical R offset in mm.")
    parser.add_argument("--phi-offset-deg", type=float, default=0.0, help="Fixed cylindrical Phi offset in degrees.")
    parser.add_argument("--z-offset-mm", type=float, default=0.0, help="Fixed cylindrical Z offset in mm.")
    parser.add_argument("--r-jitter-mm", type=float, default=0.0, help="Uniform random +/- R jitter in mm per point.")
    parser.add_argument("--phi-jitter-deg", type=float, default=0.0, help="Uniform random +/- Phi jitter in degrees per point.")
    parser.add_argument("--z-jitter-mm", type=float, default=0.0, help="Uniform random +/- Z jitter in mm per point.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for jitter.")
    parser.add_argument(
        "--keep-origins",
        action="store_true",
        help="Keep existing origins_mm if present. By default origins are removed so Stage 2 starts clean.",
    )
    parser.add_argument("--gui", action="store_true", help="Launch the Tkinter GUI.")
    args = parser.parse_args()

    if args.gui:
        run_gui()
        return 0

    if args.input_npz is None or args.output_npz is None:
        parser.error("input_npz and output_npz are required unless --gui is used.")

    summary = perturb_npz_coordinates(
        input_npz=args.input_npz,
        output_npz=args.output_npz,
        r_offset_mm=args.r_offset_mm,
        phi_offset_deg=args.phi_offset_deg,
        z_offset_mm=args.z_offset_mm,
        r_jitter_mm=args.r_jitter_mm,
        phi_jitter_deg=args.phi_jitter_deg,
        z_jitter_mm=args.z_jitter_mm,
        seed=args.seed,
        keep_origins=args.keep_origins,
    )

    print(f"Wrote perturbed NPZ: {args.output_npz}")
    print(f"Points copied: {summary['points']}")
    print(f"Coordinate keys changed: {summary['changed_names']}")
    print(f"Dropped existing origins_mm: {summary['dropped_origins']}")
    print(
        "Max absolute perturbation: "
        f"R={summary['max_abs_r_delta_mm']:.6f} mm, "
        f"Phi={summary['max_abs_phi_delta_deg']:.6f} deg, "
        f"Z={summary['max_abs_z_delta_mm']:.6f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
