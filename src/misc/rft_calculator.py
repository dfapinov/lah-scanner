#!/usr/bin/env python3
"""
Reflection-free time calculator for HALS Stage 1.

The calculator estimates the first floor-reflection delay using the image-source
method, then converts that reflection-free time into FDW transition frequencies
for common octave resolutions.
"""

import math
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk


COMMON_OCTAVE_RESOLUTIONS = (3, 6, 12, 24)
SPEED_OF_SOUND_MPS = 343.0


def calculate_cycles_from_oct_res(oct_res):
    if oct_res <= 0:
        return 1.0
    return 1.0 / (2 ** (1.0 / oct_res) - 1.0)


def calculate_rft(horizontal_mm, source_height_mm, mic_height_mm, speed_of_sound=SPEED_OF_SOUND_MPS):
    horizontal_m = max(float(horizontal_mm), 0.0) / 1000.0
    source_h_m = max(float(source_height_mm), 0.0) / 1000.0
    mic_h_m = max(float(mic_height_mm), 0.0) / 1000.0
    c = max(float(speed_of_sound), 1e-9)

    direct_m = math.hypot(horizontal_m, source_h_m - mic_h_m)
    reflected_m = math.hypot(horizontal_m, source_h_m + mic_h_m)
    excess_m = max(0.0, reflected_m - direct_m)
    rft_ms = (excess_m / c) * 1000.0

    return {
        "direct_mm": direct_m * 1000.0,
        "reflection_mm": reflected_m * 1000.0,
        "excess_mm": excess_m * 1000.0,
        "rft_ms": rft_ms,
    }


def transition_frequency_hz(rft_ms, oct_res):
    rft_s = max(float(rft_ms), 1e-12) / 1000.0
    return calculate_cycles_from_oct_res(oct_res) / rft_s


class RFTCalculatorWindow:
    def __init__(self, parent=None, initial_rft_ms=None, on_apply=None):
        self.parent = parent
        self.on_apply = on_apply
        self.root = tk.Toplevel(parent) if parent is not None else tk.Tk()
        self.root.title("Reflection Free Time Calculator")
        self.root.geometry("760x520")
        self.root.minsize(680, 480)
        self.speaker_paths = None

        self.horizontal_mm = tk.StringVar(value="100")
        self.source_height_mm = tk.StringVar(value="1100")
        self.mic_height_mm = tk.StringVar(value="1100")
        self.rft_result_ms = tk.StringVar(value="")

        if initial_rft_ms not in (None, ""):
            self.rft_result_ms.set(f"{float(initial_rft_ms):.2f}")

        self._build_ui()
        self._wire_updates()
        self._recalculate()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main)
        top.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(top, width=420, height=260, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        self.canvas.bind("<Configure>", lambda _event: self._recalculate())

        side = ttk.Frame(top)
        side.pack(side=tk.LEFT, fill=tk.Y)

        self._add_entry(side, "Speaker to Mic:", self.horizontal_mm, "mm")
        self._add_entry(side, "Speaker to Boundary:", self.source_height_mm, "mm")
        self._add_entry(side, "Mic to Boundary:", self.mic_height_mm, "mm")

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self._add_readout(side, "RFT", self.rft_result_ms, "ms")

        table_frame = ttk.LabelFrame(main, text="Reflection-Free Transition Frequency", padding=10)
        table_frame.pack(fill=tk.X, pady=(12, 0))

        self.tree = ttk.Treeview(table_frame, columns=("res", "cycles", "freq"), show="headings", height=4)
        self.tree.heading("res", text="Resolution")
        self.tree.heading("cycles", text="Cycles")
        self.tree.heading("freq", text="Transition")
        self.tree.column("res", width=140, anchor=tk.CENTER)
        self.tree.column("cycles", width=120, anchor=tk.CENTER)
        self.tree.column("freq", width=160, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X)

        buttons = ttk.Frame(main)
        buttons.pack(fill=tk.X, pady=(12, 0))

        if self.on_apply is not None:
            ttk.Button(buttons, text="Use This RFT", command=self._apply_value).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Close", command=self._close).pack(side=tk.RIGHT)

    def _add_entry(self, parent, label, var, unit):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT)
        ttk.Label(row, text=unit).pack(side=tk.LEFT, padx=(5, 0))

    def _add_readout(self, parent, label, var, unit):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=var, width=10, relief=tk.SUNKEN, anchor=tk.E).pack(side=tk.LEFT)
        ttk.Label(row, text=unit).pack(side=tk.LEFT, padx=(5, 0))

    def _wire_updates(self):
        for var in (self.horizontal_mm, self.source_height_mm, self.mic_height_mm):
            var.trace_add("write", lambda *_: self._recalculate())

    def _float_var(self, var, fallback=0.0):
        try:
            return float(var.get())
        except ValueError:
            return fallback

    def _recalculate(self):
        result = calculate_rft(
            self._float_var(self.horizontal_mm),
            self._float_var(self.source_height_mm),
            self._float_var(self.mic_height_mm),
        )

        self.rft_result_ms.set(f"{result['rft_ms']:.2f}")

        self._draw_diagram(result)
        self._update_table(result["rft_ms"])

    def _draw_diagram(self, result):
        c = self.canvas
        c.delete("all")

        width = max(c.winfo_width(), 420)
        height = max(c.winfo_height(), 260)
        margin_x = 64
        floor_y = height - 42
        speaker_x = margin_x
        mic_x = width - margin_x

        source_h = max(self._float_var(self.source_height_mm), 1.0)
        mic_h = max(self._float_var(self.mic_height_mm), 1.0)
        max_h = max(source_h, mic_h, 1.0)
        usable_h = height - 88
        source_y = floor_y - (source_h / max_h) * usable_h
        mic_y = floor_y - (mic_h / max_h) * usable_h

        c.create_line(16, floor_y, width - 16, floor_y, fill="#444444", width=3)

        speaker_center_x = speaker_x - 22
        bounce_x = (speaker_x + mic_x) / 2
        source_point_x = speaker_center_x + 48
        mic_point_x = mic_x - 10
        c.create_line(source_point_x, source_y, mic_point_x, mic_y, fill="#168a31", width=2)
        c.create_line(source_point_x, source_y, bounce_x, floor_y, fill="#cc2222", width=2)
        c.create_line(bounce_x, floor_y, mic_x - 10, mic_y, fill="#cc2222", width=2)

        speaker_dim_x = source_point_x - 24
        mic_dim_x = mic_point_x + 20
        self._draw_dimension_line(c, speaker_dim_x, source_y, floor_y, f"{self._float_var(self.source_height_mm):.0f} mm")
        self._draw_dimension_line(c, mic_dim_x, mic_y, floor_y, f"{self._float_var(self.mic_height_mm):.0f} mm")

        rft_label_y = max(20, min(source_y, mic_y) - 26)
        distance_label_y = max(34, min(source_y, mic_y) - 10)
        self._create_text_with_bg(c, (source_point_x + mic_point_x) / 2, distance_label_y, f"{self._float_var(self.horizontal_mm):.0f} mm")
        self._create_text_with_bg(c, (source_point_x + mic_point_x) / 2, rft_label_y, f"RFT: {result['rft_ms']:.2f} ms")

        c.create_rectangle(mic_x - 6, mic_y - 8, mic_x + 48, mic_y + 8, fill="#777777", outline="#333333")
        c.create_rectangle(mic_x - 10, mic_y - 12, mic_x - 2, mic_y + 12, fill="#555555", outline="#333333")
        self._draw_speaker(c, speaker_center_x, source_y)

    def _draw_dimension_line(self, canvas, x, y_top, y_bottom, label):
        label_y = (y_top + y_bottom) / 2
        gap = 18
        canvas.create_line(x, y_top, x, label_y - gap, fill="#1f5fbf")
        canvas.create_line(x, label_y + gap, x, y_bottom, fill="#1f5fbf", arrow=tk.LAST)
        self._create_text_with_bg(canvas, x, label_y, label)

    def _create_text_with_bg(self, canvas, x, y, text, anchor=tk.CENTER):
        text_id = canvas.create_text(x, y, text=text, fill="#111111", anchor=anchor)
        bbox = canvas.bbox(text_id)
        if bbox:
            pad_x, pad_y = 3, 1
            rect_id = canvas.create_rectangle(
                bbox[0] - pad_x,
                bbox[1] - pad_y,
                bbox[2] + pad_x,
                bbox[3] + pad_y,
                fill="white",
                outline="",
            )
            canvas.tag_lower(rect_id, text_id)
        return text_id

    def _draw_speaker(self, canvas, x, y):
        if self.speaker_paths is None:
            svg_path = Path(__file__).resolve().parents[2] / "images" / "speaker.svg"
            self.speaker_paths = self._load_svg_paths(svg_path)

        scale = 0.16
        for path_points in self.speaker_paths:
            points = []
            for px, py in path_points:
                points.extend((x + (px - 244.8) * scale, y + (py - 244.8) * scale))
            if len(points) >= 6:
                canvas.create_polygon(points, fill="#202326", outline="#202326")

    def _load_svg_paths(self, svg_path):
        text = svg_path.read_text(encoding="iso-8859-1")
        paths = re.findall(r'<path\s+d="([^"]+)"', text, flags=re.IGNORECASE | re.DOTALL)
        return [self._sample_svg_path(path_d) for path_d in paths]

    def _sample_svg_path(self, path_d):
        tokens = re.findall(r"[A-Za-z]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", path_d)
        points = []
        i = 0
        cmd = None
        x = y = 0.0
        start_x = start_y = 0.0
        last_cx = last_cy = None

        def is_cmd(index):
            return index < len(tokens) and re.match(r"^[A-Za-z]$", tokens[index])

        def num():
            nonlocal i
            value = float(tokens[i])
            i += 1
            return value

        def add_point(px, py):
            points.append((px, py))

        def cubic(p0, p1, p2, p3, steps=12):
            for step in range(1, steps + 1):
                t = step / steps
                mt = 1.0 - t
                px = (mt ** 3 * p0[0]) + (3 * mt ** 2 * t * p1[0]) + (3 * mt * t ** 2 * p2[0]) + (t ** 3 * p3[0])
                py = (mt ** 3 * p0[1]) + (3 * mt ** 2 * t * p1[1]) + (3 * mt * t ** 2 * p2[1]) + (t ** 3 * p3[1])
                add_point(px, py)

        while i < len(tokens):
            if is_cmd(i):
                cmd = tokens[i]
                i += 1

            if cmd in ("M", "m"):
                rel = cmd == "m"
                x_new, y_new = num(), num()
                x = x + x_new if rel else x_new
                y = y + y_new if rel else y_new
                start_x, start_y = x, y
                add_point(x, y)
                cmd = "l" if rel else "L"
                last_cx = last_cy = None
            elif cmd in ("L", "l"):
                rel = cmd == "l"
                x_new, y_new = num(), num()
                x = x + x_new if rel else x_new
                y = y + y_new if rel else y_new
                add_point(x, y)
                last_cx = last_cy = None
            elif cmd in ("H", "h"):
                value = num()
                x = x + value if cmd == "h" else value
                add_point(x, y)
                last_cx = last_cy = None
            elif cmd in ("V", "v"):
                value = num()
                y = y + value if cmd == "v" else value
                add_point(x, y)
                last_cx = last_cy = None
            elif cmd in ("C", "c"):
                rel = cmd == "c"
                x1, y1, x2, y2, x3, y3 = num(), num(), num(), num(), num(), num()
                p1 = (x + x1, y + y1) if rel else (x1, y1)
                p2 = (x + x2, y + y2) if rel else (x2, y2)
                p3 = (x + x3, y + y3) if rel else (x3, y3)
                cubic((x, y), p1, p2, p3)
                x, y = p3
                last_cx, last_cy = p2
            elif cmd in ("S", "s"):
                rel = cmd == "s"
                if last_cx is None:
                    p1 = (x, y)
                else:
                    p1 = ((2 * x) - last_cx, (2 * y) - last_cy)
                x2, y2, x3, y3 = num(), num(), num(), num()
                p2 = (x + x2, y + y2) if rel else (x2, y2)
                p3 = (x + x3, y + y3) if rel else (x3, y3)
                cubic((x, y), p1, p2, p3)
                x, y = p3
                last_cx, last_cy = p2
            elif cmd in ("Z", "z"):
                x, y = start_x, start_y
                add_point(x, y)
                last_cx = last_cy = None
            else:
                break

        return points

    def _update_table(self, rft_ms):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for oct_res in COMMON_OCTAVE_RESOLUTIONS:
            cycles = calculate_cycles_from_oct_res(oct_res)
            freq_hz = transition_frequency_hz(rft_ms, oct_res)
            self.tree.insert("", tk.END, values=(f"1/{oct_res} octave", f"{cycles:.1f}", self._format_frequency(freq_hz)))

    @staticmethod
    def _format_frequency(freq_hz):
        if freq_hz >= 1000:
            return f"{freq_hz / 1000.0:.2f} kHz"
        return f"{freq_hz:.0f} Hz"

    def _apply_value(self):
        if self.on_apply is not None:
            self.on_apply(self.rft_result_ms.get())
        self.root.destroy()

    def _close(self):
        self.root.destroy()


def main():
    app = RFTCalculatorWindow()
    app.root.mainloop()


if __name__ == "__main__":
    main()
