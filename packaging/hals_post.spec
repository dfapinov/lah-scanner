# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build specification for the HALS Post-Processing GUI.

Build (from the repository root):

    pyinstaller packaging/hals_post.spec

Produces a self-contained one-directory application under ``dist/HALS_Post``.
A one-directory build is used (rather than one-file) because the application
relies heavily on ``multiprocessing`` with the ``spawn`` start method for its
processing stages; one-directory avoids repeated temp extraction and keeps the
spawned workers fast and reliable.
"""

import os

from PyInstaller.utils.hooks import collect_all

# ``SPECPATH`` is injected by PyInstaller and points at this file's directory.
PROJECT_ROOT = os.path.dirname(SPECPATH)
SRC = os.path.join(PROJECT_ROOT, "src")
PROCESS = os.path.join(SRC, "process")
MISC = os.path.join(SRC, "misc")
IMAGES = os.path.join(PROJECT_ROOT, "images")

# The GUI resolves its resources relative to the module directory, which maps to
# the bundle root once frozen, so ship them at the top level of the bundle.
datas = [
    (os.path.join(SRC, "splash.png"), "."),
    (os.path.join(SRC, "HALS_icon.ico"), "."),
    (os.path.join(SRC, "HALS_icon.png"), "."),
    # The RFT calculator draws a speaker glyph loaded from this SVG. Ship it in
    # an "images" subfolder so it resolves via sys._MEIPASS when frozen.
    (os.path.join(IMAGES, "speaker.svg"), "images"),
]
binaries = []

# The source tree uses a flat, sys.path-based layout and imports sibling modules
# by their bare top-level names (e.g. ``from stage2_centre_origin import ...``).
# List them explicitly as hidden imports so PyInstaller bundles every stage,
# core and helper module regardless of how it is imported at runtime.
hiddenimports = [
    # src/process
    "complex_to_ir_core",
    "config_process",
    "extract_pressures_core",
    "fdw_smoothing_core",
    "schema",
    "she_solver_core",
    "stage1_fdwsmooth",
    "stage2_centre_origin",
    "stage3_optimize_she_settings",
    "stage4_run_she_solve",
    "stage5_extract_pressures",
    "stage5_pressure_utils",
    "utils",
    "viewers",
    # src/misc
    "complex_visualizer",
    "grid_condition_util",
    "perturb_npz_coordinates",
    "position_sensitivity_gui",
    "rft_calculator",
    "spatial_error_viewer",
    "synth_ir_gen",
]

# soundfile ships the native libsndfile library outside the Python package;
# collect_all makes sure the shared library travels with the build.
for pkg in ("soundfile",):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

block_cipher = None

a = Analysis(
    [os.path.join(SRC, "hals_post.py")],
    pathex=[SRC, PROCESS, MISC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HALS_Post",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SRC, "HALS_icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HALS_Post",
)
