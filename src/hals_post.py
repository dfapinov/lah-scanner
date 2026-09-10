#!/usr/bin/env python3
"""Spawn-safe entry point for the HALS post-processing GUI."""

import multiprocessing


def main():
    from hals_post_ui_core import main as run_gui

    run_gui()


if __name__ == "__main__":
    # Required so 'spawn' multiprocessing workers behave correctly when the
    # application is frozen into a standalone executable by PyInstaller.
    # Without this, each spawned worker would relaunch the full GUI.
    multiprocessing.freeze_support()
    main()
