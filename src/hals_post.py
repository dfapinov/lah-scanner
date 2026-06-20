#!/usr/bin/env python3
"""Spawn-safe entry point for the HALS post-processing GUI."""


def main():
    from hals_post_ui_core import main as run_gui

    run_gui()


if __name__ == "__main__":
    main()
