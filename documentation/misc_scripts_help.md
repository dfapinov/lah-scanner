# Miscellaneous Utilities Overview

This document outlines the three supplementary Python scripts located in the `src/misc/` directory. These tools provide supplementary capabilities for the main pipeline, including acoustic simulation, grid stability analysis, and educational visualization.

---

## 1. Synthetic Impulse Response Generator (`synth_ir_gen.py`)

**Purpose:**  
This script generates synthetic impulse responses (IRs) for any given measurement grid CSV. It simulates a circular piston source (like a loudspeaker driver) inside the grid by modeling it as a dense cluster of point sources (monopoles). This allows for testing the Spherical Harmonic Expansion (SHE) pipeline with mathematically perfect, noise-free input data.

**Key Features:**
* Simulates frequency-dependent directivity (acts as an omnidirectional monopole at low frequencies and becomes highly directional at high frequencies).
* Directly outputs `.wav` files and a modified CSV log with distance and delay metadata.

**Controls:**
* **Source Configuration:** Shift the 3D origin (X, Y, Z), change the piston radius, and set its directional facing axis.
* **Audio Settings:** Define the exact sample rate, target output duration, and volume gain.
* **File I/O:** Point the script to any measurement grid CSV and let it batch-generate the corresponding WAV files.

---

## 2. Grid Assessment Tool (`grid_condition_util.py`)

**Purpose:**  
A tool to investigate the properties of different measurement grid geometries. The primary means to do this is via visual inspection of the grid's ability for sound field separation. The script simulates an omnidirectional monopole at the centre of the grid and solves the SHE across a set frequency spectrum and target expansion Order (N).

**Key Features:**
* Evaluates **Field Separation Correlation**, calculating how well the grid can distinguish between sound radiating outward (the speaker) and sound radiating inward (zero/none in this simulated case, making mode leakage from issues like Bessel nulls and spatial aliasing highly visible).
* Evaluates the **Condition Number**, which indicates how prone the linear algebra matrix is to amplifying noise and failing (ill-conditioning).
* Evaluates **Spatial Aliasing**, measuring Gram matrix orthogonality to highlight spatial aliasing across the grid.
* Evaluates **White Noise Amplification (WNA)**, quantifying how much the measurement grid amplifies uncorrelated sensor noise.
* Evaluates **Maximum Coherence**, identifying the worst-case cross-talk between spherical harmonic modes.
* Evaluates **Effective Degrees of Freedom (EDOF)**, estimating the actual independent spatial dimensions the grid successfully captures.

**Controls:**
* **Target Order Selection:** Set the maximum expansion Order (N) to test for your specific grid design.
* **Frequency Range:** Define the start, stop, and step frequencies for the stability sweep. Usually the higher frequencies are those of interest with grid geometry. The lower frequencies are dominated by simple grid radius.



---

## 3. Complex Frequency Visualizer (`complex_visualizer.py`)

**Purpose:**  
An interactive, educational GUI tool designed to help users visualize the mathematics underlying complex frequency domain processing. It demonstrates how Real (Cosine) and Imaginary (Sine) coefficients define phase and magnitude, and how they project into a time-domain waveform.

**Key Features:**
* Real-time interactive sliders to adjust the frequency, real component, and imaginary component.
* Dual-plot display showing the Complex Plane (rotating Phasor) and the resulting Time-Domain Waveform.
* A pop-out instructional guide window formatted with rich text that explains the concepts of the complex plane, coordinate systems, and DSP signal generation.

**Controls:**
* **Parameter Sliders:** Adjust Real (Cosine) and Imaginary (Sine) values in real-time to manipulate magnitude and phase.
* **Frequency Control:** Change the frequency slider to see how the phasor speed affects the time-domain wave.
* **Interactive Guide:** Open a built-in educational Markdown document explaining the concepts visually represented on screen.