# Stage 3 cumulative internal tail power

The lower plot estimates internal radiated power discarded by stopping at N.
All coefficients come from one reference fit of order M:

`tail(N,f) = sum_(n=N+1..M,m) |C_nm(f)|^2 / sum_(n=0..M,m) |C_nm(f)|^2`

The automatic reference is the highest tested order with mean Int/Ext strictly
above 20 dB. If none qualifies, use the best-ratio order with the existing warning.
The title identifies the reference order, ratio and selection method.
The Tail-power reference dropdown selects any tested order without another solve;
entries flag ratios not above 20 dB. Changing it updates the curve and marker.
The saved PNG uses the automatic reference. Manual selections are not persisted.
The Stage 4 recommendation remains based on the existing separation/knee rule.

Average per-frequency tail fractions in linear units, then convert to dB.
-20 dB means 1% discarded; -25 dB means about 0.316%. The tail at N=M is zero
by construction and omitted from the curve. Above M is unknown, not zero.
Other tiny/zero tails have a display floor of -80 dB. Power beyond M is unknown.
Only outgoing internal C coefficients contribute; common physical power factors
cancel at each frequency. This is modeled radiated power, not reconstruction error
or the difference between separately refitted models. It can include fitted noise.

Reference fits truncated below M, with zero internal power, or nonfinite
coefficients are excluded consistently across the curve. No valid samples means
no curve. The console reports valid counts. Returned step1 fields include
internal_tail_power_db, tail_reference and tail_by_reference (all curves/counts).
Legacy individual-degree data remains available in the returned results.

Octave Resolution controls both diagnostics: 12 = 1/12 octave, 24 = 1/24 octave,
0 = all positive bins in range. Targets map to nearest available bins and are
deduplicated, including the available endpoints. Finer spacing takes longer.
This setting saves with the project. step1.sample_frequencies_hz records the
sampled frequencies. Octave sampling weights roughly equally per octave;
all-bin sampling weights bins equally. Averaging can dilute narrow-band features.

The results dialog highlights three selectable choices on both graphs:
roll-off knee, highest order strictly above 20 dB Int/Ext, and first tested order
with cumulative tail strictly below -20 dB. The tail choice excludes the reference
itself and updates when the reference dropdown changes. Missing choices are disabled;
when needed the existing best-ratio fallback remains selectable. Use in Stage 4
sends the selected radio-button choice. The ? help button explains windowing,
separation, cumulative power, the tradeoff, and the sensitivity of knee detection.
