# Stage 3 cumulative internal tail power

The top plot estimates internal radiated power discarded by stopping at N.
All coefficients come from one reference fit of order M:

`tail(N,f) = sum_(n=N+1..M,m) |C_nm(f)|^2 / sum_(n=0..M,m) |C_nm(f)|^2`

The automatic reference is the highest tested order with mean Int/Ext strictly
above 20 dB. If none qualifies, switch to direct percentile-SPL order selection; no tail reference is needed.
The title identifies the reference order, ratio and selection method.
The Tail-power reference dropdown selects any tested order without another solve;
entries flag ratios not above 20 dB. Changing it updates the curve and marker.
The saved PNG uses the automatic reference. Manual selections are not persisted.
The bottom plot checks separation suitability; it is not a direct stability measurement.

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

## Selection modes

When the band permits Int/Ext assessment and any order exceeds 20 dB, recommend
the eligible ratio knee. Soft -25 dB tail power (first tail <= -24.5 dB before the
reference) is an alternative or fallback if no eligible knee exists. Tail power
primarily informs how much modeled internal information is discarded. It cannot
measure missing power beyond the reference, whose zero tail is omitted.

Below RFT or when every order is <=20 dB, choose the actual maximum solve order
directly from the 99th-percentile SPL early plateau. Do not derive a tail reference
or apply the soft tail threshold. The maximum curve is diagnostic only. Offer the
order two below the detected plateau as recommended, and one below the plateau
as the higher-order alternative. Both labels identify the detected plateau N.
The popup defaults to the SPL tab and retains Int/Ext as inspection only, without
a tail-reference dropdown. The saved SPL graph highlights both options. Order
transfer uses the selected radio button. Only below-RFT bands reset their lower
limit to upper/4; an above-RFT band with poor separation is unchanged.

The first qualifying four-order plateau has range <=3 dB, end no more than 1 dB
below start, and median <= the supported global minimum +3 dB. Its preceding
three supported orders must have median at least max(0.5 dB, half the plateau
median in dB) higher. The second order of the window is the detected plateau N; recommend N-2 and
offer N-1 as the alternative. All seven increments
must be valid and actually add a degree. This pragmatic curve-shape heuristic is
not a physical accuracy guarantee. If the plateau is unresolved, no automatic choice
is supplied. No reference or maximum-curve fallback silently replaces it.

## Incremental SPL change (separate diagnostic)

The results dialog includes an **Incremental SPL change** tab. The original tail
power/separation graphs, reference dropdown, choices and recommendation remain.
Stage 3 always calculates incremental SPL with a fixed -40 dB assessment floor.
Main Settings contains Upper Test Range. Advanced Settings contains Lower Test
Range, octave resolution, test order range, sphere
points (default 1,000) and radius (1 m). Both range fields have circular ? help
icons with hover notes and click/keyboard activation. The command-line geometry settings are
TEST_SPL_SPHERE_POINTS and TEST_SPL_RADIUS_M. Legacy enable/floor API arguments
are accepted but ignored, and obsolete project settings are removed on save.

For every added order N, reconstruct the internal fields from the independent
N-1 and N fits on the same approximately uniform Fibonacci sphere, centred at
the measurement coordinate origin. Translate that fixed sphere into the saved
frequency-dependent acoustic origins, just as Stage 5 does. The sphere must
enclose the source. All outgoing coefficients in each fit participate, including
lower degrees which change during refitting. External coefficients are excluded.

At each sampled frequency let P be the spatial peak magnitude of the previous
internal fit. Convert both magnitudes to levels relative to that same P, clamp
both levels to the common assessment floor, and subtract. Take the largest
absolute difference across directions and then across frequencies. This is an
amplitude/SPL diagnostic; phase-only differences do not count. It is not an
average or the radiated-power fraction used by the tail graph.

With a -40 dB floor: -46 to -40 dB contributes zero, -46 to -37 contributes 3 dB,
and -30 to -24 contributes 6 dB. The same floor applies to both fits, referenced
to the previous fit's spatial peak separately at each frequency. Thus quiet
directions below the floor do not dominate, nor do louder frequency bins
automatically dominate. The plotted maximum only covers the sampled directions
and Stage 3 frequency bins; denser sampling may reveal narrower directivity detail.

Existing solves are reused. One extra solve at the order below the tested range
provides the first comparison. Reconstruction reuses the spherical basis across
all fits at each frequency, processing directions in blocks. The existing kr
order limit still applies: if neither fit actually adds a degree at any sampled
frequency, the point is marked as capped, not evidence of convergence. Numerical
truncation and nonfinite fields are excluded. A zero previous spatial peak gives
an undefined comparison; no valid frequencies produces a gap. A zero current
field is clamped normally. Rising changes can represent real detail or instability;
neither low nor high change establishes accuracy. The graph remains available
below RFT, but does not establish internal/external separation there.

Alongside the original PNG, Stage 3 saves *_spl_change.png and *_spl_change.json.
The JSON records each peak's signed/absolute change, frequency, point index,
polar angle theta (+Z axis), azimuth phi (+X toward +Y), previous/current
magnitudes in input pressure units, relative/clamped levels, previous peak
magnitude, actual fitted orders, valid/capped sample counts and evaluation
settings. A zero amplitude's unfloored dB is stored as null. Ties use the first
sample encountered. The same data is returned in result.spl_change. Changing
the tail reference does not change this diagnostic.


## Percentile definition and output

The main curve, `p99_change_db`, pools absolute local SPL changes across directions
and valid frequencies that add a degree, using NumPy's exact 99th percentile.
Capped frequency comparisons are excluded; all-capped orders are shown as zero,
marked and excluded from selection. `p99_all_frequencies_change_db` also records
the statistic including capped comparisons for inspection. Missing data is null.
The graph shows only the percentile curve; the original maximum and peak-direction details remain in the diagnostic data.
Temporary directional samples do not enter the saved JSON. `order_choices` in
the SPL output records the directly selected plateau and conservative alternative.
The old SPL-derived tail-reference helper remains only for analysis compatibility;
it is not used by the active optimizer.


The results viewer stacks stability, discarded power, and directivity change in one
scrollable view. Each metric marks its own candidate with a red dot. Radio buttons
list each available method separately, including when methods agree on N. The
eligible ratio knee has priority, followed by sound power and directivity change.
Directivity change is also evaluated in separation mode and must then exceed
20 dB Int/Ext. In SPL-only mode the offered solve order is plateau N minus two;
the previous additional minus-one choice is no longer offered.


Current directivity selection rule: choose the first valid tested order within
1 dB of the minimum incremental SPL change, then back off one order. Missing
values and fully capped increments are excluded. Use the nearest earlier valid
order if needed; if none exists, keep the lowest valid order. No preceding-drop
or four-order plateau requirement remains. A minimum at the last valid order
is flagged as unconfirmed convergence. Int/Ext priority and the >20 dB gate for
SPL backups when separation is usable remain unchanged.
