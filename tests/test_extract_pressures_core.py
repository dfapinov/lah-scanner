import math
import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn, sph_harm_y

import extract_pressures_core
import schema
from complex_to_ir_core import complex_to_ir
from stage5_extract_pressures import generate_cta2034_coords, run_sweep_extraction
from stage5_pressure_utils import (
    centered_sweep_angles,
    clockwise_sweep_angles,
    get_min_phase_delay,
    nearest_acoustic_origin,
    preview_sweep_sequence,
    smooth_fractional_octave_response,
    unique_cartesian_points,
)
from utils import translate_coordinates


@pytest.fixture
def she_data():
    freqs = np.array([100.0, 1000.0, 10000.0])
    coeffs = np.array([
        [1.0 + 0.5j, -0.3 + 0.2j],
        [0.8 - 0.1j, 0.4 + 0.6j],
        [-0.2 + 0.9j, 0.7 - 0.4j],
    ])
    return {
        schema.FREQS: freqs,
        schema.COEFFS: coeffs,
        schema.N_USED: np.zeros(len(freqs), dtype=int),
        schema.ORIGINS_MM: np.array([[0.0, 0.0, 0.0], [25.0, -10.0, 5.0], [-30.0, 20.0, 15.0]]),
        schema.FS: 48000.0,
        schema.SPEED_OF_SOUND_MPS: 343.0,
    }


COORDS = np.array([[90.0, 0.0, 1.0], [55.0, -35.0, 1.35]])


def test_nearest_acoustic_origin_selects_bin_and_converts_mm_to_metres():
    actual_hz, origin_m = nearest_acoustic_origin(
        [800.0, 997.5, 1250.0],
        [[1.0, 2.0, 3.0], [45.0, -10.0, 234.0], [7.0, 8.0, 9.0]],
        1000.0,
    )

    assert actual_hz == pytest.approx(997.5)
    np.testing.assert_allclose(origin_m, [0.045, -0.010, 0.234])


def test_nearest_acoustic_origin_rejects_mismatched_origin_count():
    with pytest.raises(ValueError, match="one XYZ position"):
        nearest_acoustic_origin([800.0, 1000.0], [[1.0, 2.0, 3.0]], 1000.0)


@pytest.mark.parametrize("phase_offset", [0.0, 0.7, -2.4, 8.0 * np.pi + 0.3])
def test_min_phase_delay_recovers_known_delay_independent_of_phase_branch(phase_offset):
    freqs = np.arange(5.0, 24000.1, 5.0)
    c_sound = 347.0
    expected_distance = 1.017
    delay = expected_distance / c_sound
    pressure = np.exp(1j * (phase_offset - 2.0 * np.pi * freqs * delay))

    assert get_min_phase_delay(pressure, freqs, c_sound) == pytest.approx(
        expected_distance, abs=1e-9
    )


def test_min_phase_delay_rejects_deep_null_phase_outliers():
    freqs = np.arange(5.0, 24000.1, 5.0)
    c_sound = 347.0
    expected_distance = 0.8
    pressure = np.exp(-1j * 2.0 * np.pi * freqs * expected_distance / c_sound)
    pressure[500:510] = 1e-8 * np.exp(1j * np.linspace(-np.pi, np.pi, 10))

    assert get_min_phase_delay(pressure, freqs, c_sound) == pytest.approx(
        expected_distance, abs=2e-3
    )


@pytest.fixture
def local_tmp_path():
    """Use a workspace temp folder in restricted test environments."""
    path = Path(__file__).parent / ".tmp" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass


def _expected_order_zero(data, mode, use_origins, c_sound=343.0):
    theta = np.radians(COORDS[:, 0])
    phi = np.radians(COORDS[:, 1])
    radii = COORDS[:, 2]
    expected = np.empty((len(data[schema.FREQS]), len(COORDS)), dtype=np.complex128)
    for index, frequency in enumerate(data[schema.FREQS]):
        origin = data[schema.ORIGINS_MM][index] / 1000.0 if use_origins else np.zeros(3)
        r_eval, theta_eval, phi_eval = translate_coordinates(radii, theta, phi, origin)
        kr = 2.0 * math.pi * frequency * r_eval / c_sound
        harmonic = sph_harm_y(0, 0, theta_eval, phi_eval)
        internal = data[schema.COEFFS][index, 0] * (
            spherical_jn(0, kr) - 1j * spherical_yn(0, kr)
        ) * harmonic
        external = data[schema.COEFFS][index, 1] * spherical_jn(0, kr) * harmonic
        expected[index] = internal if mode == "Internal" else external if mode == "External" else internal + external
    return expected


@pytest.mark.parametrize("mode", ["Internal", "External", "Full"])
@pytest.mark.parametrize("use_origins", [False, True])
def test_pressure_extractor_matches_known_field_for_all_observation_modes(she_data, mode, use_origins):
    result = extract_pressures_core.evaluate_she_field(
        COORDS,
        she_data,
        obs_mode=mode,
        c_sound=343.0,
        use_optimized_origins=use_origins,
        corr_ir_pad_phase=False,
        use_process_pool=False,
    )

    expected = _expected_order_zero(she_data, mode, use_origins)
    np.testing.assert_allclose(result["complex"], expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result["magnitude"], 20.0 * np.log10(np.abs(expected) + np.finfo(float).eps))
    np.testing.assert_allclose(result["phase"], np.angle(expected, deg=True))
    np.testing.assert_array_equal(result["freqs"], she_data[schema.FREQS])
    assert result["fs"] == 48000.0


@pytest.mark.parametrize("padding_samples", [0, 17, None])
def test_pressure_extractor_capture_padding_setting(she_data, padding_samples):
    raw = extract_pressures_core.evaluate_she_field(
        COORDS[:1], she_data, corr_ir_pad_phase=False, use_process_pool=False
    )["complex"]
    corrected = extract_pressures_core.evaluate_she_field(
        COORDS[:1],
        she_data,
        corr_ir_pad_phase=True,
        ir_capture_padding_samples=padding_samples,
        use_process_pool=False,
    )["complex"]
    active_padding = extract_pressures_core.IR_CAPTURE_PADDING_SAMPLES if padding_samples is None else padding_samples
    phasor = np.exp(1j * 2.0 * np.pi * she_data[schema.FREQS] * active_padding / she_data[schema.FS])
    np.testing.assert_allclose(corrected, raw * phasor[:, None], rtol=2e-12, atol=2e-12)


def test_pressure_extractor_rejects_negative_capture_padding(she_data):
    with pytest.raises(ValueError, match="zero or greater"):
        extract_pressures_core.evaluate_she_field(
            COORDS[:1], she_data, ir_capture_padding_samples=-1, use_process_pool=False
        )


def test_h5_and_dictionary_inputs_are_equivalent(local_tmp_path, she_data):
    path = local_tmp_path / "coefficients.h5"
    with h5py.File(path, "w") as handle:
        for key, value in she_data.items():
            handle[key] = value

    from_dict = extract_pressures_core.evaluate_she_field(
        COORDS, she_data, obs_mode="External", corr_ir_pad_phase=False, use_process_pool=False
    )
    from_h5 = extract_pressures_core.evaluate_she_field(
        COORDS, path, obs_mode="External", corr_ir_pad_phase=False, use_process_pool=False
    )

    np.testing.assert_allclose(from_h5["complex"], from_dict["complex"])


def test_thread_and_process_backends_are_equivalent(monkeypatch, she_data):
    # Keep the real spawn path lightweight while still exercising serialization
    # and result assembly, which differ from the interactive thread backend.
    monkeypatch.setattr(extract_pressures_core.multiprocessing, "cpu_count", lambda: 1)
    threaded = extract_pressures_core.evaluate_she_field(
        COORDS,
        she_data,
        obs_mode="Full",
        corr_ir_pad_phase=False,
        use_process_pool=False,
    )
    processed = extract_pressures_core.evaluate_she_field(
        COORDS,
        she_data,
        obs_mode="Full",
        corr_ir_pad_phase=False,
        use_process_pool=True,
    )

    np.testing.assert_allclose(processed["complex"], threaded["complex"], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("mode", ["Internal", "External", "Full"])
def test_full_resolution_preview_matches_bulk_extractor(she_data, mode):
    bulk = extract_pressures_core.evaluate_she_field(
        COORDS[:1],
        she_data,
        obs_mode=mode,
        corr_ir_pad_phase=False,
        use_optimized_origins=True,
        use_process_pool=False,
    )
    with extract_pressures_core.PressureEvaluationSession(
        she_data, use_process_pool=False
    ) as session:
        preview = session.evaluate_preview_response(
            COORDS[0],
            obs_mode=mode,
            use_optimized_origins=True,
            ir_capture_padding_samples=0,
        )

    np.testing.assert_array_equal(preview["freqs"], bulk["freqs"])
    np.testing.assert_allclose(preview["complex"], bulk["complex"][:, 0], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        preview["ir"], complex_to_ir(bulk["complex"][:, 0], bulk["freqs"], target_fs=48000.0)
    )


def test_preview_ir_is_selected_point_but_timing_reference_is_on_axis(she_data):
    reference_distance = 0.8
    with extract_pressures_core.PressureEvaluationSession(
        she_data, use_process_pool=False
    ) as session:
        preview = session.evaluate_preview_response(
            COORDS[1],
            reference_coord_sph=COORDS[0],
            reference_distance=reference_distance,
            obs_mode="Internal",
            use_optimized_origins=True,
            ir_capture_padding_samples=0,
            subtract_tof="Ref Origin",
        )

    bulk = extract_pressures_core.evaluate_she_field(
        COORDS,
        she_data,
        obs_mode="Internal",
        corr_ir_pad_phase=False,
        use_optimized_origins=True,
        use_process_pool=False,
    )
    selected_pre_tof = bulk["complex"][:, 1]
    expected_frd = selected_pre_tof * np.exp(
        1j * 2.0 * np.pi * bulk["freqs"] * reference_distance / 343.0
    )
    np.testing.assert_allclose(preview["complex_pre_tof"], selected_pre_tof)
    np.testing.assert_allclose(preview["complex"], expected_frd)
    np.testing.assert_allclose(
        preview["ir"], complex_to_ir(selected_pre_tof, bulk["freqs"], target_fs=48000.0)
    )
    assert preview["tof_reference_time_s"] == pytest.approx(reference_distance / 343.0)


@pytest.mark.parametrize("tof_mode", ["IR Peak", "Min Phase Ref"])
def test_detected_preview_timing_stays_on_axis_when_selected_point_changes(she_data, tof_mode):
    with extract_pressures_core.PressureEvaluationSession(
        she_data, use_process_pool=False
    ) as session:
        on_axis = session.evaluate_preview_response(
            COORDS[0],
            reference_coord_sph=COORDS[0],
            obs_mode="Internal",
            ir_capture_padding_samples=0,
            subtract_tof=tof_mode,
        )
        off_axis = session.evaluate_preview_response(
            COORDS[1],
            reference_coord_sph=COORDS[0],
            obs_mode="Internal",
            ir_capture_padding_samples=0,
            subtract_tof=tof_mode,
        )

    assert off_axis["tof_reference_time_s"] == pytest.approx(on_axis["tof_reference_time_s"])
    assert not np.allclose(off_axis["ir"], on_axis["ir"])


def test_persistent_pressure_session_reuses_pool_and_closes(monkeypatch, she_data):
    monkeypatch.setattr(extract_pressures_core.multiprocessing, "cpu_count", lambda: 1)
    session = extract_pressures_core.PressureEvaluationSession(
        she_data, use_process_pool=True, worker_count=1
    )
    first = session.evaluate_field(COORDS[:1], corr_ir_pad_phase=False)
    pool = session._pool
    second = session.evaluate_field(COORDS[1:], corr_ir_pad_phase=False)
    assert session._pool is pool
    assert first["complex"].shape == second["complex"].shape == (3, 1)
    session.close()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.evaluate_field(COORDS[:1])


def test_centered_sweep_always_contains_on_axis():
    assert centered_sweep_angles(90, 20) == [0, -20, 20, -40, 40, -60, 60, -80, 80]
    assert centered_sweep_angles(0, 20) == [0]
    with pytest.raises(ValueError, match="increment"):
        centered_sweep_angles(90, 0)
    with pytest.raises(ValueError, match="range"):
        centered_sweep_angles(-1, 10)


def test_preview_sweep_numbering_runs_clockwise_from_on_axis():
    assert clockwise_sweep_angles(90, 20) == [0, 20, 40, 60, 80, -80, -60, -40, -20]
    assert clockwise_sweep_angles(0, 20) == [0]


def test_combined_preview_numbers_horizontal_then_vertical_without_duplicate_axis():
    sequence = preview_sweep_sequence(40, 20, "hor_vert")
    assert sequence == [
        ("horizontal", 0),
        ("horizontal", 20),
        ("horizontal", 40),
        ("horizontal", -40),
        ("horizontal", -20),
        ("vertical", 20),
        ("vertical", 40),
        ("vertical", -40),
        ("vertical", -20),
    ]


def test_combined_preview_deduplicates_front_and_rear_arc_crossings():
    points = []
    for arc, angle in preview_sweep_sequence(180, 20, "hor_vert"):
        angle_rad = np.radians(angle)
        if arc == "horizontal":
            points.append([np.cos(angle_rad), np.sin(angle_rad), 0.0])
        else:
            points.append([np.cos(angle_rad), 0.0, np.sin(angle_rad)])

    unique = np.asarray(unique_cartesian_points(points))
    assert len(unique) == 34
    assert np.count_nonzero(np.all(np.isclose(unique, [1.0, 0.0, 0.0]), axis=1)) == 1
    assert np.count_nonzero(np.all(np.isclose(unique, [-1.0, 0.0, 0.0]), axis=1)) == 1


@pytest.mark.parametrize("denominator", [3, 6, 12, 24, 48])
def test_preview_fractional_octave_smoothing_is_finite_and_reduces_ripple(denominator):
    freqs = np.geomspace(20.0, 20000.0, 1000)
    ripple = np.sin(np.linspace(0.0, 80.0 * np.pi, len(freqs)))
    pressure = 10.0 ** (ripple / 20.0) * np.exp(1j * (0.3 * ripple))
    magnitude, phase = smooth_fractional_octave_response(freqs, pressure, denominator)

    assert magnitude.shape == phase.shape == freqs.shape
    assert np.all(np.isfinite(magnitude))
    assert np.all(np.isfinite(phase))
    assert np.ptp(magnitude) < np.ptp(ripple)


@pytest.mark.parametrize(("direction", "expected_count"), [("horizontal", 9), ("vertical", 9), ("hor_vert", 17)])
def test_sweep_driver_uses_centered_angles(monkeypatch, local_tmp_path, direction, expected_count):
    captured = {}

    def fake_evaluate(coords_sph, **_kwargs):
        captured["coords"] = np.asarray(coords_sph)
        return {
            "freqs": np.array([100.0]),
            "complex": np.ones((1, len(coords_sph)), dtype=np.complex128),
            "fs": 48000.0,
        }

    monkeypatch.setattr("stage5_extract_pressures.evaluate_she_field", fake_evaluate)
    result = run_sweep_extraction(
        coeff_path="unused.h5",
        output_dir=local_tmp_path,
        direction=direction,
        range_deg=90,
        increment_deg=20,
        zero_theta=90.0,
        zero_phi=0.0,
        dist_mic=1.0,
        obs_mode="Internal",
        offset_xyz=(0.0, 0.0, 0.0),
        subtract_tof="Off",
        c_sound=343.0,
        save_to_disk=False,
        generate_ir_files=False,
        apply_mic_cal=False,
        use_process_pool=False,
    )

    assert len(result["data"]) == expected_count
    assert captured["coords"].shape[0] == expected_count
    assert any(key.endswith("hor+0") or key.endswith("ver+0") for key in result["data"])
    assert np.any(np.all(np.isclose(captured["coords"][:, :2], [90.0, 0.0]), axis=1))


def test_manual_coordinate_mode_is_absolute(monkeypatch, local_tmp_path):
    captured = {}

    def fake_evaluate(coords_sph, **_kwargs):
        captured["coords"] = np.asarray(coords_sph)
        return {"freqs": np.array([100.0]), "complex": np.ones((1, len(coords_sph))), "fs": 48000.0}

    monkeypatch.setattr("stage5_extract_pressures.evaluate_she_field", fake_evaluate)
    run_sweep_extraction(
        coeff_path="unused.h5",
        output_dir=local_tmp_path,
        use_coord_list=True,
        coord_list=[(70.0, -20.0, 1.25)],
        dist_mic=1.0,
        obs_mode="External",
        offset_xyz=(0.5, 0.5, 0.5),
        subtract_tof="Off",
        c_sound=343.0,
        save_to_disk=False,
        generate_ir_files=False,
        apply_mic_cal=False,
        use_process_pool=False,
    )

    np.testing.assert_allclose(captured["coords"], [[70.0, -20.0, 1.25]], atol=1e-12)


def test_cta2034_uses_fixed_ten_degree_grid():
    coords, mapping, _deviations = generate_cta2034_coords(2.0, 90.0, 0.0)
    assert len(coords) == 70
    assert mapping["H0"] == mapping["V0"]
    assert all(f"H{angle}" in mapping and f"V{angle}" in mapping for angle in range(0, 360, 10))
