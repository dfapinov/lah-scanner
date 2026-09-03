import math

import numpy as np
import pytest
from scipy.special import spherical_jn, sph_harm_y

import she_solver_core
from utils import hankel2


def _well_conditioned_grid():
    point_count = 32
    index = np.arange(point_count, dtype=float)
    radii = np.where((index.astype(int) % 2) == 0, 0.65, 1.15)
    z = 1.0 - 2.0 * (index + 0.5) / point_count
    theta = np.arccos(z)
    phi = np.mod(index * math.pi * (3.0 - math.sqrt(5.0)), 2.0 * math.pi)
    return radii, theta, phi


def _basis_matrix(coords, order, k_value):
    radii, theta, phi = coords
    columns = []
    for n in range(order + 1):
        for m in range(-n, n + 1):
            harmonic = sph_harm_y(n, m, theta, phi)
            columns.extend((hankel2(n, radii * k_value) * harmonic,
                            spherical_jn(n, radii * k_value) * harmonic))
    return np.column_stack(columns)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("condition_metrics", [False, True])
def test_solver_recovers_known_dual_basis_coefficients(order, condition_metrics):
    coords = _well_conditioned_grid()
    k_value = 7.25
    basis = _basis_matrix(coords, order, k_value)
    rng = np.random.default_rng(20260902 + order)
    expected = rng.normal(size=basis.shape[1]) + 1j * rng.normal(size=basis.shape[1])
    measured = basis @ expected

    actual, metrics = she_solver_core._solve_one_frequency(
        400.0,
        measured,
        coords,
        order,
        k_val=k_value,
        CONDITION_METRICS=condition_metrics,
        noise_floor_start_db=-30.0,
        noise_floor_max_db=-50.0,
        max_lambda=0.0,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(metrics["residual_vector"], 0.0, atol=2e-10)
    assert metrics["residual_norm"] < 2e-10
    assert metrics["final_N"] == order
    assert metrics["stop_reason"] == "Target N Reached"
    assert np.isfinite(metrics["cond_pre"])
    assert np.isfinite(metrics["cond_post"])


def test_solver_uses_configured_speed_when_wavenumber_is_omitted(monkeypatch):
    coords = _well_conditioned_grid()
    frequency = 800.0
    speed = 340.0
    k_value = 2.0 * math.pi * frequency / speed
    basis = _basis_matrix(coords, 1, k_value)
    expected = np.arange(1, basis.shape[1] + 1) * (0.2 + 0.1j)
    monkeypatch.setattr(she_solver_core, "SPEED_OF_SOUND", speed)

    actual, _ = she_solver_core._solve_one_frequency(
        frequency, basis @ expected, coords, 1, max_lambda=0.0
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize(
    ("start_db", "maximum_db"),
    [(-30.0, -50.0), (-20.0, -20.0), (0.0, 0.0)],
)
def test_solver_regularization_settings_produce_finite_damped_solution(start_db, maximum_db):
    coords = _well_conditioned_grid()
    basis = _basis_matrix(coords, 2, 4.5)
    measured = basis @ np.ones(basis.shape[1], dtype=np.complex128)

    actual, metrics = she_solver_core._solve_one_frequency(
        250.0,
        measured,
        coords,
        2,
        k_val=4.5,
        noise_floor_start_db=start_db,
        noise_floor_max_db=maximum_db,
        max_lambda=0.01,
    )

    assert np.all(np.isfinite(actual))
    assert np.isfinite(metrics["residual_norm"])
    assert np.isfinite(metrics["cond_post"])
    assert metrics["residual_norm"] >= 0.0


def test_solver_reports_radial_function_instability(monkeypatch):
    coords = _well_conditioned_grid()
    monkeypatch.setattr(
        she_solver_core,
        "hankel2",
        lambda _n, values: np.full_like(values, np.nan, dtype=np.complex128),
    )

    coeffs, metrics = she_solver_core._solve_one_frequency(
        100.0, np.ones(len(coords[0])), coords, 1, k_val=1.0
    )

    assert coeffs.size == 0
    assert metrics["final_N"] == -1
    assert metrics["stop_reason"] == "Instability"


def test_solver_requires_a_wavenumber_without_speed_metadata(monkeypatch):
    monkeypatch.setattr(she_solver_core, "SPEED_OF_SOUND", None)
    with pytest.raises(ValueError, match="Speed of Sound"):
        she_solver_core._solve_one_frequency(
            100.0, np.ones(32), _well_conditioned_grid(), 0
        )

