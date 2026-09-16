import json

import numpy as np
import pytest
from scipy.special import sph_harm_y, spherical_jn, spherical_yn

import stage3_spl_change as change
import stage3_optimize_she_settings as stage3


@pytest.mark.parametrize('old_db,new_db,expected', [(-46, -40, 0), (-46, -37, 3), (-30, -24, 6), (-24, -30, 6)])
def test_shared_floor_and_local_level_change(old_db, new_db, expected):
    previous = np.array([1, 10**(old_db/20)], dtype=complex)
    current = np.array([1, 10**(new_db/20)], dtype=complex)
    result = change.maximum_spl_change(previous, current)
    assert result['max_change_db'] == pytest.approx(expected, abs=1e-12)
    if expected:
        assert result['point_index'] == 1
        assert np.sign(result['signed_change_db']) == np.sign(new_db-old_db)
    # The same diagnostic at a completely different overall measurement gain.
    scaled = change.maximum_spl_change(previous*1e-90, current*1e-90)
    assert scaled['max_change_db'] == pytest.approx(expected, abs=1e-10)


def test_phase_is_ignored_and_invalid_previous_peak_is_undefined():
    assert change.maximum_spl_change([1, .1], [-1, .1j])['max_change_db'] == 0
    assert change.maximum_spl_change([0, 0], [1, 1]) is None
    assert change.maximum_spl_change([1, 0], [1, np.nan]) is None
    assert change.maximum_spl_change([1, 0], [1, .001])['max_change_db'] == 0


def test_reconstruction_translates_fixed_sphere_and_includes_refitted_low_degrees():
    sphere = change.evaluation_sphere(40, 1)
    origin = np.array([.1, -.03, .2])
    c0 = np.array([1+1j])
    c1 = np.array([2+2j, 0, .3j, 0])
    fields = change.reconstruct_internal_fits([c0, c1], 2000, 347, sphere, origin)
    r, theta, phi = change.translate_coordinates(*sphere, origin)
    k = 2*np.pi*2000/347
    h0 = spherical_jn(0, k*r) - 1j*spherical_yn(0, k*r)
    h1 = spherical_jn(1, k*r) - 1j*spherical_yn(1, k*r)
    np.testing.assert_allclose(fields[0], c0[0]*h0*sph_harm_y(0, 0, theta, phi))
    np.testing.assert_allclose(fields[1], 2*fields[0] + .3j*h1*sph_harm_y(1, 0, theta, phi))


def test_frequency_maximum_and_capped_orders_are_recorded():
    fits = {1: dict(usable=True, internal_coeffs=np.array([1, 0, 0, 0])),
            2: dict(usable=True, internal_coeffs=np.array([2, 0, 0, 0])),
            3: dict(usable=False, internal_coeffs=np.ones(16))}
    peak = change.evaluate_frequency_changes((1000, [2, 3], fits, 343, np.zeros(3), 40, 1, -40))
    assert len(peak) == 1
    assert peak[0]['max_change_db'] == pytest.approx(20*np.log10(2))
    assert peak[0]['order_capped']
    higher = dict(peak[0], max_change_db=8., frequency_hz=2000.)
    result = change.summarize_changes([2, 3], [peak, [higher]], 40, 1, -40, 2)
    assert result['peaks'][0]['frequency_hz'] == 2000
    assert result['peaks'][1] is None
    assert result['valid_frequency_counts'] == [2, 0]
    json.dumps(result, allow_nan=False)


def test_real_solver_incremental_diagnostic_preserves_existing_sweep(monkeypatch, tmp_path):
    radius, theta, phi = change.evaluation_sphere(80, .2)
    radius += np.arange(80) % 3 * .03
    coefficients = np.array([1, .1, .03j, .02, .01, 0, .04j, 0, .02])
    pressure = change.reconstruct_internal_fits([coefficients], 2000, 343,
                                               (radius, theta, phi), np.zeros(3))[0]
    names = [str(i) for i in range(80)]
    monkeypatch.setattr(stage3, 'load_and_parse_npz', lambda path: dict(
        freqs=np.array([2000.]), complex_data={n: np.array([pressure[i]]) for i, n in enumerate(names)},
        filenames=names, r_arr=radius, th_arr=theta, ph_arr=phi, origins_mm=None))
    args = dict(input_dir_opti=str(tmp_path), input_filename_opti='synthetic.npz', test_order_range=(2, 3),
                test_start_db_range=(-20, -60), test_lambda_range=(1e-7, .01), test_db_transition_span=20,
                freq_start_hz=2000, freq_end_hz=2000, use_process_pool=True, spl_sphere_points=64)
    result = stage3.run_open_branch_optimizer(**args)
    control = stage3.run_open_branch_optimizer(**args, spl_change_enabled=False, spl_floor_db=-12, save_plot=False)
    # Obsolete project/API settings cannot disable SPL or alter its fixed floor.
    assert control['spl_change'] is not None
    assert control['spl_change']['floor_db'] == -40
    np.testing.assert_allclose(result['step1']['ratios'], control['step1']['ratios'])
    np.testing.assert_allclose(result['step1']['internal_tail_power_db'], control['step1']['internal_tail_power_db'])
    assert result['recommended_key'] == control['recommended_key']
    diagnostic = result['spl_change']
    assert diagnostic['orders'] == [2, 3]
    assert diagnostic['valid_frequency_counts'] == [1, 1]
    assert diagnostic['peaks'][0]['previous_order_n'] == 1
    assert diagnostic['max_change_db'][0] > .01
    assert diagnostic['max_change_db'][1] < 1e-6
    assert 'details_path' not in diagnostic
    assert diagnostic['peaks'][0]['frequency_hz'] == 2000


def test_percentile_rejects_isolated_peak_and_excludes_capped_zero_dilution():
    old=np.ones(1000,dtype=complex)
    new=old.copy();new[:20]=10;new[0]=100
    active=change.maximum_spl_change(old,new,include_samples=True)
    active.update(order_n=2,order_capped=False)
    capped=change.maximum_spl_change(old,old,include_samples=True)
    capped.update(order_n=2,order_capped=True)
    result=change.summarize_changes([2,3],[[active],[capped],[capped],[capped]],1000,1,-40,4)
    assert result['max_change_db'][0]==40
    assert result['p99_change_db'][0]==20
    assert result['p99_all_frequencies_change_db'][0]==0
    assert result['added_degree_frequency_counts'][0]==1
    assert result['p99_change_db'][1] is None
    assert '_absolute_change_db' not in result['peaks'][0]
    json.dumps(result,allow_nan=False)
