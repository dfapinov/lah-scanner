import numpy as np
import pytest

import stage3_optimize_she_settings as stage3


def test_three_order_choices_use_strict_thresholds():
    choices = stage3.stage3_order_choices([2, 3, 4, 5], [25, 23, 21, 20], [1]*4,
                                         {'n': 3}, [-20, -21, -np.inf, np.nan], 4)
    assert {key: value['n'] for key, value in choices.items()} == {'knee': 3}


def test_recommend_highest_eligible_of_three_candidates():
    choices = stage3.stage3_order_choices([2, 3, 4, 5, 6], [24, 23, 22, 21, 19], [1]*5,
                                         {'n': 3}, [-19, -21, -25, -26, -np.inf], 6)
    assert choices['tail']['n'] == 4
    assert stage3.recommended_stage3_choice(choices) == 'knee'
    choices = stage3.stage3_order_choices([2, 3, 4], [20, 19, 18], [1]*3,
                                         {'n': 3}, [-21, -26, -np.inf], 4)
    assert choices == {}
    assert stage3.recommended_stage3_choice(choices) is None


@pytest.mark.parametrize('ratio,expected', [(24, 'tail'), (19, None)])
def test_optimizer_returns_only_eligible_choices(monkeypatch, ratio, expected):
    monkeypatch.setattr(stage3, 'load_and_parse_npz', lambda path: {
        'freqs': np.array([1000.]), 'complex_data': {'p': np.array([1.])},
        'filenames': ['p'], 'r_arr': np.ones(1), 'th_arr': np.ones(1),
        'ph_arr': np.ones(1), 'origins_mm': None})
    monkeypatch.setattr(stage3, 'get_grid_limit', lambda *args: (5, 72))
    def worker(args):
        n = args[5]
        shares = np.array([.9, .05, .04, .008, .0015, .0005])[:n+1]
        shares /= shares.sum()
        return dict(N=n, st_db=args[6], mx_db=args[7], lam=args[8], ratio_db=ratio,
                    err=1, internal_degree_fraction=shares[-1], internal_degree_shares=shares)
    monkeypatch.setattr(stage3, '_worker', worker)
    result = stage3.run_open_branch_optimizer('', 'unused', (2, 5), 1000, 1000,
                                              save_plot=False, use_process_pool=False)
    assert result['recommended_key'] == expected
    if expected:
        assert result['options'][expected]['n'] == 3
    else:
        assert not result['options']
        assert result['warning']


def test_choices_exclude_reference_zero_and_handle_missing_choices():
    choices = stage3.stage3_order_choices([2, 3], [20, 19], [1, 1], None, [-19, -np.inf], 3)
    assert choices == {}
    choices = stage3.stage3_order_choices([2, 3], [22, 21], [1, 1], None, [-np.inf, np.nan], 2)
    assert 'tail' not in choices


def test_tail_choice_updates_with_reference():
    args = ([2, 3, 4], [25, 23, 21], [1, 1, 1], None)
    assert 'tail' not in stage3.stage3_order_choices(*args, [-23.99, -np.inf, np.nan], 3)
    assert stage3.stage3_order_choices(*args, [-24.0, -np.inf, np.nan], 3)['tail']['n'] == 2
    assert stage3.stage3_order_choices(*args, [-19, -25, -np.inf], 4)['tail']['n'] == 3


def test_tail_reference_uses_highest_qualifying_order_and_strict_threshold():
    assert stage3.select_tail_reference([2, 3, 4, 5], [30, 21, 20, 19]) == {'n': 3, 'ratio': 21., 'fallback': False}
    assert stage3.select_tail_reference([2, 3, 4], [19, 20, 18]) == {'n': 3, 'ratio': 20., 'fallback': True}
    with pytest.raises(ValueError):
        stage3.select_tail_reference([2], [np.nan])


def test_cumulative_tail_uses_one_reference_and_linear_averaging():
    coeffs = np.zeros(18, dtype=complex)
    coeffs[0] = np.sqrt(90)
    coeffs[2] = np.sqrt(9)
    coeffs[8] = 1j
    coeffs[1::2] = 1e9
    shares = stage3.calc_internal_degree_shares(coeffs, 2)
    np.testing.assert_allclose(shares, [.9, .09, .01])
    assert stage3.calc_internal_degree_shares(coeffs, 3) is None
    values, count = stage3.calc_cumulative_tail([0, 1, 2, 3], 2, [shares, shares, None])
    np.testing.assert_allclose(values[:2], [-10, -20])
    assert values[2] == -np.inf
    assert np.isnan(values[3])
    assert count == 2
    values, count = stage3.calc_cumulative_tail([1, 2], 2, [None])
    assert np.all(np.isnan(values)) and count == 0


def test_saved_tail_plot(tmp_path):
    path = tmp_path / 'tail.png'
    assert stage3.save_stage3_order_sweep_plot(
        [2, 3, 4], [22, 21, 18], save_path=str(path),
        internal_tail_power_db=[-25, -np.inf, np.nan],
        tail_reference={'n': 3, 'ratio': 21, 'fallback': False}) == str(path)


def test_octave_frequency_selection_and_endpoints():
    freqs = np.array([0, 100, 141, 200, 283, 400, 450, 500])
    indices = stage3.select_stage3_frequency_indices(freqs, 100, 450, 1)
    np.testing.assert_array_equal(freqs[indices], [100, 200, 400, 450])
    indices = stage3.select_stage3_frequency_indices(freqs, 450, 100, 2)
    np.testing.assert_array_equal(freqs[indices], [100, 141, 200, 283, 400, 450])


def test_all_bins_and_fine_resolution_deduplicate():
    freqs = np.array([0, 100, 101, 102, 200])
    for resolution in (0, 96):
        indices = stage3.select_stage3_frequency_indices(freqs, 0, 102, resolution)
        np.testing.assert_array_equal(indices, [1, 2, 3])
    np.testing.assert_array_equal(stage3.select_stage3_frequency_indices(freqs, 101, 101), [2])


@pytest.mark.parametrize('resolution', [-1, .5, np.nan, np.inf])
def test_invalid_octave_resolution(resolution):
    with pytest.raises(ValueError, match='resolution'):
        stage3.select_stage3_frequency_indices([100, 200], 100, 200, resolution)


def test_empty_positive_frequency_range():
    with pytest.raises(ValueError, match='positive frequency'):
        stage3.select_stage3_frequency_indices([0, 100], 0, 50)


def test_internal_degree_share_ignores_external_and_preserves_scaling():
    coeffs = np.zeros(18, dtype=complex)  # degrees 0, 1, 2
    coeffs[0] = np.sqrt(99)
    coeffs[8] = 1j  # first C coefficient of degree 2
    coeffs[1::2] = 1e6
    for scale in (1, 1e-100, 1e100):
        assert stage3.calc_internal_degree_fraction(coeffs * scale, 2) == pytest.approx(.01)
    assert stage3.calc_internal_degree_fraction(coeffs, 1) == 0
    assert np.isnan(stage3.calc_internal_degree_fraction(coeffs, 3))
    assert np.isnan(stage3.calc_internal_degree_fraction(np.zeros(18), 2))


def test_aggregate_uses_linear_shares_and_excludes_unavailable():
    db, count = stage3.aggregate_internal_degree_power([0, .02, np.nan])
    assert db == pytest.approx(-20)
    assert count == 2
    assert stage3.aggregate_internal_degree_power([0]) == (-np.inf, 1)
    db, count = stage3.aggregate_internal_degree_power([np.nan])
    assert np.isnan(db)
    assert count == 0


def test_worker_does_not_attribute_capped_degree_to_requested_order(monkeypatch):
    monkeypatch.setattr(stage3, 'get_kr_limit', lambda *args: 1)
    monkeypatch.setattr(stage3, '_solve_one_frequency',
                        lambda **kwargs: (np.ones(8), {'residual_norm': 0}))
    coords = np.ones(4)
    result = stage3._worker((1000, coords, coords, coords, coords, 2,
                             -50, -60, 1e-10, 1, 343, 2))
    assert np.isnan(result['internal_degree_fraction'])


def test_worker_uses_dataset_sound_speed(monkeypatch):
    monkeypatch.setattr(stage3, 'get_kr_limit', lambda *args: 2)
    def solve(**kwargs):
        assert kwargs['k_val'] == pytest.approx(2 * np.pi * 1000 / 347)
        return np.ones(18), {'residual_norm': 0}
    monkeypatch.setattr(stage3, '_solve_one_frequency', solve)
    coords = np.ones(4)
    result = stage3._worker((1000, coords, coords, coords, coords, 2,
                             -50, -70, 1e-10, 1, 347, 2))
    assert result['internal_degree_fraction'] == pytest.approx(5 / 9)


def test_saved_plot_with_zero_and_unavailable_power(tmp_path):
    path = tmp_path / 'sweep.png'
    assert stage3.save_stage3_order_sweep_plot(
        [2, 3, 4], [30, 24, 18], save_path=str(path),
        internal_degree_power_db=[-20, -np.inf, np.nan],
    ) == str(path)
    assert path.stat().st_size > 0
