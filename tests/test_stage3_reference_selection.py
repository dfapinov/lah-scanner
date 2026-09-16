import numpy as np
import pytest
import stage3_optimize_she_settings as stage3


def diagnostic(values, support=None):
    return dict(orders=list(range(2, 2+len(values))), max_change_db=values,
                added_degree_frequency_counts=support or [10]*len(values))


def test_sustained_rise_reference_backs_off_one_more_order():
    d = diagnostic([20, 15, 10, 11, 12, 14, 15, 16])
    ref = stage3.select_spl_tail_reference(d['orders'], [19]*8, d)
    assert ref['rise_start_n'] == 7
    assert ref['pre_rise_n'] == 6
    assert ref['n'] == 5
    assert ref['provisional']


def test_no_rise_disabled_or_capped_minimum_is_not_a_reference():
    orders = list(range(2, 10))
    assert stage3.select_spl_tail_reference(orders, [21]*8, None)['n'] is None
    for d in [diagnostic([20, 15, 10, 9, 8, 7, 6, 5]),
              diagnostic([20, 15, 10, 0, 0, 4, 5, 6], [10,10,10,0,0,10,10,10]),
              diagnostic([20, 15, 10, 11, 12, 14, None, 16])]:
        assert stage3.select_spl_tail_reference(orders, [21]*8, d)['n'] is None


def test_soft_boundary_and_tail_priority_over_higher_knee():
    options = stage3.stage3_order_choices([2,3,4,5], [25]*4, [1]*4,
                                          {'n':4}, [-24.49,-24.5,-27,-np.inf], 5)
    assert options['tail']['n'] == 3
    assert stage3.recommended_stage3_choice(options) == 'knee'
    assert stage3.stage3_order_choices([2,3], [25]*2, [1]*2, None,
                                       [np.nan]*2, None, tail_only=True) == {}


@pytest.mark.parametrize('ratio,below_rft,expect_spl', [(19,False,True), (20,False,True),
                                                      (21,False,False), (21,True,True)])
@pytest.mark.parametrize('borrow_pool', [False, True])
def test_optimizer_selects_spl_when_no_separation_reference(monkeypatch, ratio, below_rft, expect_spl, borrow_pool):
    import stage3_spl_change as spl
    monkeypatch.setattr(stage3, 'load_and_parse_npz', lambda _: dict(
        freqs=np.array([1000.]), complex_data={'p':np.array([1.])}, filenames=['p'],
        r_arr=np.ones(1), th_arr=np.ones(1), ph_arr=np.ones(1), origins_mm=None,
        stage3_rft_lower_hz=2000 if below_rft else 500))
    monkeypatch.setattr(stage3, 'get_grid_limit', lambda *args: (9,200))
    def worker(args):
        n=args[5]
        # Simulate a capped frequency: fallback must retain its limited shares.
        shares=np.array([.99,.008,.001,.001])[:min(n+1,4)]
        shares=shares/shares.sum()
        return dict(N=n, st_db=args[6], mx_db=args[7], lam=args[8], ratio_db=ratio,
                    err=1, internal_degree_fraction=.001,
                    internal_degree_shares=shares if not expect_spl else None,
                    limited_degree_shares=shares)
    monkeypatch.setattr(stage3, '_worker', worker)
    monkeypatch.setattr(spl, 'evaluate_frequency_changes', lambda _: [])
    monkeypatch.setattr(spl, 'summarize_changes', lambda *args: dict(diagnostic([20,15,10,6,4,4.1,4.2,4]), p99_change_db=[20,15,10,6,4,4.1,4.2,4]))
    calls = []
    class BorrowedPool:
        def map(self, fn, tasks):
            calls.append(fn.__name__)
            return map(fn, tasks)
    pool = BorrowedPool() if borrow_pool else None
    result=stage3.run_open_branch_optimizer('', 'unused', (2,9), (-20,-60), (1e-7,.01), 20,
                                           800,1200, save_plot=False, use_process_pool=borrow_pool,
                                           process_pool=pool)
    if borrow_pool:
        assert calls == ['worker', 'evaluate_frequency_chunk']
    assert result['tail_only'] == expect_spl
    assert result['below_rft'] == below_rft
    assert result['test_band_hz'] == ([300,1200] if below_rft else [800,1200])
    assert result['step1']['tail_reference']['n'] == (None if expect_spl else 9)
    assert result['recommended_key'] == ('spl' if expect_spl else 'tail')
    assert result['options'][result['recommended_key']]['n'] == (5 if expect_spl else 2)
    if expect_spl:
        assert 'tail' not in result['options']
        assert 'spl_safe' not in result['options']
        assert result['step1']['tail_reference']['sample_count'] == 0
        assert 'not used for selection' in result['ratio_note']


def test_unresolved_reference_reports_specific_reason():
    assert 'disabled' in stage3.select_spl_tail_reference([2,3], [19,19], None)['reason']
    d=diagnostic([10,9,8,7])
    assert 'No sustained SPL rise' in stage3.select_spl_tail_reference(d['orders'], [19]*4, d)['reason']


def test_shallow_plateau_uses_n11_and_ignores_capped_endpoint():
    d=diagnostic([24.208,15.231,19.547,20.031,14.76,16.442,9.073,9.329,
                  6.346,4.141,4.838,5.904,4.161,0], [25]*13+[0])
    d['p99_change_db']=[9.126,6.368,5.876,3.098,3.246,2.074,1.558,1.311,
                        .992,.949,.780,1.195,.508,0]
    ref=stage3.select_spl_tail_reference(d['orders'],[10]*14,d)
    assert ref['n']==11
    assert ref['method']=='SPL low-change plateau onset'
    assert ref['plateau_start_n']==10
    assert ref['plateau_end_n']==13
    assert ref['primary_metric']=='p99_change_db'
    # A rising percentile contradicts a seemingly quiet maximum plateau.
    d['p99_change_db'][-5:-1]=[3,3,3,3]
    assert stage3.select_spl_tail_reference(d['orders'],[10]*14,d)['n'] != 11


def test_plateau_needs_four_real_increments_after_descent():
    for d in [diagnostic([4]*10), diagnostic([20,15,10,4,4,4,0], [10]*6+[0]),
              diagnostic([20,15,10,4,4,None,4])]:
        assert stage3.select_spl_tail_reference(d['orders'],[10]*len(d['orders']),d)['n'] is None


def test_smoother_percentile_does_not_delay_existing_warning():
    d=diagnostic([20,15,10,11,12,14,15,16])
    d['p99_change_db']=[8,6,3,2,1,1.5,2,2.5]
    ref=stage3.select_spl_tail_reference(d['orders'],[10]*8,d)
    assert ref['n']==5
    assert ref['reference_candidates']['maximum']==5


def test_direct_spl_selection_ignores_maximum_and_ratio_and_offers_backoff():
    d=diagnostic([99]*8)
    d['p99_change_db']=[20,15,10,6,4,4.1,4.2,4]
    options,_=stage3.select_spl_order_choices(d['orders'],[-10]*8,[1]*8,d)
    assert options['spl']['n']==5
    assert options['spl']['near_minimum_n']==6
    assert options['spl']['backoff_orders']==1
    assert set(options)=={'spl'}
    assert stage3.recommended_stage3_choice(options)=='spl'
    changed,_=stage3.select_spl_order_choices(d['orders'],[30]*8,[.001]*8,
                                           dict(d,max_change_db=[0]*8))
    assert {k:v['n'] for k,v in options.items()}=={k:v['n'] for k,v in changed.items()}
    assert all(np.isnan(v['tail_db']) for v in options.values())


def test_direct_spl_cannot_use_capped_zeros_or_legacy_maximum_instead():
    d=diagnostic([20,15,10,6,0,0,0,0], [10]*4+[0]*4)
    d['p99_change_db']=d['max_change_db']
    assert stage3.select_spl_order_choices(d['orders'],[0]*8,[1]*8,d)[0]['spl']['n']==4
    assert stage3.select_spl_order_choices(d['orders'],[0]*8,[1]*8,diagnostic([20,15,10,6,4,4,4,4]))[0]=={}


@pytest.mark.parametrize('values,counts,expected', [
    ([4,4,4,4],[1]*4,2),
    ([9,8,7,6],[1]*4,3),
    ([9,None,7,6],[1]*4,2),
    ([9,8,0,0],[1,1,0,0],2),
    ([9,8,7,6],[0]*4,None),
    ([9,7.01,7,6],[1]*4,3),
])
def test_simple_spl_boundary_and_missing_support(values,counts,expected):
    d=diagnostic(values,counts)
    d['p99_change_db']=values
    options,_=stage3.select_spl_order_choices(d['orders'],[25]*4,[1]*4,d)
    assert options.get('spl',{}).get('n')==expected
