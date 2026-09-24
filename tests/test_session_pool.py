"""Real spawned workers exercise ownership, recovery and bounded dispatch."""
import os
import time
import threading
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from session_pool import SessionPool, PoolView, borrow_pool, install_session_pool


def _identity(value):
    time.sleep(0.02)
    return os.getpid(), value


def _bad_task(value):
    raise ValueError('bad input')


def _crash(value):
    os._exit(3)


def test_session_reuse_limits_errors_and_restart():
    pool = SessionPool(workers=2)
    install_session_pool(pool)
    try:
        pool.start()
        # An immediate request must wait for initialization without losing work.
        view = borrow_pool(1)
        assert [v for _, v in view.map(_identity, range(4))] == list(range(4))
        executor = pool.executor()
        processes = list(executor._processes.values())
        assert len(processes) == 2 and all(p.is_alive() for p in processes)
        with borrow_pool(2) as borrowed:
            assert sorted(v for _, v in borrowed.imap_unordered(_identity, range(7), chunksize=2)) == list(range(7))
        borrowed.close()
        borrowed.join()
        assert pool.executor() is executor
        with pytest.raises(ValueError, match='bad input'):
            list(view.map(_bad_task, range(5)))
        assert list(view.map(abs, [-3])) == [3]
        with pytest.raises(Exception, match='terminated abruptly'):
            list(view.map(_crash, [0]))
        assert pool.state == 'failed'
        pool.start()
        assert list(borrow_pool().map(abs, [-4])) == [4]
        assert pool.executor() is not executor
        processes += list(pool.executor()._processes.values())
    finally:
        pool.close()
        install_session_pool(None)
    assert all(not p.is_alive() for p in processes)
    assert borrow_pool() is None


def test_close_during_startup_releases_waiters():
    pool = SessionPool(workers=2)
    pool.start()
    pool.close()
    pool._starter.join(timeout=10)
    assert not pool._starter.is_alive()
    with pytest.raises(RuntimeError, match='closed'):
        pool.executor()
    assert pool._executor is None


def test_startup_timeout_can_be_retried():
    pool = SessionPool(workers=1, startup_timeout=0)
    try:
        pool.start()
        with pytest.raises(RuntimeError, match='timed out'):
            pool.executor()
        pool._starter.join(timeout=10)
        pool.startup_timeout = 60
        pool.start()
        assert pool.executor() is not None
    finally:
        pool.close()


def test_queue_creation_failure_releases_waiters(monkeypatch):
    import session_pool

    def denied(*args):
        raise PermissionError('pipe unavailable')

    monkeypatch.setattr(session_pool.multiprocessing, 'get_context', denied)
    pool = SessionPool(workers=1)
    pool.start()
    assert pool._ready.wait(timeout=5)
    with pytest.raises(RuntimeError, match='pipe unavailable'):
        pool.executor()
    pool.close()


@pytest.mark.parametrize('unordered', [False, True])
def test_borrowed_view_bounds_concurrency(unordered):
    # A thread executor lets us observe simultaneous tasks without adding any
    # shared state or instrumentation to the actual worker implementation.
    active = peak = 0
    lock = threading.Lock()

    def task(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return value

    pool = SessionPool(workers=8)
    with ThreadPoolExecutor(max_workers=8) as executor:
        pool._executor = executor
        pool.state = 'ready'
        pool._ready.set()
        view = PoolView(pool, workers=2)
        mapper = view.imap_unordered if unordered else view.map
        assert sorted(mapper(task, range(12))) == list(range(12))
        assert peak == 2


@pytest.mark.parametrize('workers', [2, None])
def test_gui_early_run_and_shutdown(tmp_path, workers):
    script = r'''
import sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path[:0] = ['src', 'src/process']
import session_pool
factory = session_pool.SessionPool
worker_count = None if sys.argv[2] == 'None' else int(sys.argv[2])
session_pool.SessionPool = lambda: factory(workers=worker_count)
project = Path(sys.argv[1])
ir_dir = project / 'ir'
ir_dir.mkdir()
for i in range(3):
    signal = np.zeros(4096)
    signal[100+i] = 0.7
    signal[140+i] = 0.1
    sf.write(ir_dir / f'IR_{i}.wav', signal, 48000, subtype='FLOAT')
settings = dict(input_dir=str(ir_dir), out_dir=str(project / 'output'),
    output_filename='stage1.npz', fdw_rft_ms=5, fdw_oct_res=6,
    fdw_max_cap_ms=20, enable_smoothing=True, smoothing_oct_res=6,
    fdw_alpha_hf=0.2, fdw_alpha_lf=1.0, fdw_windows_per_oct=3,
    peak_detect_threshold_db=-12, enable_auto_gain=True, target_peak_db=-3,
    keep_raw_and_smoothed=True, use_process_pool=True)
import hals_post_ui_core as ui
app = ui.SpkrScannerApp()
app.withdraw()
completed = []
assert app._start_stage_job('Stage 1', lambda: app._run_stage1_job(settings), completed.append)
deadline = time.monotonic() + 60
while not completed and time.monotonic() < deadline:
    app.update()
    time.sleep(0.01)
assert len(completed) == 1, app.cli_text.get('1.0', 'end')
assert (project / 'output' / 'stage1.npz').is_file()
assert (project / 'output' / 'stage1_smoothed.npz').is_file()
settings['use_process_pool'] = False
_, reference = app._run_stage1_job(settings)
actual = completed[0][1]
np.testing.assert_allclose(actual[0], reference[0])
for index in (1, 2):
    assert len(actual[index]) == 3
    for name, values in actual[index].items():
        np.testing.assert_allclose(values, reference[index][name])
app._poll_process_pool()
assert app.status_var.get().startswith('Ready.')
assert app.process_pool.initialized_workers == app.process_pool.workers
processes = list(app.process_pool.executor()._processes.values())
app.on_closing()
deadline = time.monotonic() + 15
while app._pool_close_thread.is_alive() and time.monotonic() < deadline:
    app.update()
    time.sleep(0.01)
assert not app._pool_close_thread.is_alive()
assert all(not p.is_alive() for p in processes)
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path), str(workers)], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
