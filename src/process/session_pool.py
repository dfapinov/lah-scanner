"""App-owned spawn workers; command-line stages retain their local pools.

Borrowed views never own worker lifetime. Maps bound the number of in-flight
chunks so a large shared pool still respects each stage's memory/CPU limit.
"""
import itertools
import multiprocessing
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool


_session = None


def install_session_pool(pool):
    global _session
    _session = pool


def borrow_pool(workers=None):
    return None if _session is None else PoolView(_session, workers)


def _initialize(ready, release):
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    # Import the actual worker modules before reporting readiness (no GUI).
    import stage1_fdwsmooth
    import stage2_centre_origin
    import stage3_optimize_she_settings
    import stage3_spl_change
    import stage4_run_she_solve
    import extract_pressures_core
    ready.put(os.getpid())
    # Prevent quick probes from completing while the parent is still submitting
    # them. Otherwise the executor reuses idle workers instead of spawning the
    # requested count, and readiness waits forever for workers that don't exist.
    release.wait()


def _ping():
    return os.getpid()


def _chunk(func, items):
    return [func(item) for item in items]


class SessionPool:
    def __init__(self, workers=None, startup_timeout=180):
        # ProcessPoolExecutor on Windows supports at most 61 workers.
        default_workers = max(1, (os.cpu_count() or 1) // 2)
        self.workers = workers or default_workers
        if os.name == 'nt':
            self.workers = min(self.workers, 61)
        self.startup_timeout = startup_timeout
        self.state = 'new'
        self.error = ''
        self.initialized_workers = 0
        self._executor = None
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._starter = None

    def start(self):
        with self._lock:
            if self.state not in ('new', 'failed'):
                return
            if self._starter is not None and self._starter.is_alive():
                return
            self.state, self.error = 'starting', ''
            self.initialized_workers = 0
            self._ready.clear()
            self._starter = threading.Thread(target=self._warm, daemon=True, name='pool-startup')
            self._starter.start()

    def _warm(self):
        ready = None
        release = None
        try:
            deadline = time.monotonic() + self.startup_timeout
            ctx = multiprocessing.get_context('spawn')
            ready = ctx.Queue()
            release = ctx.Event()
            with self._lock:
                if self.state == 'closed':
                    return
                self._executor = ProcessPoolExecutor(self.workers, mp_context=ctx,
                                                     initializer=_initialize, initargs=(ready, release))
                probes = [self._executor.submit(_ping) for _ in range(self.workers)]
            pids = set()
            while len(pids) < self.workers:
                if self.state == 'closed':
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError('Worker startup timed out')
                for probe in probes:
                    if probe.done():
                        probe.result()  # Surface initializer failures promptly.
                try:
                    pids.add(ready.get(timeout=0.1))
                    self.initialized_workers = len(pids)
                except queue.Empty:
                    pass
            release.set()
            for probe in probes:
                probe.result(timeout=max(0, deadline - time.monotonic()))
            with self._lock:
                if self.state != 'closed':
                    self.state = 'ready'
        except Exception as exc:
            self.fail(exc)
        finally:
            if release is not None:
                release.set()
            if ready is not None:
                ready.close()
            self._ready.set()

    def executor(self):
        self._ready.wait()
        with self._lock:
            if self.state != 'ready':
                raise RuntimeError(f'Process pool {self.state}: {self.error}. Use Retry pool to restart it.')
            return self._executor

    def _stop(self):
        executor, self._executor = self._executor, None
        if executor is None:
            return
        # Python 3.10-3.13 lack terminate_workers(). Snapshot before shutdown
        # clears this internal collection; kept here as the compatibility shim.
        processes = list((getattr(executor, '_processes', None) or {}).values())
        terminate = getattr(executor, 'terminate_workers', None)
        if terminate is not None:
            terminate()
        else:
            executor.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)

    def fail(self, exc):
        with self._lock:
            if self.state != 'closed':
                self.state, self.error = 'failed', str(exc)
            self._stop()
            self._ready.set()

    def close(self):
        with self._lock:
            self.state = 'closed'
            self._ready.set()
            self._stop()


class PoolView:
    def __init__(self, owner, workers=None):
        self.owner = owner
        self.workers = min(workers or owner.workers, owner.workers)

    def _map(self, func, iterable, chunksize=1, unordered=False):
        executor = self.owner.executor()
        items = iter(iterable)
        pending = []
        try:
            while True:
                while len(pending) < self.workers:
                    chunk = list(itertools.islice(items, max(1, chunksize)))
                    if not chunk:
                        break
                    pending.append(executor.submit(_chunk, func, chunk))
                if not pending:
                    break
                future = (next(iter(wait(pending, return_when=FIRST_COMPLETED).done))
                          if unordered else pending[0])
                pending.remove(future)
                yield from future.result()
        except BrokenProcessPool as exc:
            self.owner.fail(exc)
            raise
        finally:
            # Ordinary task errors do not poison the pool. Drain/cancel this
            # call's remaining work before permitting another stage to run.
            for future in pending:
                future.cancel()
            for future in pending:
                try:
                    future.result()
                except Exception:
                    pass

    def map(self, func, iterable, chunksize=1):
        return self._map(func, iterable, chunksize)

    imap = map

    def imap_unordered(self, func, iterable, chunksize=1):
        return self._map(func, iterable, chunksize, unordered=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def close(self):
        pass

    def join(self):
        pass
