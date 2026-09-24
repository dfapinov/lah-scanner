# Shared processing workers

HALS Post starts its process pool in the background when the app opens. The
bottom-left status changes from **Starting the process pool...** to
**Ready. Process pool: N workers.** only after every worker has imported the
processing modules. Project selection and settings remain available meanwhile.
During startup the status also shows how many workers have initialized.
If Run is pressed early, the job waits in its background thread for readiness.

Workers remain available across projects and stages until the app closes.
Stages 1–5 and the Stage 5 live preview borrow the same pool. Closing a preview
releases its coefficient data without closing the workers. Starting a stage
closes the preview and waits for its current calculation in the background.

The pool starts half as many workers as logical CPUs (rounded down, minimum
one), capped at 61 on Windows by Python's process executor. For example, 24
logical CPUs start 12 workers. This reserves memory for the imported numerical libraries
even while idle. It does not make every calculation use every worker:

| Work | Maximum simultaneous worker tasks |
| --- | --- |
| Stage 1 IR processing | Pool size |
| Stage 2 speed-of-sound candidates | Smaller of candidate count and pool size |
| Stage 2 landscape rendering | Half the logical CPUs, capped by pool size |
| Stage 3 order/regularization fits | 6, capped by pool size |
| Stage 3 spherical evaluation | 4, capped by pool size |
| Stage 4 solve | Configured jobs, capped by pool size |
| Stage 5 extraction and preview | Pool size |

Stage 2's sequential search and its two-thread descent retain their existing
algorithm. Sharing workers does not remove input loading, data serialization,
or computation time; it removes repeated worker startup and module imports.

Startup errors and a three-minute worker-initialization timeout produce an
unavailable status. Waiting jobs fail through the normal processing error log.
Use **Retry pool** once the current job has finished to start fresh workers,
then run the stage again. A crashed worker invalidates the pool in the same
way. Ordinary task errors leave healthy workers available; jobs are never
automatically rerun. Closing the app terminates workers in the background,
including during startup or processing, before destroying the window.

Standalone command-line scripts keep their own per-run pools. The shared
service is installed only by the GUI. No project data is retained in worker
globals, and native numerical-library threads are limited to one per worker
to avoid multiplying the CPU load.
