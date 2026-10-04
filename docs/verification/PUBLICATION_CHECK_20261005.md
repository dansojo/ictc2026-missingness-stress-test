# Publication check — 2026-10-05 (Asia/Seoul)

The public test command `python -B scripts/run_tests.py` was rerun using CPython 3.12.14 and the existing isolated environment with preinstalled dependencies. 244 tests were discovered: 238 passed, 6 private-fixture checks were skipped. The separate private-fixture suite was not rerun; its 6 passes remain historical October 4 evidence. This is not a clean-install test or a full study rerun.

The static distribution check `python -B scripts/verify_release.py` was rerun for publication. It checks source hashes, syntax and selected-content exclusions; it does not establish legal rights or full scientific reproduction.

The author confirms direct authorship of the research code. No software license has been selected. Final poster files await the repository QR/link update. Long-running experiments were not launched.
The initial post-edit test attempt found one missing README link target because this record had not yet been created. After creating it, all eight public groups passed (38, 88, 23, 9, 21, 42, 6, 17 tests; six conditional skips). No research code or test assertion was changed.
