# MIT application check — 2026-10-05

The author authorized MIT for author-owned software. The standard LICENSE carries Copyright (c) 2026 Tae Ho Kim; scope exclusions are documented separately in README and LICENSE_STATUS.md. Data, manuscript, poster, third-party components and research-result files are not granted MIT rights by this change.

The existing root-file test first failed on the newly added LICENSE. Its expected root files were updated to README.md, .gitignore and LICENSE; no research code or scientific assertions changed.

Fresh checks: `python -B scripts/run_tests.py` discovered 244 tests, with 238 passes and 6 conditional private-fixture skips. `python -B scripts/verify_release.py` passed, checking all 136 research sources with no issues. CPython 3.12.14 and the existing isolated Windows environment were used. No private fixture suite, clean installation or full research run was performed.