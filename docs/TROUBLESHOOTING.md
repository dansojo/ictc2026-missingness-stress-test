# Execution boundaries and failure messages

Run documented commands from the release root with Python 3.12 and matching requirements. Keep stdout/stderr logs outside the publication tree because failures may include private paths.

| Symptom | Meaning and action |
|---|---|
| `Assertions must remain enabled` | Remove optimization flags and `PYTHONOPTIMIZE`; original scientific assertions are mandatory. |
| Missing role / input outside read roots | Complete the private JSON configuration; do not distribute machine paths. |
| Missing author protocol, M0, donor or reference artifacts | Provider data alone is insufficient for current launchers. Obtain authorized author artifacts; public synthetic tests still run. |
| Existing output / protected input overlap | Choose another private output outside the release and inputs; preserve failed runs. |
| Source/input/hash mismatch | Identify the differing version; do not edit frozen hashes or run a new freeze to force acceptance. |
| Two external source checks unavailable | The recorded missing files are UCSD ExtraSensory `cv5Folds.zip` and provider `README.txt`; see `EXTERNAL_PREREQUISITES.md`. |
| Six public tests skipped | Private fixtures are absent. The separate 6/6 pass does not make fixtures public or remove default skips. |
| Full comparison exits 2 | Strict comparison failed. Preserve this even when tolerant diagnostic checks pass. |
| Fresh lineage rejects changed source bytes | A sealed run binds its execution code version. Preserve the October 2 release ZIP and original manifests. Later code changes need a new run or separately reported scoped rechecks. |
| Poster asset rejected | Only approved exact PDF/PNG bytes in `docs/assets/posters.json` are allowed. |
| Registry error 10013 | Socket access was denied previously. No bypass was attempted; clean installation remains unverified. |

`scripts/portable_run.py check` validates only its printed scope and does not fit models. `scripts/smoke.py` performs bounded new calculations using archived inputs and writes aggregate reports. Stage runners and `scripts/fresh_chain.py` can write participant-derived records; their output trees stay private. Failures return a nonzero status; smoke reports retain an error type without the exception payload.

Full base execution, all external fits, bootstrap summaries and final paper figures remain outside current run authorization. A bounded success does not change that status.
