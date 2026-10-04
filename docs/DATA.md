# Inputs and privacy

Obtain data from its providers under the applicable terms. This repository does not download data or grant access/redistribution rights. The author's local copies were found; permission to redistribute them was not established. Official entry points are [ETRI's paper-cited competition data page](https://dacon.io/competitions/official/236690/data), [StudentLife](https://studentlife.cs.dartmouth.edu/dataset.html), and [ExtraSensory](https://extrasensory.ucsd.edu/). Check the provider's current access, permitted uses, citation and redistribution terms before obtaining or sharing data. Provider pages timed out during the October 2 follow-up, so no current permission or license is inferred. No data was downloaded or access requested.

Base input layout:

```text
<data>/raw/ch2026_metrics_train.csv
<data>/raw/ch2026_submission_sample.csv
<data>/raw/ch2025_data_items/ch2025_<sensor>.parquet
<data>/external/StudentLife/dataset/...
<data>/external/ExtraSensory.per_uuid_features_labels/*.features_labels.csv.gz
<data>/external/02_experiments/external_data/readiness/...
```

The 12 sensor names are mScreenStatus, mUsageStats, wPedo, mActivity, wHr, mLight, wLight, mACStatus, mGps, mWifi, mBle and mAmbience. Use the pinned provider versions. Similar versions may fail hashes/schema validation. ExtraSensory processing starts from provider-distributed feature CSVs, not raw accelerometer waveforms.

Readiness requires `verify_external_data.py`, `prepare_external_data.py`, `dataset_locations.json`, `schema_manifest.json` and `file_manifest.csv`. The last two contain provider participant identifiers and stay with authorized private inputs. Their original hashes remain in `paper/input_manifest.json`, and external execution still verifies them. They are not bundled as public source fixtures.

The external inventory builder additionally reads `<external-provider>/02_experiments/external_data/extrasensory/downloads/cv5Folds.zip` and `README.txt`. Both were absent from the authorized local archive. Code inspection shows that the builder unconditionally opens the ZIP, checks the five-fold participant partitions and hashes the README. They are **execution-required for inventory and complete source-provenance verification**, although the final LOSO fitter does not use those folds. Existing prepared-window prediction checks can run without them. Obtain the exact provider versions only through the official provider process after access is authorized, then compare them to the existing private source-manifest pins. No URL or replacement README is guessed, and neither pin is dropped.

For this session's bounded checks, `--hub` expects these original archive-relative paths:

```text
90_ARCHIVE/output/ictc2026-final-preparation/2026-09-10/reviewer_extension/full_m0_01/
90_ARCHIVE/output/ictc2026-final-preparation/2026-09-11/imputation_extension/
90_ARCHIVE/output/ictc2026-final-preparation/2026-09-11/crossday_imputation_v1/
90_ARCHIVE/output/ictc2026-final-preparation/2026-09-11/external_inventory/
90_ARCHIVE/output/ictc2026-final-preparation/2026-09-11/external_prediction/
```

The adapter reads cell/mask/reference inputs, historical manifests and expected outputs directly, with hash checks; it does not copy them. Donor caches include raw values, and prepared external windows include labels, so neither belongs in a public repository. JSON reports contain only aggregate check counts, differences, timing and error types. Console tracebacks and redirected diagnostic logs can contain identifying input details on failure and must remain private.

Observed local sizes: ETRI 15 files / 128,912,754 bytes; StudentLife 1,984 files / 3,042,012,275 bytes; ExtraSensory 60 files / 225,325,871 bytes. The historical base full-run directory occupies 2,078,899,033 bytes, excluding separately stored inputs and other runs. These are inventory measurements, not disk requirements guaranteed for every platform.

Excluded assets include per-person CSVs, sensor observations, donor pickles, prepared windows, predictions, trained model states, screenshots/emails, author administrative records, local root maps, virtual environments and caches. The preserved experiment config contains the published pseudonymous cohort roster only, without per-person observations or outcomes; removing it would change the fixed scientific configuration.

## What a new reader can run

The synthetic public suite needs only code and dependencies. The current `scripts/smoke.py`, archived portable routes and bounded `scripts/fresh_chain.py` also need trusted author artifacts: original frozen protocols, source provenance, saved upstream manifests and private references. Several are excluded because they contain participant-linked records or private paths. Provider downloads do not supply these artifacts.

This candidate exposes tested research code and bounded author-archive validation, not a standalone provider-only route. Before describing it as independently reproducible for a new reader, decide how necessary artifacts can be provided under applicable rights or implement and separately validate a provider-only provenance route. Do not publish private archives to fill this gap.
