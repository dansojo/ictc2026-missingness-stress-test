# Verification — 2 October 2026

**The nine-stage bounded raw-to-recovery run completed. Its recovery results match exactly; tiny heart-rate preparation differences keep the all-column exact comparison false. Full reproduction remains pending. No GitHub publication occurred.** Current measured results are in `verification.json`.

## Actual execution

| Check | Result | Scope |
|---|---|---|
| Public suite | 234 discovered; 228 passed; 6 conditional skips; 0 failed | Eight subprocess groups; synthetic/unit checks |
| Previously skipped fixtures | 6/6 passed separately | Private source, schema, display and stress fixtures; no full new downstream fits |
| New bounded raw chain | 9 stages; 5 cells; 50 draws; 504.9 seconds | 12 raw sensors; full853-day preparation; full800-cell selection/scales; new donor partitions; exact recovery rows; preparation drift reported below |
| Portable M0 -> legacy | 1 cell; 100 M0 mask rows; 300 legacy rows | M0 newly computed from saved upstream tables, then consumed by legacy; exact public checks |
| Portable same-day / cross-day | 1,000 / 800 rows; 300 legacy / 100 baseline rows exact | Original frozen inputs, 1 cell each; public max difference 0 |
| External canonicalization | 6,211 windows; output SHA identical to frozen prediction input | Recomputed from archived prepared windows; no provider inventory build or model fit |
| Earlier broader recovery smoke | 5,000 / 4,000 rows; 10,000 / 8,000 scalars exact | Five original primitive cells; archived masks/donors |
| Post-adaptation smoke regression | Same-day/cross-day/external all pass | 1,000 / 800 recovery rows; 2 external fits |
| External probabilities | 7,474 checks; 4,484 exact; max difference 2.220446049250313e-16 | 74 held-out windows, 3,700 paired masks; unchanged rtol=atol=1e-12 |
| Archived-paper comparison | 30/30 selected values at printed precision | Existing population aggregates, not newly regenerated medians |

The 6,211-window regenerated file has SHA-256 `a4f74d30debf58bec47fa3906d14977efd834f08de8dfe3a5e8205090b843c7f`. Same-day/cross-day runs verify their actual consumed inputs; original frozen files remain unchanged. New outputs and diagnostic logs are private. A fixture render passing pixel equality concerns the old base `fig2_results`, not the final manuscript's inline Figure 1.

## Environment

Execution used CPython 3.12.14 in an isolated venv with matching already-installed historical libraries attached through a local `.pth`. All 11 direct pins matched/imported. This is **not a successful clean installation**. A second fresh environment had only pip after official PyPI HTTPS failed with Winsock access-denied error 10013 and pip exited 1. Package availability was not established by that failure. The exact enforcing component is unknown; no network/security settings, privileges or transports were changed. Linux remains unverified.

## Comparison with the submitted paper

The local final PDF is 265,341 bytes, four pages, SHA-256 `2e05756b27f7d54ea2cbb162550d4d2a9371c764aac7f479231b581d8009320f`. Its corresponding final manuscript was read. The table below is an **archived result comparison**, not a claim that the present bounded sample regenerated population medians.

| Metric | Final paper | Archived aggregate | Current full rerun |
|---|---:|---:|---|
| ETRI state-ratio contrast | 0.463 | 0.463349512 | Not run |
| ETRI intensity contrast | 0.376 | 0.376368552 | Not run |
| StudentLife state / intensity | 0.290 / 0.305 | 0.2904673993 / 0.3050670624 | Not run |
| ExtraSensory state / intensity | 0.091 / 0.184 | 0.0912795452 / 0.1838787885 | Not run |
| Linear recovery: mobile / wearable light | 0.193 / 0.306 | 0.1928329681 / 0.3060605868 | Not run |
| Other-day same-time median: mobile / wearable | 0.437 / 0.528 | 0.4365153372 / 0.5283995385 | Not run |
| External observed AUROC: StudentLife / ExtraSensory | 0.448 / 0.854 | 0.4484126984 / 0.8538411458 | Not run |
| ExtraSensory block-minus-random log loss | 0.00142 | 0.001416507889 | Not run |
| Same-day / other-day scheduled evaluations | 500,000 / 400,000 | 500,000 / 400,000 | Only 5,000 / 4,000 recomputed |

`paper-targets.csv` also checks finite-pair denominators, confidence bounds, downstream drift, prior-relative log loss and external cohort/window counts. `docs/paper-comparison-provenance.json` records source hashes. This is a selected 30-value audit; it is not exhaustive verification of every table cell, CI, figure, or manuscript statement.

## Historical strict comparison remains failed

Saved base-run evidence records 160 fits and 943,392 prediction rows, with byte-identical predictions against its historical reference, but tiny Brier aggregate CSV differences. The three recorded maxima are 8.326672684688674e-17, 3.7947076036992655e-19 and 5.421010862427522e-20. The underlying cause is not established by this work.

`strict_full_comparison_passed=false` and the full-comparison exit code **2** are preserved. The bounded adapter's tolerant probability checks do not override that strict failure. Historical 89-value display verification covers the older base display scope, not all final Table III/IV extensions or the final inline Figure 1.

## Source integrity and current limits

All 138 selected original source files were rehashed unchanged. Two private readiness metadata files are excluded; 136 originals are in the public source ledger, including 10 selected builders. Nineteen files have explicit original/current hashes for path, provenance, packaging and test adaptations. An AST comparison found 1,561 unchanged top-level functions; scientific calculation/mask/repair/extraction/fit/prediction/canonicalization functions remain unchanged. See `docs/path-adaptation-audit.json` and `docs/PROVENANCE.md`.

The public scan passed with no prohibited data/model/archive files, unreviewed CSVs, credential-pattern matches, personal home paths or Python parse errors. This static scan is not a license grant or exhaustive proof of secret absence. No provider observations, donor caches, prepared windows, participant predictions, trained states or final PDF are public deliverables. The fixed config retains the paper's pseudonymous cohort roster.

Full donor rebuilding/finalization, raw external inventory, all 102 external LOSO fits, full base execution and exhaustive final-paper numerical/figure regeneration remain unrun. Missing `cv5Folds.zip` and `README.txt` prevent complete external inventory/source-provenance execution. The final LOSO fitter does not consume these folds, but the inventory does.

Portable archived-input execution and a separate bounded fresh lineage route are connected. Original full-manifest gates remain intact; the fresh route verifies original provenance separately from its new exact parent manifests. Full-study execution stays on hold. See `docs/FRESH_CHAIN.md`, `docs/PORTABLE_STAGES.md` and `docs/FULL_RUN_PLAN.md`. Repository name, visibility and license remain pending.


## Bounded raw-chain follow-up (2026-10-02)

The new raw-to-recovery chain completed all stage gates in 504.9 seconds. All nine stage manifests are separately sealed; this is explicitly bounded, not a full paper reproduction.

- Public suite: **234 discovered, 228 passed, 6 default conditional skips, 0 failed**. The six private fixtures separately passed **6/6**, with zero skips.
- Fresh processing used all 12 raw sensor files, all 853 participant-days, all 800 G2 selection/normalization cells, five target cells and 50 draws. The original code, seed 42, mask rules, scales and full within-participant donor dates were preserved.
- The fresh-versus-archive exact comparison is recorded separately from stage execution success.

- The generated runtime path configuration, raw files, source bytes, historical provenance, exact parent manifests and every stage output are hash-bound. Ten lineage regression tests include changed raw/parent/output/seed/scope, foreign cells, non-target donor bytes and swapped active configurations.
- Original full M0/legacy/cross-day freeze trees were checked independently. They were not rewritten. Historical outcome tables were used only in the later independent comparison pass.

| Fresh result compared afterward with archive | Scope | Observed result |
|---|---|---|
| raw canonical files | files=12 | All fields/values exact; byte-identical=12 |
| prepare primitives | rows=853 | EXACT COMPARISON FAILED; see quantified differences below; byte-identical=False |
| prepare baselines | rows=1920 | EXACT COMPARISON FAILED; see quantified differences below; byte-identical=False |
| prepare representations | rows=7716 | EXACT COMPARISON FAILED; see quantified differences below; byte-identical=False |
| g2 selected_days | rows=5 | All fields/values exact |
| g2 cell_replays | rows=250 | All fields/values exact; storage dtype differs: donor_sensor_day_id |
| g2 mask_intervals | rows=250 | All fields/values exact; storage dtype differs: donor_sensor_day_id |
| g2 deletion_audit | rows=250 | All fields/values exact; storage dtype differs: donor_sensor_day_id |
| stress cell_replays | rows=500 | All fields/values exact |
| stress mask_ledger | rows=500 | All fields/values exact |
| stress target_ledger | rows=250 | All fields/values exact |
| M0 cells | cells=5 | All fields/values exact |
| donor partitions | partitions=5 | All fields/values exact |
| legacy recovery | rows=1500 | All fields/values exact |
| same-day recovery | rows=5000 | All fields/values exact |
| cross-day recovery | rows=4000 | All fields/values exact |

- `prepare primitives` / `heart_rate_settling_delta__median_slope_per_hour`: 10 values differ; maximum absolute difference 1.7763568394002505e-15.

- `prepare primitives` / `heart_rate_settling_delta__heart_rate_settling_slope`: 10 values differ; maximum absolute difference 1.7763568394002505e-15.

- `prepare primitives` / `heart_rate_settling_slope`: 10 values differ; maximum absolute difference 1.7763568394002505e-15.

- `prepare primitives` / `heart_rate_settling_slope__median_slope_per_hour`: 10 values differ; maximum absolute difference 1.7763568394002505e-15.

- `prepare primitives` / `heart_rate_settling_slope__heart_rate_settling_slope`: 10 values differ; maximum absolute difference 1.7763568394002505e-15.

- `prepare baselines` / `center`: 18 values differ; maximum absolute difference 4.440892098500626e-16.

- `prepare baselines` / `scale`: 4 values differ; maximum absolute difference 4.440892098500626e-16.

- `prepare baselines` / `calibration_center`: 22 values differ; maximum absolute difference 4.440892098500626e-16.

- `prepare baselines` / `calibration_scale`: 11 values differ; maximum absolute difference 4.440892098500626e-16.

- `prepare representations` / `heart_rate_settling_slope`: 72 values differ; maximum absolute difference 1.7763568394002505e-15.

- `prepare representations` / `heart_rate_settling_slope__center`: 1146 values differ; maximum absolute difference 4.440892098500626e-16.

- `prepare representations` / `heart_rate_settling_slope__scale`: 256 values differ; maximum absolute difference 4.440892098500626e-16.

- `prepare representations` / `heart_rate_settling_slope__z`: 159 values differ; maximum absolute difference 8.881784197001252e-16.

- `prepare representations` / `physiological_settling_score`: 106 values differ; maximum absolute difference 8.881784197001252e-16.

- `prepare representations` / `routine_deviation`: 8 values differ; maximum absolute difference 2.220446049250313e-16.

Additional strict checks found exact equality for all 800 selected cells, all 80 G2 standardizers, all 160 population k=0 baseline rows, the five recovery primitives' 853 × 67 value/quality fields and their 600 associated baseline rows. The preparation differences are localized to heart-rate settling slope and its k=14/physiology/routine descendants. Aggregate evidence is in `docs/fresh-prepare-differences.json`.

The fresh-versus-archive all-column exact comparison is **false**. These differences are reported without relaxing tolerance or modifying calculations. The source of the floating-point variation is not established by this comparison.

The submitted-paper comparison still consists of 30 archived values matching printed precision; no full new paper table/figure comparison is claimed. The earlier historical strict downstream comparison remains **false / exit 2**, including its tiny Brier CSV differences. No tolerance was relaxed.

Unexecuted: complete study, all 50 donor partitions, all 102 external model fits, full bootstrap summaries and final manuscript TikZ displays. Clean registry installation and Linux execution remain unverified. The two missing ExtraSensory inventory inputs are documented in `docs/EXTERNAL_PREREQUISITES.md`; they do not block this ETRI chain.

Earlier follow-up sections record previous snapshots. This section and `verification.json` contain the current scope and counts. Failed development runs remain private. No provider data was downloaded; no GitHub repository, push or publication occurred.


## Missing-file provider and original source

Both `cv5Folds.zip` (10,803 bytes) and `README.txt` (6,692 bytes) are recorded as **UC San Diego / UCSD ExtraSensory Dataset** distribution artifacts. The original source is the [official ExtraSensory project and features/labels distribution](https://extrasensory.ucsd.edu/), including its participant cross-validation partition. They are not StudentLife files.

Evidence: archived `external_inventory/source_manifest.json` lines 303–310 pin both filenames, sizes and SHA-256 values; `FEASIBILITY_REPORT.md` line 72 cites the official dataset documentation and describes its already-local official README as pinned. `audit_external_inventory.py` reads the ZIP's five fold rosters and hashes the README. Exact expected hashes and evidence-file hashes are in `docs/EXTERNAL_PREREQUISITES.md`.

The current official search result confirms a features/labels download and README; direct page retrieval returned HTTP 502. The acquisition log and exact per-file download endpoints remain unverified. No replacement URL or file was invented, and nothing was downloaded. These files block the external inventory rebuild, not the successfully executed bounded ETRI chain.

The new whole-table exact comparison remains **false**: heart-rate slope and derived preparation fields differ by at most **1.7763568394002505e-15**. No numerical acceptance threshold was relaxed.


## October 3 publication recheck

Fresh recheck: 239 public tests discovered, 233 passed, 6 conditional skips; private fixtures 6/6 passed again. All three bounded extra-experiment reruns passed their original comparisons. External inventory remains blocked by the two missing files. See [the detailed recheck](../RECHECK_20261003.md). Final poster assets and the final upload ZIP are pending. Earlier sections describe their dated snapshots.
