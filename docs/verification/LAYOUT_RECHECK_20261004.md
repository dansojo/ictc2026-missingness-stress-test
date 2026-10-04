# Layout recheck: 4 October 2026

The organized candidate has two root files, `README.md` and `.gitignore`, and five root directories: `docs/`, `extensions/`, `original_paper_reproduction/`, `scripts/`, and `tests/`. Five public entrypoints now live in `scripts/`; four helper modules live in `scripts/_support/`. Arguments are unchanged; old root command paths require the new `scripts/` prefix. See [migration notes](../LAYOUT.md).

## Preservation and validation

The preserved previous candidate's 177 files still match its baseline hashes. Both research trees, all 136 selected research-source entries, and the source ledger bytes remain unchanged. All 138 selected files in the read-only original archive were rehashed unchanged, including the two excluded participant-linked metadata files. No raw or participant-level outputs were copied into the public candidate.

| Check actually run after relocation | Result |
|---|---|
| Public unittest suite | 244 discovered; 238 passed; 6 conditional private-fixture skips; no failures |
| Private fixtures, separately enabled | 6 passed; these tests reuse archived fixtures |
| New layout regression tests | 5 passed; failures were observed before the relocation |
| Absolute CLI paths from an unrelated directory containing spaces | Help/import checks passed with inherited `PYTHONPATH` removed |
| Base, M0, legacy, same-day, cross-day, donor, prediction and canonicalization input gates | All eight passed |
| External inventory input gate | Blocked as expected: two original ExtraSensory provenance files are absent |
| Same-day reconstruction, one primitive cell | 1,000 rows; 2,000/2,000 scalar comparisons exactly equal; max difference 0 |
| Other-day reconstruction, one primitive cell | 800 rows; 1,600/1,600 scalar comparisons exactly equal; max difference 0 |
| External prediction, one held-out participant per dataset | 2 new fits; 74 windows; 7,474 comparisons; 4,484 exactly equal; max difference 2.220446049250313e-16 |
| Canonicalization via relocated launcher and child process | Completed; synthetic duplicate check passed |
| Static candidate audit | Passed; 136 source entries checked; no reported inclusion, syntax, secret-pattern or personal-path issues |
| Independent read-only review | No Critical or Important findings; four minor navigation/path descriptions corrected |

The first private-fixture invocation used the wrong archived test-source root: five tests passed and one import failed. Correcting that private configuration produced 6/6 passes. The failed log is retained privately; no source or test assertion was changed to get the pass. The external inventory gate remains a real blocker, not a successful experiment.

These are fresh bounded computations from existing authorized archived inputs, with comparisons to archived references. Same-day, other-day and prediction runs checked 5, 8 and 5 input files unchanged, respectively. The public suite uses synthetic/public fixtures; the separate private suite is reference-fixture validation, not a new population experiment. The five new layout tests account for the increase from the previous 239-test suite.

## Scope that remains unverified

The approximately 11-hour full study and nine-stage raw chain were not rerun for this layout change. Historical evidence remains in [VERIFICATION.md](VERIFICATION.md), [the October 3 recheck](../RECHECK_20261003.md), and [structured results](verification.json). The 30-value [paper comparison](paper-targets.csv) is an archived-aggregate audit, not a new paper-wide reproduction.

The historical strict Brier comparison remains false/exit 2. The earlier fresh preparation all-column exact comparison remains false, with maximum difference 1.7763568394002505e-15. No tolerances, masks, seeds, feature definitions, frozen hashes or fitting rules were changed.

Validation used CPython 3.12.14 and matching preinstalled dependencies through the isolated interpreter's `.pth` path. A clean online dependency installation and Linux execution remain unverified. Public launchers still require authorized author archives, frozen protocols and private references in addition to provider data.

Latest poster PDFs are version 1 according to the parent handoff; those bytes and the preview PNGs are not present locally. Earlier failed transfers refer to their own earlier versions. [Poster status](../assets/README.md) remains pending, and the ZIP is explicitly an incomplete candidate without posters. Repository name, visibility, code license and public contact details are undecided. No GitHub creation, push or publication occurred.

## Commands

From this candidate's root:

```text
python -B scripts/run_tests.py
python -B scripts/verify_release.py
python -B scripts/smoke.py imputation_extension --hub "<authorized-archive>" --limit 1 --output "<new-private-directory>"
python -B scripts/smoke.py crossday_imputation_v1 --hub "<authorized-archive>" --limit 1 --output "<different-new-private-directory>"
python -B scripts/smoke.py external_prediction --hub "<authorized-archive>" --limit 1 --output "<third-new-private-directory>"
python -B scripts/portable_run.py run canonicalize --config "<private-paths.json>" --output "<fourth-new-private-directory>"
```

Use [private-fixture instructions](../SKIPPED_TESTS.md) for the six additional tests. Keep all inputs, configuration and run outputs outside the public tree. [SHA256SUMS.txt](../provenance/SHA256SUMS.txt) lists paths relative to the repository root and omits its own checksum.
