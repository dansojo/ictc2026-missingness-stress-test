# Bounded fresh raw-to-recovery chain

`scripts/fresh_chain.py` is a separate later execution adapter. It preserves the original
historical freeze route and verifies those original files independently. It does
not rewrite a protocol or use new hashes in place of old frozen hashes.

Use the private configuration described in `PORTABLE_STAGES.md`. The `raw` role
must point to an already authorized complete ETRI raw root. Historical roles are
still required to validate source and protocol provenance. No data is downloaded.

```text
python -B scripts/fresh_chain.py --config "<private-paths.json>" --output "<private-output>/fresh-chain-01" --cells 5
```

The output directory must be new, outside all configured inputs and this release.
It contains participant records and must remain private. The command refuses
Python optimization and limits execution to 1–5 target cells. It has no full-study
or resume mode. Preserve any failed run and choose another output directory.

The execution order is raw → prepare → G2 → stress → M0 → legacy → donors →
same-day → cross-day. Raw processing calls the original twelve sensor
canonicalizers. Preparation retains all 853 participant-days. Original selection
and standardizers use all 800 selected G2 cells and the original full-population
LOSO baselines. After that, the driver selects the first participant/day for each
of five recovery primitives, preserving all 50 draws and root seed 42. It builds
all calendar dates in each selected participant/primitive donor partition; only
the target date and overlapping support are excluded by the original recovery
code. Calculation, seed serialization, mask generation and repair functions are
the original implementations.

G2's unmasked canonical replay parity is checked for the executed target cells.
The full 800-cell standardizer values come from freshly extracted primitives;
this bounded adapter does not replay all 800 unmasked cells. It also omits the
base study's G1 analysis, calibration/modality sweeps, prediction fits, bootstrap
summaries and final paper displays. It is therefore explicitly a **bounded fresh
chain**, not a complete paper reproduction.

`run-contract.json` pins raw files, execution sources and historical provenance.
Each stage's `lineage.json` hashes every output and the exact parent manifests,
binds target cell identities, and records `full_reproduction=false`. The generated
execution path configuration is pinned and checked before and after every stage.
Old archive freeze checks and actual fresh input checks both remain active.
Changed data, parents, non-target donor bytes, seeds, scope, added outputs or
crosswired configurations are rejected.

Fresh calculations do not read archived canonical values, selected-day outcomes,
masks, references, scales or donor caches as numerical inputs. Historical tables
can be read afterward in an independent comparison pass. Current measured
results are in `VERIFICATION.md` and `verification.json`.
