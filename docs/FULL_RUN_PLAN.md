# Full-run readiness: hold remains

The full study was not launched. See `PORTABLE_STAGES.md` for working private configuration/commands and `FRESH_CHAIN.md` for the separately bounded fresh route.

| Boundary | Current evidence |
|---|---|
| Original source files | 138 rehashed unchanged; 136 selected public originals |
| Saved experiment inputs | Earlier 1,139 unique files / 1,679,172,453 bytes audited; located files matched pins |
| Portable archived-input routes | M0, same-day and cross-day bounded execution passed; all prepared prediction-window contracts passed |
| New bounded handoff | New 1-cell M0 -> legacy produced 300 rows and four exact public checks |
| External canonicalization | 6,211 windows regenerated from archived prepared windows; byte-identical pinned output |
| Donor full inventory/finalization | Source selected, imports/upstream checks pass; full build not run |
| External provider inventory | Blocked: missing cv5Folds.zip and README.txt are mandatory provenance inputs |
| Bounded ETRI raw-to-recovery chain | Newly executed: five cells, all 50 draws, nine sealed stages; exact comparison scope/differences in VERIFICATION.md |
| Entire raw-to-paper chain | Full study unexecuted and on hold |
| Fresh package installation | Not verified: official PyPI socket access failed with WinError 10013 |

The inventory ZIP/README are auxiliary to the final LOSO model but mandatory in the inventory code. The ZIP is unconditionally opened to validate provider folds and the README is pinned. Their absence must not be described as harmless for rebuilding inputs. Existing prepared-window model checks remain usable. No download or permission request was made.

Winsock error 10013 is WSAEACCES: socket access was denied. This agrees with both recorded direct HTTPS attempts and pip's connection failure. It does not show that the pinned package versions are absent. The exact enforcing component (sandbox, endpoint policy or another system component) was not identified. No network/security setting was changed, no administrator elevation was requested, and no alternate transport bypass was attempted. See [Microsoft's Winsock error reference](https://learn.microsoft.com/en-us/windows/win32/winsock/windows-sockets-error-codes-2). Numerical checks used an isolated interpreter with already-installed matching historical libraries; that is distinct from a clean dependency install.

Completed follow-up: the bounded fresh lineage bridge now verifies original historical provenance and fresh actual parents independently. Five cells completed all stage gates in the raw → M0 → legacy → same-day/cross-day chain, with full-population selection/scales and complete selected donor partitions. Ten corruption/configuration tests and independent review support that scope. See `VERIFICATION.md` for exact downstream comparisons and any quantified preparation-table differences. This does not establish a full study. Base full runs, full 50-partition donor rebuild, all 102 external fits and all paper summaries remain unexecuted in this session. Do not create a new freeze or relabel old manifests to connect them.

Once full-run execution is explicitly resumed, use a new private run root and preserve failed attempts. Base `--resume` only reuses completed stages after their manifests/hashes validate; it does not continue within a partially written stage. Other experiment runners have no resume flag. Stop normally, wait for children to exit and never remove a live process lock. Retain strict-comparison false/exit 2 if exact comparisons fail; tolerant diagnostics are separate.

Historical base work sums to roughly 11 hours. Additional recorded M0/legacy/same-day/donor/cross-day/prediction stages sum to 41.9 minutes, excluding inventory, summaries/bootstrap, audits and overhead. These are historical timings, not predictions. The current machine previously reported 16 logical CPUs, 30.9 GiB RAM (13.6 GiB available) and about 702 GiB workspace disk free. Full-study peak RAM is unknown; historical preprocessing peak was 4.16 GB. Keep original concurrency/seeds unless a separately reviewed change is requested.
