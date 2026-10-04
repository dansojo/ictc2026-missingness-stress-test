# Layout and command migration

Prepared in a separate candidate; the previous release remains intact. The research trees retain their names, relative paths and bytes. All 136 selected source entries and the ledger bytes are unchanged.

## Public commands

The five public CLI names and arguments are unchanged, now under `scripts/`:

| Previous | Current |
|---|---|
| `python -B run_tests.py` | `python -B scripts/run_tests.py` |
| `python -B verify_release.py` | `python -B scripts/verify_release.py` |
| `python -B smoke.py ...` | `python -B scripts/smoke.py ...` |
| `python -B portable_run.py ...` | `python -B scripts/portable_run.py ...` |
| `python -B fresh_chain.py ...` | `python -B scripts/fresh_chain.py ...` |

Existing shortcuts must adopt the scripts/ prefix. No root wrappers are retained. Absolute script paths work from another CWD; relative config/output arguments remain relative to the caller's CWD.

Launchers explicitly add `scripts/_support/`; the portable runner exports it to child processes so original adapters keep their flat imports. Direct internal imports require the caller to supply the appropriate Python path and are not a separate public CLI.

Private configuration stays outside the repository. Update `source_roots[].code` and code-related `read_roots` for this checkout; original source/data roots remain read-only.

## File locations

- `scripts/_support/`: bounded upstream, fresh lineage, portable paths and release support.
- `tests/`: packaging/scientific-contract and relocation regression checks.
- `docs/verification/`: dated results, structured verification, paper comparison and release status.
- `docs/provenance/SOURCE_MANIFEST.json`: unchanged research source ledger.
- `docs/provenance/SHA256SUMS.txt`: release-root-relative checksums, excluding itself.
- `docs/assets/`: poster status/manifest; actual PDF/PNG transfer remains incomplete.

Root files are README.md and existing .gitignore only. No license is invented; the existing nested requirements file remains the installation source.

