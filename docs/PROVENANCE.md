# Provenance and path adaptations

[The source ledger](provenance/SOURCE_MANIFEST.json) records 136 selected original files. 19 have documented current and original SHA-256 values. All 138 selected original files, including two excluded participant-linked metadata files, were rehashed unchanged in the read-only archive.

Adaptations separate code, private input roots and private output roots; map historical input paths without changing frozen bytes; connect private comparison fixtures; and validate actual consumed donor/provider files. Source freeze verification checks the old source bytes and reviewed packaged bytes independently. The two excluded readiness metadata files retain their mandatory input pins. Configurations and numerical algorithms/seeds are unchanged.

`docs/path-adaptation-audit.json` lists every changed file and top-level function. 1561 original top-level function ASTs are unchanged. Changed functions are path, input/output, provenance and fixture orchestration; the named calculation, masking, repair, donor-extraction, feature, fitting, prediction and canonicalization functions were checked unchanged. This audit is supporting evidence, not a substitute for full numerical reproduction.

Ten upstream builders were selected: all five donor scripts and five external inventory scripts. The superseded `verify_inventory_attempt01.py` was excluded. The historical parallel donor continuation script is included for provenance; the serial builder is the fresh launcher route.

New adapters (`scripts/_support/portable_paths.py`, `scripts/portable_run.py`, `private_reference.py`, `scripts/smoke.py`, release tests and auditors) are later release work, not historical preregistration. No original protocol was rewritten. Changes found during review included separating code/output directories, binding checks to consumed inputs, protecting configured roots, resolving old/fresh source gates, and disabling Python optimization that would remove original assert-based checks. Failed development runs are retained privately.

Current success includes a separately sealed bounded raw-to-recovery chain. `scripts/fresh_chain.py`, `scripts/_support/fresh_lineage.py` and `scripts/_support/bounded_upstream.py` are later release orchestration. Original frozen manifests remain distinct from the new run contract and stage parents. The generated active path configuration is fixed before stage execution and checked before/after each operation. Independent review found and resolved its initial missing hash binding.

The current audit counts 1561 unchanged original top-level function ASTs and 19 adapted source files; see the JSON audit for changed orchestration functions. Strict historical full-comparison failure remains false/exit 2. No tolerance, mask, feature, seed or model was changed to obtain a pass. No source data or per-person results belong in the release. The full study remains on hold.


## Author confirmation — 2026-10-05

Tae Ho Kim confirms that the original research code was authored by him and transferred when the previous computer was returned. vendor, full_vendor and g2_vendor are historical research snapshots, not evidence of third-party ownership. The source review did not identify incorporated third-party code. Dependencies are installed separately and retain their licenses. Dataset and publication-material rights remain separate.
