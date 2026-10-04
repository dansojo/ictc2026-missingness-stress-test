# Conditional private-fixture checks

The default public suite still skips six tests when private fixtures are not configured. All six were run separately on 2026-10-02 with authorized read-only fixture roots: **6 passed, 0 failed, 0 skipped**. Nothing was bundled into the public release to make them pass.

| Test | Private fixture and observed result |
|---|---|
| Downstream source adapter semantics/deletion guard | Original synthetic fixture source; exact contexts, patches, tables and deliberate deletion-count fault rejection passed |
| Eight evidence CSV formulas | Archived diagnostic/downstream tables; all eight rendered CSV calculations matched references exactly |
| Result prose/design responds to changed values | Same tables modified only in memory; generated prose/count assertions passed |
| Fixed display annotations | Private condition summary; expected counts passed and changed abstention count was rejected |
| Display adapter | Temporary private comparison-fixture rendering; four output assets and PNG pixel equality passed |
| Small stress replay | Existing canonical/raw-derived primitive fixture; archived target and two freshly generated masks passed |

Configuration: `ICTC_PRIVATE_REFERENCE`, `ICTC_TEST_ROOTS` (JSON with etri_sdd/etri_source), and `FULL_DOWNSTREAM_ORIGINAL_TEST_ROOT`. Keep all three outside the release. The display fixture is a test of the older base renderer, not proof that submitted Figure 1's inline TikZ has been regenerated. Temporary rendering/test files remained private. These fixture tests do not run the full downstream study.
