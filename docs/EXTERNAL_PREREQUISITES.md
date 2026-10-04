# Missing external inventory prerequisites

The author archive's `external_inventory/source_manifest.json` pins these two
files under `02_experiments/external_data/extrasensory/downloads/`. That directory
is absent from the copied hub. A read-only provider-input audit found 161 matching
files, two missing files and no hash mismatches across 163 checks.

| File | Expected bytes | SHA-256 |
|---|---:|---|
| `cv5Folds.zip` | 10,803 | `66183635452d8c1b250b570fba3000296bf29e4d54eda6af270d2b4f5a3b9ee6` |
| `README.txt` | 6,692 | `a5d222248abd0a2380b832be2bbec316812c7ca4b9c7831a6ae065b481ffa8c0` |

Local feasibility and source-manifest evidence identifies these as artifacts of
the UCSD ExtraSensory features/labels distribution. The
[official ExtraSensory project](https://extrasensory.ucsd.edu/) describes the
features/labels data and five-fold participant partition. The missing README is
the recorded provider README, not the locally written readiness report. The
original acquisition log and exact download endpoints are not available here;
this work did not authenticate replacement bytes or guess download URLs.

`audit_external_inventory.py` hashes the ZIP, opens its five-fold rosters and
hashes the README before writing final windows. Both are required to recreate
that inventory and its complete provenance. The later prediction runner instead
uses pinned prepared windows and leave-one-subject-out folds; it does not consume
the provider fold ZIP/README during fitting. Existing prepared-window checks
remain usable, but cannot be reported as a rebuilt provider inventory.

These two files are unrelated to the bounded ETRI raw-to-recovery chain. Restore
the original authorized copies with these exact hashes before attempting the
external inventory route. Any new acquisition needs separate authorization;
none was performed, skipped around or substituted here. Availability and data
redistribution rights still require provider terms to be checked by the author.

StudentLife is a separate source: the local readiness record says its available
copy came through a Kaggle redistribution, while Dartmouth is the scientific
dataset source. Do not describe that copy as a verified direct Dartmouth
download or attribute the two missing ExtraSensory files to StudentLife.

## Source evidence retained with this release

**Provider: University of California San Diego (UC San Diego / UCSD), ExtraSensory Dataset. Original source: the official ExtraSensory features-and-labels distribution and its participant cross-validation partition, https://extrasensory.ucsd.edu/.** This attribution concerns both missing files; neither belongs to StudentLife.

The archived `2026-09-11/external_inventory/source_manifest.json`, lines 303–310, records the two expected local filenames, byte sizes and SHA-256 values shown above. The same inventory's `FEASIBILITY_REPORT.md`, line 72, cites the official ExtraSensory site and states that its already-local official README was pinned in that manifest. The inventory program opens the ZIP rosters and hashes that README. These three records establish how the research used the files; they are not a new verification of absent file contents.

During this delivery pass, the official site's search result still identified the features-and-labels ZIP and its README, but a live page fetch returned HTTP 502. An authenticated original per-file download endpoint or acquisition log was not recovered. The project homepage is the verified source attribution; individual download URLs are deliberately not guessed. No data download was attempted.

Evidence-file SHA-256:

- `source_manifest.json`: `6098ed91175cbbf39651274db917b1920fde5d98f73e90ca1bba515106b3e710`

- `FEASIBILITY_REPORT.md`: `fa41fbddca3158c2d949229eec7ff24d2ee0a13fdaac4a0956382de063004a31`
