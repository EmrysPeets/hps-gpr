# v4.9.13: completed study and remaining qualifications

The requested conditional 90% CLs calibration is complete for all 456 scope/mass points. All planned first sampling refinements and the three combined second attempts have finished; no worker or scheduling deferral remains. All 41 combined masses pass both methods' endpoint precision requirements. The fixed/profiled median observed-limit ratios are 1.446, 1.333, 1.199, and 1.346 for 2015, 2016, 2021, and their combination. These are observed comparisons, not expected-power estimates.

## Remaining finite-Monte-Carlo qualifications

| Scope | Method | Masses (MeV) |
|---|---|---|
| 2015, 100% | profiled | 22, 51 |
| 2016, 100% | profiled | 44, 45, 54, 58, 59, 71, 82, 83, 93 |
| 2016, 100% | fixed | 72 |
| 2021, 10% | fixed | 77 |

All of these endpoints are finite and remain plotted with their precision marker. Eleven fail the additional sampling-range/normalization-readiness check after passing the original endpoint gates. The profiled 2016 endpoints at 82 and 83 MeV fail an original MC gate while passing sampling readiness. The exact per-truth traces, checks and source checkpoint are in `summary/truth_specific_limits.csv` and the selected result JSONs.

If higher precision is required, use the frozen attempt-2 sampling policy on these preidentified individual coordinates, with fresh 1,024-draw proposal strata and a new output directory. Select from MC diagnostics only. Retain the original 47 inference hashes, truth/support/kernel choices, validation seeds and numerical gates; keep the ordinary 4 GiB memory bound and one BLAS thread per worker. Do not relax a failed gate or choose a method from its favorable observed limit. This extra precision campaign was not started, preserving the user's weekly reserve.

## Scope of any future physics claim

- The reported calibration is conditional on reviewed kernels. Historical hyperparameter and support selection were not repeated. A paired study would be needed before extending these results to that entire procedure.
- The generating family contains the mass-local GP truth and one archived stress truth per dataset. The combination tests the two joint all-GP/all-stress scenarios, not every mixed assignment. The archived stress shapes retain their documented fit limitations.
- The disclosed 2016 numerical exception and remaining common-systematic qualifications remain unresolved by this conditional calibration.
- Local p-values are not scan-wide significance. A global trials correction cannot substitute for a valid local sampling distribution. The 500-toy-per-cell validation screen has limited power and does not establish percent-level coverage.

## Rebuild and publication

Run `python3 -B study_results/v4p9p13_calibration_20260905/rebuild_products.py` from the canonical repository. It consumes the exact ordered `collection_inputs.json`, checks source and numerical identities, rebuilds all eight figure pairs, and typesets the 31-page note with the independent 71 MeV reverse-truth validation. Do not overwrite frozen runs. Native observed ROOT inputs remain external; canonical paths and environment identities are recorded in the README and provenance.

The 74 MeV scalar reference recovery and later metadata-only finalization are fully retained, including both failures, original bank identities, successful agreement checks, the exact saved numerical result, and the separately finalized derivative. No failed toy was dropped and no original numerical result was altered in finalization.
