# HPS-GPR v4.9.7: stopped 2016 support qualification

> **FAILED QUALIFICATION / STOPPED BY FROZEN RULE**
>
> Canonical state: `study_stopped_no_provisional_edge`.
>
> There is **no v4.9.7 observed 2016 result and no v4.9.7 combined result**.

This release records a scientifically useful negative result. It attempted to
choose a lower GP-support edge for the 2016 full-statistics data with a
predeclared, pull-blind injection-recovery test. No eligible edge passed the
frozen Phase-1 practical gate, so the protocol required the study to stop
without retuning. Phase 2, support freezing, the 65 MeV holdout, the
support-specific observed 2016 scan, and the requested 2015/2016/2021
combination were not run.

## What was attempted

The support study used the exact dataset key `2016`, upper support edge
210 MeV, and seven lower-edge candidates:

`028_210`, `029_210`, `030_210`, `031_210`, `032_210`, `033_210`, and
`034_210`.

Edges 28--33 MeV were eligible to freeze; 34 MeV was a geometry control. The
selection grid comprised four masses (44, 49, 54, and 59 MeV), three injected
strengths (0, 2, and 5 sigma), and 25 paired background toys per cell in
Phase 1. The 65 MeV mass was excluded as a post-freeze mechanism holdout.

An edge had to satisfy all frozen technical checks and all three practical
requirements:

- at least 9 of 12 cell means with `abs(mean pull) < 0.75`;
- at least 3 of 4 zero-signal cell means with `abs(mean pull) < 0.75`; and
- worst `abs(mean pull) < 1.25`.

Optimizer branches were chosen only by reproducible maximum GP log marginal
likelihood, with covariance and bound checks. Fitted signal amplitude,
recovery, epsilon-squared, p-value, and upper-limit strength were forbidden
from branch or support selection.

## Outcome

The expected Phase-1 grid contained 2,100 rows; 2,098 were accepted. Supports
30 and 32 MeV each lost one retained row and therefore also failed the
technical 25-of-25-per-cell requirement. Every support had only 3 of 12 cell
means and 1 of 4 zero-signal cell means below 0.75. Their worst absolute mean
pulls ranged from 2.281 to 2.696, above the 1.25 guard.

No eligible edge passed. Edge 33 MeV had the numerically smallest worst
absolute mean pull, 2.28132791864934, but the frozen protocol did not permit a
lowest-score fallback. The canonical decision is therefore
`no_provisional_edge`, with `phase2_supports=[]` and
`observed_scan_authorized=false`.

The independent auditor reproduced that result. The machine-readable terminal
gate is `audit/production_authorization_denied.json`: its audit status passes,
but production status is `production_blocked`, every downstream authorization
is false, and its required protocol action is `stop without retuning`.

### Conditional-truth caveats

The background truth is a conditional source-conditioned stress construction,
not a coverage ensemble or physical background model. Its low-mass shape is a
logistic-exp-Chebyshev degree-5 fit to a pre-existing 2016 10% development
sample/subset. Event-level evidence does not establish that subset as
statistically independent from the full sample, so it supplies partial
observed-shape information. Before support selection, full-2016 bin values
entered only through the scalar 26--210 MeV count of 73,145,594; no
support-specific full-data fit, local p-value, or upper limit ranked an edge.

The truth blends to an archived `fShiftSigPowTail` expectation above the
75--85 MeV transition. That component has archived `fit_ok: false`, although
the independent audit verified a finite nonnegative shape, in-bound
parameters, a positive tail, Pearson chi-square/ndf 0.99003897, expected-total
consistency, and bitwise reproduction of all 100 Poisson toys. Its
nonconverged-fit-status waiver applies only to this conditional stress truth.

The post-decision analytic-mean check removes Poisson fluctuations and retains
the same alternating mass-dependent mismatch: pull ranges are -0.959 to
-0.650 at 44 MeV, +1.923 to +2.181 at 49 MeV, -2.783 to -2.579 at 54 MeV,
and +3.465 to +3.627 at 59 MeV. This diagnoses failure of the conditional
source-recovery test; it is not an observed result, coverage measurement,
sensitivity, exclusion, or significance statement.

## Downstream products that must be absent

The following are absent by design, not missing deliverables:

| Product | Canonical v4.9.7 status |
|---|---|
| Phase-2 75-background continuation | Not authorized; not run |
| Confirmed/frozen 2016 support | Does not exist |
| 65 MeV support holdout | Not authorized; not evaluated |
| Full-2016 support-specific observed card or scan | Not authorized; not run |
| Reviewed 142-row 2016 observed-state ledger | Does not exist |
| Reviewed three-dataset combined-state ledger | Does not exist |
| 2015/2016/2021 v4.9.7 combined upper limit | Does not exist |
| 100-toy v4.9.7 combined epsilon-squared bands | Do not exist |

Any file or figure from an earlier release that shows a combined result remains
historical lineage and must not be relabeled as v4.9.7.

## Canonical v4.9.7 products

| Path | Role | Claim boundary |
|---|---|---|
| `STUDY_PROTOCOL.md` and `study_spec.json` | Frozen rules and exact study contract | Predeclared design, not a result |
| `SCIENTIFIC_SCOPE_CLARIFICATION.md` | Corrects truth-source language without changing the numerical protocol | Provenance qualification |
| `reference/2016_threshold_truth_fit_summary.json`, `inputs/2016_threshold_qualified_background_toys_100.*`, `qa/truth_product_validation.json` | Frozen conditional truth, 100 paired toys, and QA | Conditional stress truth only |
| `derived/analysis/phase1_selection_decision.json` | Canonical support decision | `no_provisional_edge`; no fallback |
| `audit/static_truth_audit.json` and `audit/phase1_selection_audit.json` | Independent reconstruction of truth and Phase-1 gates | Audit evidence |
| `audit/production_authorization_denied.json` | Terminal machine gate | Blocks every downstream production stage |
| `derived/analysis/failed_support_study_summary.json` and associated CSVs/figures | Compact failure and post-decision diagnostic record | No observed or coverage claim |
| `signal_audit/` | Audited explanation of the earlier 2021 65 MeV change | Local-asymptotic robustness diagnostic |
| `note/source/` and `note/HPS_GPR_Analysis_Note_v4p9p7.pdf` | Self-contained analysis-note source and QA-passed release note | Documents the stopped state; contains no new combined result |

The final post-fix note PDF has SHA-256
`cc1a80878d915ad4ed8f2438c2fd5b613d7fae3ffc0793891a898188a91084a1`.
The pagination repair, fresh source-only rebuild, semantic scan, and rendered-page
QA all pass. The superseded pre-fix candidate and its hash are not canonical.

## Why the earlier apparent signal changed

The raw 2021 data feature did not disappear. Its fitted interpretation changed
when the accepted 2021 GP prescription moved from the historical v4.2
40--300 MeV support state to the historical v4.9.5 36--300 MeV state.

At 65 MeV:

| Quantity | Historical v4.2 support 40 | Historical v4.9.5 support 36 | Change |
|---|---:|---:|---:|
| Fitted amplitude, events | 28,038.9233543 | 17,100.8535703 | -39.0103% |
| Amplitude uncertainty, events | 6,609.5299891 | 7,136.4977696 | +7.97285% |
| Local asymptotic p0 | 1.0570174747e-5 | 0.008133321058 | x769.459 |
| Local asymptotic Z | 4.2524933513 | 2.4028773514 | -1.849616 |
| 90% asymptotic epsilon-squared UL | 1.1718399421e-5 | 8.4208872194e-6 | -28.1396% |
| GP log marginal likelihood | 1648.80608746 | 1676.03939277 | +27.2333 |

The observed ROOT input, resolution, density normalization, and conversion
factor were identical. On a common grid, the v4.9.5 diagonal GP state raises
the continuum estimate at 65 MeV by about 1,949 counts per 0.625 MeV bin and
its integral through the nominal masked window by 23,484 counts (0.159413%).
The fitted narrow-signal coefficient consequently decreases.

The accepted support change also shifts the five-native-bin rebin phase by
0.25 MeV, so support extent and bin phase are not isolated. Branch repair is
not the explanation: 65 MeV was not repaired in v4.9.5, and the historical
v4.2 raw-to-accepted change is only 0.091 event and about 0.000017 in Z. This
audit establishes support-prescription dependence, not that the feature is
background or that signal is absent. All quoted p0 and Z values are local and
asymptotic, with no look-elsewhere calibration.

## Canonical, historical, and scaffold states

| Class | Material | How it may be described |
|---|---|---|
| **Canonical v4.9.7** | Stopped 2016 support qualification, denial gate, failure diagnostics, signal-robustness audit, and analysis note | A failed predeclared conditional qualification; no observed or combined v4.9.7 result |
| **Historical v4.2 combined state** | Accepted 2015-full + 2016-full + 2021-10% combination with 300 conditional toys | Historical accepted combined result; lineage only, not recomputed by v4.9.7 |
| **Historical v4.9.5 2021-only state** | Accepted support-36, 300 MeV-high 2021-only observed curve | Historical 2021-only result; not a three-period combination |
| **Unexecuted source scaffolding** | `OBSERVED_2016_WORKFLOW.md`, `observed_2016_workflow_manifest.json`, `README_COMBINED_SCAFFOLD.md`, `combined_scaffold_manifest.json`, and their scripts | Fail-closed code/contracts prepared before the terminal decision; not evidence that production ran |

The scaffold manifests' earlier “awaiting support” status is superseded by the
terminal denial. Do not execute observed-card construction, observed scanning,
ledger assembly, combined-limit production, or combined toy-band production
under this frozen study.

## Analysis note and QA

The self-contained note source is under `note/source/`. Its v4.9.7 sections
document the stopped support study, the downstream production gate, and the
2021 signal-robustness audit while retaining v4.2 combined and v4.9.5
2021-only results under explicit historical labels.

The release-note target is:

`note/HPS_GPR_Analysis_Note_v4p9p7.pdf`

with a bundle-local mirror at:

`output/pdf/HPS_GPR_Analysis_Note_v4p9p7.pdf`.

The repository-level mirror is `../../output/pdf/HPS_GPR_Analysis_Note_v4p9p7.pdf`.
All four build, canonical, bundle-local, and repository-level PDF copies are
byte-identical. The final PDF has 237 pages. A fresh source-only Tectonic build
has identical extracted text; semantic checks found no unresolved placeholders,
undefined references, fatal errors, or overfull boxes. Poppler rendered all 237
pages, ten contact sheets were inspected, and 23 result-boundary pages received
high-resolution inspection. The critical absence statement begins page 95 as a
complete sentence, and standalone historical v4.2 and v4.9.5 result pages are
explicitly labeled. See `note/qa/note_render_qa.json` and `note/qa/NOTE_QA.md`.

## Validation commands

Run from this release directory. These commands audit the frozen stopped state;
they do not authorize or start observed production.

```bash
python3 audit/independent_freeze_audit.py static \
  --accept-broad-tail-fit-status-for-conditional-stress-truth-only
python3 audit/independent_freeze_audit.py phase1 \
  --accept-broad-tail-fit-status-for-conditional-stress-truth-only
python3 audit/independent_freeze_audit.py blocked \
  --accept-broad-tail-fit-status-for-conditional-stress-truth-only
python3 validate_observed_2016.py blocked-state
```

The production combined validator intentionally has no scaffold-only mode: it
requires products that this stopped study never created. No command in this
release can convert the failed Phase-1 outcome into authorization without
defining a new, separately versioned protocol.

Verify the stable decision chain with:

```bash
shasum -a 256 \
  STUDY_PROTOCOL.md study_spec.json SCIENTIFIC_SCOPE_CLARIFICATION.md \
  derived/analysis/phase1_selection_decision.json \
  audit/independent_freeze_audit.py audit/static_truth_audit.json \
  audit/phase1_selection_audit.json \
  audit/production_authorization_denied.json
```

## Isolated-worktree and copy-back status

This package was assembled in the isolated checkout
`/private/tmp/hps-gpr-v4p9p7.YuUjEM/repo` at Git commit
`e2c930f3f879742b2846e3fca1ee1b7e8d99ecc6` (`origin/main`), not in the
materially dirty primary checkout.

**Copy-back status: completed 2026-09-02 16:17:28 PDT (-0700).** The transfer
was restricted to the allowlisted release directory and the named PDF mirror.
The isolated and primary release inventories were checked path by path, byte
for byte, and by SHA-256; the stopped-state validator passed all 11 checks in
both locations. The primary HEAD and remote were unchanged. A canonicalized
fingerprint of all 42,393 pre-existing Git-status records outside the two
allowlisted destinations was also identical before and after transfer. See
`PROVENANCE.md` for the copy-back ledger and exact fingerprint.
