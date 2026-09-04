# Provenance: HPS-GPR v4.9.7 stopped support study

## Release identity and terminal state

| Field | Value |
|---|---|
| Study | `v4p9p7_2016_support_combined_100toy_20260902` |
| Release label | `v4.9.7` |
| Study date | 2026-09-02 |
| Canonical state | `study_stopped_no_provisional_edge` |
| Canonical support decision | `no_provisional_edge` |
| Required protocol action | `stop without retuning` |
| Isolated source checkout | `/private/tmp/hps-gpr-v4p9p7.YuUjEM/repo` |
| Source Git revision | `e2c930f3f879742b2846e3fca1ee1b7e8d99ecc6` (`origin/main`) |
| Source remote | `git@github.com:EmrysPeets/hps-gpr.git` |
| Primary checkout | `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow` |
| Primary HEAD before copy-back | `5229f544ab67ed06c48a9b769ffac9f9b18c2a5c` |
| Copy-back | Completed 2026-09-02 16:17:28 PDT (-0700), restricted to the release directory and named PDF mirror |

The only canonical scientific outcome of the proposed 2016-to-combination
extension is the failed, predeclared Phase-1 support qualification. There is no
v4.9.7 observed 2016 scan, combined upper limit, or combined toy band.

## Frozen inputs and exact hashes

### Data and truth inputs

| Path | SHA-256 | Bytes / count | Permitted role |
|---|---|---:|---|
| `inputs/source_2016_10pct.root` | `789e619fcbeb5e81f9193d3e224bc17919983477a037bf3d79692327555f9fd4` | 3,090,442 bytes; 7,475,607 counts over 26--210 MeV | Pre-existing 2016 10% development-subset shape input |
| `inputs/source_2016_full.root` | `c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301` | 4,441,109 bytes; 73,145,594 counts over 26--210 MeV | One scalar normalization before freeze; later observed use remained unauthorized |
| `inputs/2016_threshold_qualified_background_toys_100.root` | `689c700dc358db439a5da3eaa4bba4ee37f9d2d157afd10680b80cee1be2e912` | 1,102,629 bytes; 100 toys | Frozen conditional Poisson-toy product |
| `inputs/2016_threshold_qualified_background_toys_100.manifest.json` | `2c79965165c7186bb1bab4bb392d58c76974544735512a92f3c33ff8c3496773` | 47,288 bytes | Toy inventory, seeds, and hashes |
| Manifest content hash | `2c7cc7960fe6a1f4a3782537647855fc4d45599f22c206d4ef366a66ca2ee482` | -- | Content-level toy binding |
| `reference/2016_threshold_truth_fit_summary.json` | `239573b96cbf3efeb230da93f07071d28aad55023010666ed03c3ff2d3d2f8d9` | selected degree 5 | Frozen truth-fit and degree-selection record |
| `qa/truth_product_validation.json` | `18829241566069f75ba7b1069d22b8e134ff06b03bc915c0cde02bfc818ea536` | status `pass` | Truth-product QA |

Both ROOT sources use histogram `h_Minv_General_Final_1`. The toy seed is
20260902. All 100 backgrounds were regenerated bitwise and paired by toy index
across support, test mass, and injected strength.

The selected low-mass model is the lowest degree passing the frozen source-GOF
gates: degree 5 in the logistic-exp-Chebyshev candidate family. Degree 4 fails;
degrees 5--10 pass, so no higher-degree score optimization was used. The final
mean blends the degree-5 low-mass fit into the archived
`fShiftSigPowTail` expectation across 75--85 MeV and is normalized to the one
full-sample scalar count.

### Truth limitations and waiver

The 10% input is not established as event-level disjoint from the full sample.
It must be called a pre-existing development sample/subset that supplies
partial observed-shape information, not an independent sample. The full 100%
array was not used for support-specific fits before freeze: only its scalar
26--210 MeV total entered the generating mean. No full-sample fitted amplitude,
local p0, or upper limit ranked support edges.

The archived broad-tail metadata records `fit_ok: false`. Independent checks
found Pearson chi-square/ndf 0.99003897, finite nonnegative values, free
parameters inside their stated bounds, a positive 85--210 MeV tail, agreement
with the archived expected total, and bitwise toy regeneration. Accepting that
status is an explicit waiver for a conditional source-conditioned stress truth
only. It is not approval as a physical background model, coverage ensemble,
calibrated sensitivity, exclusion model, or significance calibration.

## Frozen protocol, card, and runtime

| Path | SHA-256 | Role |
|---|---|---|
| `STUDY_PROTOCOL.md` | `81e5954c6bb1073010f32af8ab2fccc94d922f94018abe6416238e9d92cbec02` | Frozen support-selection protocol |
| `study_spec.json` | `4382bfa6298cafe43d45026708017ca3e43179700f2ab5c76a557411874c8b3f` | Machine-readable study specification |
| `SCIENTIFIC_SCOPE_CLARIFICATION.md` | `7e90ed186396f3e209f6591ccdd28df714b642137797c07e0ed048bd02656b2c` | Correct truth-source language and waiver scope |
| `inputs/frozen_v4p2_analysis_card.yaml` | `5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055` | Base analysis configuration |
| `run_support_scan.py` | `86e0349fe959553a213d648adc25d3bcf54a69400767ad59e6f14a141aecd773` | Support-scan production driver |
| `analyze_support_scan.py` | `b7a4bd123eb5b85a271bcb8dfd9e1d865fc686ec54d35102d02be532f64bf3e0` | Frozen Phase-1 analyzer |
| `confirm_support_edge.py` | `917a42888061280983f82a487e1e169c4dd3b9516f0e0334cae2e38acf83fedb` | Frozen Phase-2 confirmation logic; never invoked for production |
| `runtime_overlay/hps_gpr/__init__.py` | `342aaa16dc390a3b79ef605987de8dc610b87e9bc774fe5edfec5e7a56883687` | Complete archived-package marker |
| `runtime_overlay/hps_gpr/gpr.py` | `1c83cae238e87a4e94928c97fb737947c22a3f88b16dfaf955d48ab6b4771dd5` | Archived instrumented GP implementation |
| `runtime_overlay/hps_gpr/io.py` | `b36f8da7671a0fc0958b663e11d83a1a4421e90d1aab9b10e40c31ce078035db` | Archived instrumented I/O implementation |

Preflight failed closed unless Python resolved `hps_gpr.io` inside this runtime
overlay and exposed the required optimizer-state fields. This prevents a
correct file manifest from masking an import of the base checkout package.

The exact scan geometry was:

- dataset `2016`, support high edge 210 MeV;
- candidate lows 28--34 MeV, with 28--33 eligible and 34 a control;
- test masses 44, 49, 54, and 59 MeV;
- injected strengths 0, 2, and 5 sigma;
- Phase 1 toy indices 0--24 for all seven supports;
- Phase 2 toy indices 25--99 only for a provisional edge and immediate
  available neighbors;
- 65 MeV reserved as a post-freeze holdout and forbidden from selection.

## Execution chronology

1. The protocol, study specification, sources, truth builder, runtime overlay,
   support driver, and analyzers were frozen and hash-bound before support
   extraction.
2. The truth builder selected degree 5 under the fixed gate, blended the broad
   tail, applied only the full-sample scalar normalization, and produced 100
   seed-20260902 paired Poisson backgrounds. Static and product QA passed under
   the explicit conditional-truth waiver.
3. A one-toy/one-support benchmark verified the complete overlay import and
   production path. An earlier invalid benchmark made through the wrong Python
   package was quarantined and did not enter the study.
4. Seven single-worker Phase-1 lanes evaluated all seven support candidates.
   The planned grid contained 2,100 rows. The accepted ledger contains 2,098;
   supports 30 and 32 MeV each have one retained irreproducible optimizer refit
   and fail the complete-cell technical rule.
5. The frozen analyzer found that every support has 3/12 qualifying cell means,
   1/4 qualifying zero-signal means, and a worst absolute mean pull above 1.25.
   All eligible supports therefore failed. The 33 MeV edge had the smallest
   score, 2.28132791864934, but no fallback selection was allowed.
6. The independent auditor reconstructed optimizer branches from attempt
   ledgers, recomputed all cell/support summaries, and reached the same
   `no_provisional_edge` decision. It emitted the terminal production-denial
   artifact.
7. The protocol stopped. Phase 2, support freezing, the 65 MeV holdout,
   observed-data card construction, observed scanning, reviewed-ledger
   assembly, combination, and 100-toy combined bands did not run.
8. Post-decision analytic-mean and historical 2021 signal-robustness diagnostics
   were produced without changing or reopening the frozen selection.

## Canonical decision and independent audits

| Path | SHA-256 | Status |
|---|---|---|
| `derived/analysis/phase1_selection_decision.json` | `be1ac60e7b0420fc762a030ad579c855f65b20e41e4c32b03d514a804c82e71d` | `no_provisional_edge`; Phase 2 empty; observed unauthorized |
| `audit/independent_freeze_audit.py` | `c53bd7bc066d37bc593b910a109912c26719ecd5d61bd13974a6b2e826a51058` | Final stable independent auditor |
| `audit/static_truth_audit.json` | `f27ff7400a82a8b0667e172766026b9007e2155eb447ccae05bf6adf17094964` | `pass` under explicit conditional-stress waiver |
| `audit/phase1_selection_audit.json` | `1118f5b293719bffe17217c5d24a6bf32f74a7a453b4ffd038fae7a34fce9416` | `pass`; independently no selected support |
| `audit/production_authorization_denied.json` | `c71b569da432723715922532e763b79dec6c0f9f04a08f84c0e190345c9d2b60` | Audit pass; production blocked |
| `derived/analysis/failed_support_study_summary.json` | `4b3c7f8d8ca5cc07fa202a122227bccb9a6b586d767aaa7185703e0624a5e700` | Compact terminal summary |

The canonical decision binds these Phase-1 products:

| Product | Rows | SHA-256 |
|---|---:|---|
| `derived/analysis/phase1_accepted_rows.csv` | 2,098 | `228e5bf6b6bc7b30d74afb79f875a39db958bc0c31ac1a72febf25ff438a55bd` |
| `derived/analysis/phase1_cell_summary.csv` | 84 | `4b370d258f51da017230c7b08a2e15bce81db516a30749047aaadbc856f1078e` |
| `derived/analysis/phase1_support_summary.csv` | 7 | `29d5c22a50e39d8f538dafd5dd2deb146013711e126e3dd3a01a1499676b2124` |
| `derived/analysis/phase1_adjacent_paired_differences.csv` | 72 | `6dac7d80e1d63c3c0e28ac80c1c4136ab7e420ef7d486f8fb4d8e2b5498bc47c` |

The post-decision failure bundle contains:

| Product | SHA-256 |
|---|---|
| `derived/analysis/phase1_support_failure_summary.csv` | `cbe491db71daa4465819c5e0c46ad162fbf0263365539f7375aade9c498139b8` |
| `derived/analysis/analytic_mean_zero_signal_all_supports.csv` | `36ca0bbd12023cc750189fca1e2fff7cf4d924d321c9cbfa0eaaf86ea03a6052` |
| `derived/analysis/phase1_technical_exclusions.csv` | `942ec977b37e355412b4ebbcdb5b30a06ec384dc60dcdb9710841cf400163766` |

The analytic-mean zero-signal pull ranges are -0.958756 to -0.650380 at
44 MeV, +1.922714 to +2.181175 at 49 MeV, -2.782579 to -2.578930 at
54 MeV, and +3.465240 to +3.626866 at 59 MeV. This uniform alternating
mismatch is diagnostic only and was not used to choose a support.

## Canonical inventory versus lineage and scaffolding

### Canonical v4.9.7 evidence

Canonical evidence comprises the frozen protocol/specification and sources,
truth products and audits, executed Phase-1 ledgers, the no-edge decision,
production-denial artifact, failure diagnostics, the historical 2021
signal-robustness audit, and the post-fix analysis note after its final QA and
hash record. None is an observed 2016 or combined v4.9.7 result.

### Historical result lineage

- **Historical v4.2 combined state:** the accepted
  2015-full + 2016-full + 2021-10% result with 300 conditional toys. It is
  referenced for provenance and comparison only; v4.9.7 did not reproduce or
  supersede it.
- **Historical v4.9.5 2021-only state:** the accepted 2021-only observed scan
  using support 36--300 MeV and its reviewed branch state. It is not a
  three-period combination.

### Unexecuted source scaffolding

`OBSERVED_2016_WORKFLOW.md`, `observed_2016_workflow_manifest.json`,
`README_COMBINED_SCAFFOLD.md`, `combined_scaffold_manifest.json`, and their
card-building, review, assembly, validation, and cached-band scripts are
unexecuted, fail-closed source artifacts. Their earlier “awaiting support”
language is superseded by the canonical production-denial gate. They must not
be used as evidence for an observed scan, a reviewed state, a combination, or
toy bands.

For traceability, `combined_scaffold_manifest.json` had SHA-256
`1b4a2883a807d9ec4535c1fc5ce9276cbbe908e42460a6efb1476e3e7d98e1dd`
when the scaffold was frozen, and its ledger assembler
`assemble_reviewed_state_ledger.py` had SHA-256
`7e749adb00ef8d552580217616e4732e838d08b8295bde7509b36d668ca6854e`.
Those hashes bind code, not a production result.

## Signal-robustness audit and source lineage

The signal audit is canonical v4.9.7 diagnostic evidence about two historical
2021 states. Its machine summary is
`signal_audit/derived/signal_robustness_summary.json`, SHA-256
`0a657904b9c079c5471e87baffb2d517c522ff1b71be23c50ccf7d978ddd1289`.
Its artifact manifest has SHA-256
`9a0dfbfcf73984ba1849bbc9e8d36031fdc023371923a3d0876bbf77bbea9b89`;
semantic QA is 13/13 pass
(`b8e967874afea48b4106e08bbe97dd69209e98eab60ace27476d6af61d0f64ad`),
and PDF-render QA is 3/3 pass
(`ae56fa1820e60f83f75d6c033c98f9efb68a5cc3b8f48ccfe559246c77ef1eae`).

Primary lineage inputs were:

| Historical source | SHA-256 | Role |
|---|---|---|
| v4.2 reviewed individual ledger | `1e3e99fb7c0a171d6d496de87ac6664b485928042b2cede242dffab55e0cc410` | Accepted old 2021 curve |
| v4.2 exact 65 MeV extraction | `dc06707637511644e6bad06638451351a9995b9e363b4cfe0aeddcae18bf3c4f` | Exact old 65 MeV state |
| v4.2 standalone/pairwise rows | `efa73576adae356d4805b7548a0bc14da4d4a2572fd6a19cc6404e7cd5386e47` | Historical scope diagnostic |
| v4.2 all-three reviewed bands | `8f4b37ff6a998e236c1ea959db56a76f21ce509c05f24c17675cef676fcbeadd` | Historical all-three 65 MeV row |
| v4.9.5 support-36 2021 curve | `28e6a10b8633fc69c1bab62d32fe39417c42ac886ef27f74ca0c9aeb7cc620e9` | Accepted new 2021-only state |
| v4.9.5 optimizer-repair ledger | `8391d3f7ffbb9e6585f1a43de0bce03c9d27f0c132544201554fe4c3e654fb58` | Confirms 65 MeV was not repaired |
| Common 2021 observed ROOT source | `3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4` | Same observed input for both states |

At 65 MeV, the historical v4.2 support-40 state has fitted amplitude
28,038.9233543 +/- 6,609.5299891 events, local asymptotic
`p0=1.0570174747e-5` (`Z=4.2524933513`), epsilon-squared 90% asymptotic
upper limit `1.1718399421e-5`, and GP log marginal likelihood 1648.80608746.
The historical v4.9.5 support-36 state has 17,100.8535703 +/-
7,136.4977696 events, `p0=0.008133321058` (`Z=2.4028773514`), upper limit
`8.4208872194e-6`, and log marginal likelihood 1676.03939277.

Thus the fitted amplitude changed by -39.0103%, its uncertainty by +7.97285%,
p0 by a factor 769.459, and the upper limit by -28.1396%. Density
normalization and the signal-yield conversion factor are identical. The new
fixed GP state raises the common-window continuum integral by 23,484 events
(0.159413%) and the GP mean at 65 MeV by about 1,949 counts per 0.625 MeV.
The raw feature remains in the data; its narrow-signal attribution is not
robust to the support prescription.

This is not attributable to optimizer repair at 65 MeV. The v4.9.5 repairs are
at 94, 152, and 212 MeV. The historical v4.2 raw and accepted 65 MeV states
differ by only 0.091 event and roughly 0.000017 in Z. The lower-support change
also moves the coarse-rebin phase by 0.25 MeV, so the available comparison
does not isolate support extent from rebin phase. All p0 and Z values are local
and asymptotic, without look-elsewhere correction. Controlled pair/all-three
swaps are diagnostics, not official combined scans.

## Explicit non-execution and absence assertions

The following canonical paths are absent and must remain absent for this
frozen study:

| Stage | Prohibited/missing canonical path or product |
|---|---|
| Positive confirmation | `audit/confirmation_freeze_audit.json` |
| Support freeze | `derived/analysis/support_freeze_decision.json` |
| Observed 2016 card | `inputs/v4p9p7_observed_2016_full_frozen_support_card.yaml` and manifest |
| Observed primary scan | `observed_scan/2016_full_primary/results_single.csv` |
| Observed repair/review | `observed_scan/final_2016/optimizer_repair_ledger.csv`, repair plan, review summary, and reviewed CSV |
| Observed validation | `qa/observed_2016_review_validation.json` |
| Combined production | Reviewed three-state ledger, combined observed curve, and 100-toy combined-band products |

`validate_observed_2016.py blocked-state` independently checks the first six
observed-stage absences, reports `observed_data_evaluated=false`, and requires
the denial gate. The combination is likewise unauthorized because its required
frozen 2016 support and reviewed 142-row ledger do not exist.

## Analysis note build and QA

The analysis-note source is self-contained under `note/source/`. It carries the
terminal v4.9.7 result boundary, labels the v4.2 combination and v4.9.5
2021-only scan as historical states, and incorporates the failed support and
signal-robustness figures. `note/source/writing_sample.tex` is an inherited,
separate artifact and is not a v4.9.7 deliverable.

The canonical release targets are:

- `note/HPS_GPR_Analysis_Note_v4p9p7.pdf`;
- repository mirror `output/pdf/HPS_GPR_Analysis_Note_v4p9p7.pdf`.

The pre-fix candidate split a critical negation across a page boundary and is
not canonical. The repaired final note begins page 95 with the complete sentence
that the v4.9.7 combined upper limit and 100-toy combined band are absent by
construction. Historical v4.2 and v4.9.5 result headings and captions are also
explicit on their standalone pages.

Final note record:

| Item | Value |
|---|---|
| Canonical PDF SHA-256 | `cc1a80878d915ad4ed8f2438c2fd5b613d7fae3ffc0793891a898188a91084a1` |
| PDF size and pages | 33,598,322 bytes; 237 Letter pages |
| Source-tree inventory | 249 files; aggregate SHA-256 `6bdef881e198b30fce61e733f7667b5e6f16268d38e33977376865bf39e36dd0` |
| Extracted-text SHA-256 | `b9d0ae76e6b4c248f371ce6f1ad045178c2328171a68d71aceeb929d57d944b6` |
| Fresh source-only PDF SHA-256 | `735472b9f0a2b60af7cb96f9042718907577ad7816d86b7b77b935eaca4dd9e4` |
| Source-only extracted text | Byte-identical to canonical extracted text |
| Render coverage | 237/237 pages at 90 dpi; ten contact sheets; 23 selected pages at 180 dpi |
| `note/qa/note_render_qa.json` SHA-256 | `ac353b85dbd8c8aff56fa047b9ff9d0bc956361f1f1fdb5bda4b383524e6b1ba` |
| `note/qa/NOTE_QA.md` SHA-256 | `cfd8a4a70663a8d52e44b21a2f8c5f44e57c59c75cebd04921d57afe40528b13` |

Acceptance checks completed:

- successful source-only Tectonic build;
- zero missing, outside-tree, or absolute source references and zero symlinks;
- no undefined references/citations, fatal errors, or overfull boxes;
- semantic extraction with the v4.9.7 non-result statements intact;
- Poppler rendering of every page and visual contact-sheet inspection;
- high-resolution inspection of the v4.9.7 support, signal-audit,
  result-boundary, conclusions, and final bibliography pages;
- byte-identical build, canonical, bundle-local, and repository-level mirrors.

## Validation record and commands

The stopped-state checks are:

```bash
python3 audit/independent_freeze_audit.py static \
  --accept-broad-tail-fit-status-for-conditional-stress-truth-only
python3 audit/independent_freeze_audit.py phase1 \
  --accept-broad-tail-fit-status-for-conditional-stress-truth-only
python3 audit/independent_freeze_audit.py blocked \
  --accept-broad-tail-fit-status-for-conditional-stress-truth-only
python3 validate_observed_2016.py blocked-state
```

The last recorded blocked-state result in
`observed_2016_workflow_manifest.json` is `pass`, with all prohibited products
absent and `observed_data_evaluated=false`. Static code preflights for the
unexecuted observed and combined scaffolds do not supersede the production
denial.

## Isolated-worktree and copy-back ledger

| Item | Recorded state |
|---|---|
| Isolated package source | `/private/tmp/hps-gpr-v4p9p7.YuUjEM/repo/study_results/v4p9p7_2016_support_combined_100toy_20260902/` |
| Isolated Git revision | `e2c930f3f879742b2846e3fca1ee1b7e8d99ecc6` |
| Primary destination | `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/v4p9p7_2016_support_combined_100toy_20260902/` |
| PDF mirror destination | `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/output/pdf/HPS_GPR_Analysis_Note_v4p9p7.pdf` |
| Primary HEAD before transfer | `5229f544ab67ed06c48a9b769ffac9f9b18c2a5c` |
| Primary remote | `git@github.com:EmrysPeets/hps-gpr.git` |
| Current transfer status | **Copied back and verified** at 2026-09-02 16:17:28 PDT (-0700) |
| Primary target state | Exact release inventory present; named PDF mirror present |

The primary checkout was materially dirty before this work. The transfer was
therefore limited to the two destinations above. The recursive
`release_manifest.json` records every immutable payload path, byte count,
SHA-256, and role; it explicitly excludes only itself and the mutable validator
report. Its final file SHA-256 is reported in the transfer handoff to avoid a
self-referential provenance update.

| Transfer verification | Result |
|---|---|
| Isolated-to-primary release comparison | Identical relative paths, byte counts, and SHA-256 values |
| Final PDF mirror | 33,598,322 bytes; SHA-256 `cc1a80878d915ad4ed8f2438c2fd5b613d7fae3ffc0793891a898188a91084a1` |
| Isolated validator | PASS, 11/11 checks; production remains blocked |
| Primary validator | PASS, 11/11 checks; production remains blocked |
| Primary HEAD after transfer | `5229f544ab67ed06c48a9b769ffac9f9b18c2a5c` (unchanged) |
| Primary remote after transfer | `git@github.com:EmrysPeets/hps-gpr.git` (unchanged) |
| Outside-allowlist Git-status records | 42,393 before and after |
| Outside-allowlist canonical status SHA-256 | `e73bb01d8157a895f15ad9c9f2dd9c3e2322961962cde8d1f2c472f3a560cf40` before and after |

The first destination-side validator invocation correctly exposed that its
static-audit resolver still consulted the live repository for paths labeled
`repository/...`. The resolver was made portable by binding those entries to
the packaged `runtime_combined/` snapshot, whose hashes exactly match the
frozen static audit. The live PDF mirror remains a separate validation target.
No canonical scientific input, Phase-1 result, or claim changed in this
packaging-only correction.

## Claim boundary

The v4.9.7 support result is a terminal failure of a predeclared conditional
source-conditioned pull-recovery criterion. It is not an observed 2016 result,
coverage test, expected sensitivity, exclusion, or significance statement.
The 2021 signal audit is a local-asymptotic robustness and mechanism diagnostic;
it cannot classify the 65 MeV feature as signal or background. The historical
v4.2 combined result and historical v4.9.5 2021-only result retain their own
release identities. No combined v4.9.7 physics result exists.
