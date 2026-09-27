# v5.8 history and source audit

Read-only specialist review completed against current checkout sources and ledgers. No fit, toy, source reconstruction or parent modification was performed. The report section is `source/history.tex`; numeric details and SHA-256 identities are in `results/study_ledger.json`.

## Main answer

The user's concern about a change of inference is valid. The quoted 3.4525 -> 2.886 comparison includes **local reference centering and scaling**, not merely a global look-elsewhere correction. The unchanged raw maximum occurs at 90.5 MeV. Its own reference score is 2.78978; the reference score at 91.5 MeV is 2.88595 because its offset is smaller. The subsequent global tail produces Z=1.25423, a separate step. Do not describe the full difference as a trials penalty or as a new observation.

The reviewed documents do not establish a decision to make reference calibration the production inference. They consistently qualify it as a conditional diagnostic and preserve the raw observed fit. v5.8.0 explicitly recommends retaining the fixed-mass likelihood result as the starting statistic; v5.8.2/3 introduce diagnostic local probabilities conditional on an estimated frozen GP source. The consolidation should acknowledge this change of diagnostic ordering openly rather than claiming centering never occurred.

## v5.0.5 exact context

- Section 4.20 is PDF pp58--61, `source/sections/v5_global_significance.tex`.
- Eq88 defines a=r(B); Eq89 defines one-bin perturbation directions; Eqs90--92 covariance, scale and correlation; Eq93 draws r*=a+sZ; Eq94 defines standardized z; Eq95 gates on raw positive fit; Eq96 defines maximum/global tail; Eq97 counts exceedances.
- Section 6.11 starts p114, `source/sections/v5_global_results.tex`.
- The note explicitly warns that this offset/gate extension does not automatically acquire the paper's discovery interpretation; no RBF kernel is imposed on the significance-field covariance.
- The combined ~9 score at76 has raw r=.166 and stress a=-8.700; it is reference mismatch. Individual2016 raw r=-2.460. This is already explained as gate plus offset rather than a resonance width.
- The two-truth empirical calibration and raw asymptotic local p-values are different objects. Passing former cannot automatically establish latter under arbitrary new references. v5.8.1 nominal-source raw5% rejection101/512@90 and97/512@117 is direct evidence of that distinction.

## Figure explanations

v5.8.2 Figure3: the plot script confirms colored a is a raw deterministic root, black is mean((r_toy-a)/s), gray is SD((r_toy-a)/s). Those moments are different quantities on a shared axis. Near-zero centered mean does not show that a was negligible. Physics specialist owns improved visualization and deep source discussion.

v5.8.3 Figure4: for each mass lag0.5--20MeV, script averages matrix diagonals. Band is pair-location10th--90th percentiles, not statistical uncertainty. Exact varying-width Gaussian overlaps form benchmark. Negative lobes can reflect positive signal-region weights plus compensating background responses; the audit does not isolate causality of mask switching. Full global calculation uses whole covariance, not mean-lag curve. In particular rho<0 does not mean independent.

v5.8.3 Figure5: reference-local threshold on x axis, common exceedance probability on y. Paper-style known-background smooth-template benchmark and FWHM-count heuristic differ from actual fitted-response matrix. All plotted thresholds>=1.5 exceed every positive-fit gate in saved fields. 2016 observed threshold2.88595489 yields one-mass.00195114, FWHM.02953024, template.06521001, field.10487948, independent-grid.42461283. MC bands omit source uncertainty. Comparing with FWHM is not a proof of overconservatism.

## Preserved limits

- v5.8.1 source absorption is a construction vulnerability, not proof that a specific observed peak is signal or exactly that percentage of observed signal was subtracted.
- Source mean uses full support, observed counts and kernel anchored at76MeV; local extractor uses moving2.25sigma masks and mass-dependent kernels. Distinct operations explain nonzero deterministic response even with a smooth source.
- 5.8.2/3 use0.5MeV grid;5.8.4/5 use1MeV. Do not compare global numbers as if configurations matched.
- v5.8.4 width controls change both held-out and likelihood bins, keep source fixed. No3sigma source-policy test exists in these old results.
- One full2016 width2.6 tail comparison is flagged, not automatic pass.
- q_R exact likelihood identity does not make sqrt(q_R) a Z. Local/global calibration differs from shared coupling, Fisher and Stouffer.
- Free-amplitude baseline peak:92MeV, q_R17.90417, localZ3.58287, globalp.0196798/Z2.06041; directglobal7/256 and95%CI.011063--.055525. Raw q_R itself peaks91MeV.
- Coherence:195/256 lower-tail exceedances, add-onep.762646; no unusual observed mass clustering.
- LEE affects discovery tails, not pointwise90%CLs endpoints. Discovery-reach forecasting is a separate signal-plus-background problem.
- Retain2.25sigma. Do not choose methods/widths/domains by smallest reported p.

## Figure delivery paths

Root has already copied the two relevant v5.8.3 plots under `figures/archive_response_resolution.pdf` and `figures/archive_trials_comparison.pdf`. `history.tex` uses those portable copies through the agreed `fig` macro. Full study ledger includes other important original figure paths for optional inclusion.

## Memory use

Quick pass used MEMORY.md lines1--24 only to route v5.8 study paths; all scientific statements above verified from current checkout. Related rollout IDs01a0b05f-003c-7402-b414-47156e7690e5 and01a0bad7-b610-7671-8164-d18b6934c385. No memory writes.

## Independent cross-review of the consolidation

Read the new physics/statistics sections, ledgers, diagnostic scripts and root decisions/conclusions/reproducibility sources. The raw and reference maximum calculations use their respective saved maxima consistently; the discrete adjacent-Gaussian crossing probability and union bound are correct; the reported rho selection demonstrably cannot establish region independence. The high-psum rebinning, nominal factor10 and source injection divided by10 correctly implement the stated conditional diagnostic. Full-visible4.476% and same-support4.483% count ratios are different domains and are not inconsistent. Sample identity, exposure and transfer caveats are retained. No substantive physical/statistical contradiction found in those new claims.

Flags sent to owners: physics and statistics figure optional arguments initially included `linewidth` even though root macro appends it; use numeric fractions. Physics scripts initially depended on original checkout and fixed wall-clock deadline; physics owner reports these changed to bundled input paths and relative bounded runtime with STOP guard. Root combined-method caption should distinguish Fisher gated p-values from continuous signed Stouffer scores. Repetition of raw/reference mapping across opening, history and statistics can be trimmed editorially; the new same-field upcrossing plot is important and should remain because it answers the paper comparison directly.
