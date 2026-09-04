# v4.9.9 prospective 2016-full GP-support protocol

Frozen before evaluating any candidate-specific predictive score on 2026-09-02.

## Purpose and independence from v4.9.7

This is a new, separately versioned model-selection study.  It does not amend,
override, or reinterpret the terminal `no_provisional_edge` decision in
`v4p9p7_2016_support_combined_100toy_20260902`.  The v4.9.7 conditional
injection-recovery truth showed an alternating analytic-mean mismatch and no
edge passed its frozen pull gate.  This study asks a different question using
held-out background-control prediction and has its own fail-closed rules.

No fitted signal amplitude, signal pull, local p-value, upper limit, expected
band, or toy result is computed or inspected by this selector.

## Reviewed reference prescription

The later matched v4.1 review of the recovered 2016 10% histogram is the
reference, not the older YAML whose single `range_2016: [0.035, 0.210]`
served both search and support roles.  The reviewed card is

`study_results/v4p1_2021_ls_exposure_ensembles_20260804/configs/config_2016_10pct_wide_support_lsupper12.yaml`

with SHA-256 `2d993d9f7c79f738d3b9157b33d473354e81ae138f7c963789263d063dd87762`.
It explicitly uses the 39--180 MeV search and the 30--210 MeV GP data support.
The reference support in this study is therefore 30--210 MeV.

## Immutable inputs

- Development histogram: the recovered/pre-existing 2016 10% development
  sample, `h_Minv_General_Final_1` in
  `../v4p9p7_2016_support_combined_100toy_20260902/inputs/source_2016_10pct.root`.
  File SHA-256:
  `789e619fcbeb5e81f9193d3e224bc17919983477a037bf3d79692327555f9fd4`;
  values-plus-edges SHA-256:
  `db85ff94c74855549c45b173116b36a70a4183ac86db8f374d5b3202c1410656`.
- Confirmation histogram: 2016 full data, the same values and edges as
  `/Users/emryspeets/root_files/EventSelection_pass4Full.root`, archived at
  `../v4p9p7_2016_support_combined_100toy_20260902/inputs/source_2016_full.root`.
  File SHA-256:
  `c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301`;
  values-plus-edges SHA-256:
  `c4225e5bacdb2c1791a6ba943b47695cfddee7f9efed1ffff77065350ba48632`.

No event- or run-level evidence establishes the 10% histogram as disjoint from
the full sample.  It is a development subset/source and Phase 2 is therefore a
same-experiment control-region confirmation, not independent external
validation.

## One-factor candidate grid and geometry

- Fixed upper support edge: 210 MeV.
- Freeze-eligible lower edges: 29, 30, 31, 32, and 33 MeV.
- Reference/tie winner: 30 MeV.
- Ineligible geometry control: 34 MeV.
- Native bin width: 0.05 MeV; production rebin factor: five; coarse width:
  0.25 MeV.  Integer-MeV edge steps preserve the coarse-bin phase.
- Search interval retained for later production: 39--180 MeV.
- Production training exclusion: 2.25 detector-resolution sigma.
- At the 39 MeV endpoint the lower edge of that exclusion is
  35.0632852939 MeV.  Support 33 leaves eight coarse low-side training bins;
  support 34 leaves only four and is therefore a diagnostic control only.

Every other fit setting is fixed to the reviewed v4.1 2016 prescription:
`pre_log=true`, `alpha=1/y`, ConstantKernel times RBF, resolution-scaled local
length bounds with lower factor 0.9 and upper factor 12, the dataset-stat upper
floor factor 0.8, constant bounds `[1e-8, 1e18]`, and 12 optimizer restarts.

## Strictly out-of-search blocked prediction

The support selector is forbidden from fitting or scoring any 39--180 MeV
data bin in either phase.  It also leaves guard gaps 38.75--39 MeV and
180--181 MeV unused.  The only admissible bins are:

- candidate-specific low extension: `[edge, 35.25)` MeV, training only;
- common low control: `[35.25, 38.75)` MeV;
- common high control: `[181, 210)` MeV.

The common controls are partitioned before fitting:

| Block | Half-open interval (MeV) | Coarse bins |
|---|---:|---:|
| L1 | [35.25, 36.00) | 3 |
| L2 | [36.00, 36.75) | 3 |
| L3 | [36.75, 37.75) | 4 |
| L4 | [37.75, 38.75) | 4 |
| H1 | [181.00, 188.25) | 29 |
| H2 | [188.25, 195.50) | 29 |
| H3 | [195.50, 202.75) | 29 |
| H4 | [202.75, 210.00) | 29 |

For each fold, the named block is held out.  Training uses the
candidate-specific low extension and the seven other control blocks only.
Thus the held-out values cannot be absorbed by the fitted background, and no
search-region value can influence a fit or score.

The resolution-scaled kernel policy is evaluated at five predeclared anchors,
39, 65, 100, 140, and 180 MeV.  The data used are identical at each anchor;
the anchors only exercise the production mass-dependent kernel bounds.

## Optimizer and covariance rules

Each support/dataset/anchor/fold fit is repeated with deterministic random
seeds 1961, 5813, and 9049.  Each repeat includes the nominal kernel start plus
12 random restarts.  The selected branch is the finite maximum GP log marginal
likelihood within that exact support/dataset/anchor/fold cell.  LML is never
compared across supports.

A cell passes only if:

1. all three repeats finish with finite LML and finite positive kernel values;
2. at least two repeats reproduce the selected LML within absolute `1e-4`;
3. the reproduced length scales agree within relative `1e-3`;
4. neither the selected length nor constant is within relative `1e-3` of a
   bound; and
5. the count-space predictive covariance plus Poisson diagonal is finite,
   symmetric positive definite, has no eigenvalue below
   `-1e-8 * max(diag(V))`, and requires Cholesky jitter no larger than
   `1e-8 * median(diag(V))`.

A support passes the technical gate only if every required cell passes.  No
interpolation or smoothed replacement is allowed.

## Predictive scores

For held-out count vector `y`, lognormal GP count mean `mu`, and lognormal GP
count covariance `C`, define `V = C + diag(mu)`.  The primary proper score is
the multivariate-normal negative log predictive density per bin,

`NLPD = [ (y-mu)^T V^-1 (y-mu) + log det(V) + n log(2 pi) ] / (2 n)`.

The joint Mahalanobis statistic per bin and the Poisson deviance per bin are
stored as diagnostics.  They cannot replace the primary NLPD direction.

For each dataset and support, scores are first averaged over the five kernel
anchors within each block.  Low and high block means then receive equal total
weight:

`S = 0.5 * mean(L1..L4) + 0.5 * mean(H1..H4)`.

This prevents the 116-bin high control from swamping the 14-bin low control.
All comparisons are paired block-by-block to the 30 MeV reference.  For the
paired improvement `Delta = S_30 - S_candidate`, positive is better.  Its
stratified standard error is

`SE = sqrt(0.25 * var(Delta_L)/4 + 0.25 * var(Delta_H)/4)`.

## Sequential selection and null fallback

Phase 1 evaluates all eligible supports and the 34 MeV geometry control on the
2016 10% development histogram.  A nonreference support becomes a Phase-2
qualifier only if it passes the technical gate and all of the following:

1. primary paired NLPD improvement over support 30 is greater than one
   stratified standard error;
2. low-control NLPD improvement is positive;
3. high-control NLPD degradation, if any, is smaller than one high-stratum
   standard error;
4. stratified paired Poisson-deviance improvement is nonnegative; and
5. after deleting any one of the eight blocks, the primary paired NLPD
   improvement remains positive.

Phase 2 evaluates support 30, every Phase-1 qualifier, and their immediate
eligible neighbors on the full-2016 control bins.  Only a Phase-1 qualifier may
displace the reference, and it must pass the same five requirements in Phase 2.

If multiple candidates qualify in both phases, rank them by the smaller of
their Phase-1 and Phase-2 `Delta/SE` values.  Values within 0.25 are a practical
tie; choose the edge nearest 30 MeV, then the lower edge to retain more support.

If no candidate passes both phases, if development and confirmation do not
support the same candidate, or if predictive differences are practically tied,
retain the reviewed 30--210 MeV reference provided it passes both technical
gates.  If support 30 fails either technical gate, stop with no selected
support.  The 34 MeV geometry control can never be frozen.

## Claim boundary and downstream authorization

A successful freeze is a data-informed, out-of-search control-region support
choice for a partially unblinded analysis.  It is not a coverage result,
expected sensitivity, exclusion, global-significance calibration, or proof
that the background model is physically correct.  Later observed/asymptotic
limits and local p-values are conditional on the frozen support.  Toys and
bands are outside this study.

No observed 2016 scan or combination is authorized until the Phase-2 decision,
all ledgers, input/code hashes, and a machine validation report exist.
