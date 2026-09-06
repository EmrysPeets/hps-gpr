# 2021 10% conditional 65 MeV central-window replacements

This study asks whether the 65 MeV feature in the observed 2021 10% spectrum
persists when the central region is replaced by background-only pseudo-data.
It is grounded in the frozen HPS-GPR v4.2 analysis and makes no change to the
production inference settings.

## Scope and interpretation

- The requested continuous interval is
  \(65\ {\rm MeV}\pm2.5\sigma_m=[59.6958,70.3042]\ {\rm MeV}\), using
  \(\sigma_m(65\ {\rm MeV})=2.121696875\ {\rm MeV}\).
- On the production 0.625 MeV grid, the \(\pm2.25\sigma_m\) and
  \(\pm2.5\sigma_m\) center selections are identical: both select 16 analysis
  bins, with complete-bin edges \([60,70)\) MeV. The ROOT construction replaces
  the corresponding 80 native 0.125 MeV bins.
- The limit and local-\(p_0\) scans retain the frozen v4.2
  \(\pm2.25\sigma_m\) extraction/training/edge-guard geometry. The wider
  \(\pm2.5\sigma_m\) request affects only which complete central bins are
  replaced; it does not redefine the inference card.
- No nonzero signal strength was specified, so the closed interpretation is
  background-only: \(A_{\rm inj}=0\).
- These are **conditional central-window replacements**, not independent
  global-null pseudoexperiments. Both lanes share the same unchanged observed
  spectrum outside \([60,70)\) MeV. Consequently, the outputs do not establish
  expected sensitivity, coverage, a global \(p\)-value, or a discovery claim.

## Two replacement lanes

1. **GP mean:** the exact accepted v4.2 2021 fixed-GP state at 65 MeV is
   reconstructed from the reviewed state ledger. Independent binwise Poisson
   counts are drawn around its count-space mean.
2. **Functional form:** `fGenGammaThresh` is fitted with a binned Poisson
   likelihood over 50--85 MeV, excluding \([60,70)\) MeV and requiring
   nonempty sidebands on both sides. Its fitted amplitude is profiled
   analytically, five deterministic optimizer starts are compared, and the fit
   must pass the recorded deviance, Pearson, bound-occupancy, and stability
   gates before a Poisson replacement can be drawn. This is a smooth
   interpolation stress truth, not a physical generator.

The two random streams are independent `PCG64` child streams of master seed
`20260806`. The source file, state, fit, seed, array, and output hashes are
recorded in `derived/input_provenance.json`.

## Inference card

Each lane uses the v4.2 2021 settings:

- scan range 50--250 MeV in 1 MeV steps;
- 0.625 MeV analysis bins (`neighborhood_rebin: 5`);
- local resolution-scaled RBF ceiling factor \(k_{\max}=15\);
- 12 optimizer restarts;
- profiled extraction with negative \(\hat A\) allowed;
- observed/asymptotic 90% CLs and local asymptotic \(p_0\);
- no expected limit bands and no toys.

Repeated unchanged-card scans are reviewed by maximum GP log-marginal
likelihood. A selected state is accepted only after another unchanged-card run
reproduces its LML, constant, and length scale within the declared tolerances.
Interpolation is forbidden. Pending masses are listed in
`derived/<lane>_repair_masses.txt`.

## Reviewed result

At 65 MeV, the original v4.2 2021 10% scan has
\(\hat A=28038.9\), \(\sigma_A=6609.53\),
\(p_0=1.05702\times10^{-5}\) (\(Z=4.25249\)), and an observed
90% CL upper limit \(\epsilon^2_{\rm up}=1.17184\times10^{-5}\).

The two conditional replacements give:

| Lane | \(\hat A\) | \(\sigma_A\) | local \(p_0\) | \(Z\) | observed 90% CL \(\epsilon^2_{\rm up}\) |
|---|---:|---:|---:|---:|---:|
| GP mean | -5125.82 | 6602.76 | 0.5 | 0 | \(2.77357\times10^{-6}\) |
| `fGenGammaThresh` | -2195.58 | 6602.95 | 0.5 | 0 | \(3.12419\times10^{-6}\) |

Thus, in both fixed-seed conditional counterfactuals the original positive
65 MeV local excess is absent; the fitted amplitude is negative and the
one-sided local test returns \(p_0=0.5\). This diagnoses the dependence of the
v4.2 65 MeV feature on the replaced central counts. It is not an ensemble
probability for background to reproduce the original fluctuation.

The functional sideband fit passed its declared gates with Poisson
deviance/ndf \(=1.06902\), Pearson \(\chi^2/{\rm ndf}=1.06899\), and
deviance \(p=0.24296\). Optimizer review closed all 201 masses in each lane.
Twelve GP-lane and eleven functional-lane mass points exhibited more than one
optimizer branch across the initial attempts; unchanged-card targeted repeats
reproduced every selected maximum-LML state. No kernel-bound-selected state
remained.

## Reproduction prerequisite and repo-relative commands

The cards, generated in-repository paths, and commands below are relocatable
with the repository. Input construction additionally requires the external
observed 2021 10% ROOT file at
`/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root`, with
SHA256
`3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4`
and histogram key `preselection/h_invM_8000`. That external prerequisite is
deliberately explicit and is not bundled here.

With that prerequisite present, run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/build_inputs.py

PYTHONDONTWRITEBYTECODE=1 \
  python3 study_results/v4p2_followups_20260806/pseudo65/validate_study.py \
  --stage inputs

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 -m hps_gpr.cli scan \
  --config study_results/v4p2_followups_20260806/pseudo65/configs/config_obsUL90_2021_10pct_gpmean_replacement_v4p2.yaml \
  --output-dir study_results/v4p2_followups_20260806/pseudo65/runs/gp_mean/attempt_01

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 -m hps_gpr.cli scan \
  --config study_results/v4p2_followups_20260806/pseudo65/configs/config_obsUL90_2021_10pct_funcform_replacement_v4p2.yaml \
  --output-dir study_results/v4p2_followups_20260806/pseudo65/runs/functional_form/attempt_01
```

Repeat both scan commands with `attempt_02` output directories, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 study_results/v4p2_followups_20260806/pseudo65/review_scans.py

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/postprocess.py

PYTHONDONTWRITEBYTECODE=1 \
  python3 study_results/v4p2_followups_20260806/pseudo65/validate_study.py \
  --stage final

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 -m pytest -q \
  study_results/v4p2_followups_20260806/pseudo65/tests/test_pseudo65.py
```

If review reports an unreproduced maximum-LML branch, rerun that mass with the
same YAML and `--mass-min M --mass-max M` into
`runs/<lane>/repairs/mMMM/attempt_NN`, then rerun `review_scans.py`.

## Main artifacts

- `inputs/pseudo65_background_replacements.root`: source copy, both
  pseudo-observed histograms, and both central expectations in one ROOT file.
- `derived/input_provenance.json`, `derived/functional_fit_qc.json`, and
  `derived/input_validation.json`: construction ledger and pre-scan gates.
- `derived/gp_mean_results_reviewed.csv` and
  `derived/functional_form_results_reviewed.csv`: optimizer-reviewed observed
  results.
- `derived/optimizer_audit.json` and per-lane optimizer ledgers: branch and
  repeat audit.
- `derived/m065_results_summary.csv`: exact original and replacement results at
  65 MeV.
- `plots/pseudo65_observed_limit_p0_aligned.{png,pdf}`: two-column by
  three-row spectrum/limit/local-\(p_0\) presentation figure.
- `plots/pseudo65_central_window_zoom.{png,pdf}`: central construction check.
- `plots/CAPTION.txt`: results-section/appendix-ready caption.

No analysis-note source is edited by this study.
