# Deterministic functional-mean shape-bias diagnostic

This isolated pseudo65 follow-up tests whether the stored
`fGenGammaThresh` interpolation itself induces a 61--63 MeV positive response
when analyzed with the unchanged v4.2 2021 GP inference card.

## Inputs and statistical meaning

The diagnostic ROOT file copies the observed 2021 10% histogram exactly, then
replaces the 80 native bins in `[60,70)` MeV with either:

1. the already validated fractional `fGenGammaThresh` expectation; or
2. the already validated fractional fixed-GP expectation.

The observed counts outside `[60,70)` MeV remain bitwise identical. These
hybrid spectra are deterministic central-mean, Asimov-like shape probes. They
are **not observed datasets**, complete Asimov datasets, pseudoexperiments, or
an ensemble. Running the card's local asymptotic extraction on fractional
counts is used only to quantify the GP response to the two central shapes.

For comparison, the final CSV and plot also use the checksum-locked,
optimizer-reviewed functional and GP-mean **single Poisson draws** from the
parent pseudo65 study. Their deviations from the deterministic curves include
the particular binwise fluctuation in each draw.

No expected limit, limit band, toy calibration, coverage statement, global
`p`-value, or probability that an interpolation creates a shoulder is
reported.

## Frozen card

Both deterministic lanes are generated from
`pseudo65/configs/config_obsUL90_2021_10pct_funcform_replacement_v4p2.yaml`.
The generated YAML files differ only in:

- `path_2021`;
- `hist_2021`; and
- `output_dir`.

Thus the 0.625 MeV analysis bins, resolution-scaled training and extraction
geometry, 2021 length-scale ceiling factor 15, 12 optimizer restarts,
negative-`Ahat` extraction, and local asymptotic calculation are unchanged.
Expected bands and toys remain disabled.

## Reproduction

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 study_results/v4p2_followups_20260806/pseudo65/functional_mean_shape_bias_20260806/build_inputs.py

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 -m hps_gpr.cli scan \
  --config study_results/v4p2_followups_20260806/pseudo65/functional_mean_shape_bias_20260806/configs/config_functional_mean_shape_bias.yaml \
  --output-dir study_results/v4p2_followups_20260806/pseudo65/functional_mean_shape_bias_20260806/runs/functional_mean/attempt_01 \
  --mass-min 0.055 --mass-max 0.075

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 -m hps_gpr.cli scan \
  --config study_results/v4p2_followups_20260806/pseudo65/functional_mean_shape_bias_20260806/configs/config_gp_mean_shape_bias.yaml \
  --output-dir study_results/v4p2_followups_20260806/pseudo65/functional_mean_shape_bias_20260806/runs/gp_mean/attempt_01 \
  --mass-min 0.055 --mass-max 0.075
```

Repeat the two scan commands with `attempt_02`, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/functional_mean_shape_bias_20260806/review_and_plot.py

PYTHONDONTWRITEBYTECODE=1 \
  python3 study_results/v4p2_followups_20260806/pseudo65/functional_mean_shape_bias_20260806/validate.py
```

The review selects the maximum finite GP log-marginal-likelihood state at each
mass and requires a second unchanged-card attempt to reproduce its LML,
constant, and length scale. Interpolation is forbidden.

## Artifacts

- `inputs/deterministic_central_means.root`: both fractional central-mean
  hybrids, the source histogram, copied stored expectations, and semantic
  metadata.
- `derived/input_provenance.json`: source/config hashes and array-level
  construction checks.
- `derived/functional_mean_results_reviewed.csv` and
  `derived/gp_mean_results_reviewed.csv`: reviewed direct pipeline outputs over
  55--75 MeV.
- `derived/comparison_55_75MeV.csv`: aligned deterministic-mean and
  single-Poisson-draw `Ahat`, uncertainty, local `p0`, and local `Z` values.
- `MEMO.md`: compact quantitative answer with interpretation boundaries.
- `plots/functional_mean_shape_bias_Ahat_p0.{pdf,png}`: aligned `Ahat` and
  local-`p0` diagnostic.
- `qa/validation.json`: final fail-closed input, card, optimizer, numerical,
  provenance, and rendering validation.

No existing pseudo65 artifact, GP ensemble, or analysis-note source is edited
by this diagnostic.
