# HPS-GPR v6.2: native smeared-MC injection and recovery

This study draws exactly 1,000, 5,000, 10,000 or 30,000 selected MC candidates and injects them into 20 background toys at each available native mass from 60 through 260 MeV. The 40 MeV sample is excluded. The 880 injected spectra receive paired pole-centered and MC-core-shifted extractions, for 1,760 primary fits. The 220 paired zero-signal spectra and simulation-only controls bring the total to 6,160 toy extraction fits. Another 308 deterministic expected-count (Asimov) fits diagnose bias without random fluctuations.

## Finding

Moving the fit/training window to the fixed MC core improves recovery in this ensemble, but does not eliminate under-recovery. At 30,000 injected candidates, raw mean recovery is 68.5–98.3% for pole-centered windows and 75.2–104.2% for shifted windows, including the labeled 260 MeV extension. The known-background control gives 95.9–102.9%. Subtracting the fitted yield of the paired zero-signal toy gives incremental recovery of 71.5–89.5% and 78.0–93.8%, respectively. That subtraction is a diagnostic; it is not applied to the primary yields, pulls or intervals.

| Mass (MeV) | Pole recovery | Shifted recovery | Pole pull mean / width | Shifted pull mean / width | Pole nominal95% contains truth | Shifted nominal95% contains truth |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 68.5% | 75.2% | -0.49 / 1.23 | -0.35 / 1.10 | 18/20 | 19/20 |
| 80 | 98.3% | 104.2% | -0.06 / 0.88 | 0.13 / 0.85 | 20/20 | 20/20 |
| 100 | 81.2% | 88.3% | -0.81 / 0.75 | -0.47 / 0.80 | 20/20 | 20/20 |
| 120 | 88.8% | 94.8% | -0.59 / 1.10 | -0.26 / 0.99 | 18/20 | 18/20 |
| 140 | 88.9% | 93.1% | -0.70 / 0.91 | -0.42 / 0.87 | 19/20 | 19/20 |
| 160 | 86.0% | 91.9% | -1.06 / 0.95 | -0.58 / 0.93 | 17/20 | 19/20 |
| 180 | 86.9% | 91.6% | -1.15 / 1.05 | -0.71 / 0.93 | 16/20 | 19/20 |
| 200 | 86.9% | 90.2% | -1.28 / 0.65 | -0.93 / 0.68 | 17/20 | 19/20 |
| 220 | 86.8% | 90.2% | -1.43 / 1.09 | -1.04 / 1.09 | 15/20 | 17/20 |
| 240 | 83.7% | 85.3% | -1.87 / 0.74 | -1.64 / 0.75 | 7/20 | 9/20 |
| 260* | 82.7% | 85.3% | -1.93 / 0.90 | -1.59 / 0.77 | 10/20 | 13/20 |

*260 MeV is an explicit extension beyond the inherited 50–250 MeV extraction grid. It uses the archived 250 MeV kernel and the nominal resolution evaluated at 260 MeV. It is not a newly validated scan endpoint.

The deterministic 240 MeV, 30k case isolates the mechanism: contaminated-sideband fits return 24,023 and24,859 candidates for pole and shifted windows, while clean-sideband fits return 29,743 and29,724. Known-background fits return 30,000. The broader selected-MC distribution enters external training bins, raising the GP background prediction and reducing the extracted signal.

The low-injection recovery ratios can be far from one because the injection is small compared with background fluctuations and any background-model offset. Inspect fitted-yield errors, paired increments and null controls together. Twenty toys give imprecise pull widths and containment fractions: tables retain binomial Clopper–Pearson 95% intervals for the latter. No physical coverage calibration is claimed.

## Exact procedure

- Dataset/background: the inherited 2021 10% spectrum and the pinned v5.8.2 all-data GP mean on 36–299.75 MeV. The generating GP uses its archived 76 MeV source kernel. One positive generating mean is used for every mass, method and injection level. New independent Poisson background draws are made per mass/toy, then reused across injection levels and methods.
- MC: native 0.1 MeV histograms from `/sdf/data/hps/physics2021/preselection/v13/ap_signal_prompt_smeared`. Each exact-N draw uses a multinomial over CDF-rebinned analysis bins plus below/above-support categories. This is equivalent to sampling the histogram with uniform locations inside source bins. Recorded histogram overflow remains outside analysis support. Signal draws are independent across injection levels and toys; methods share the same draws.
- Yield: the fit parameter is full-selected candidate yield. The signal vector is the bin probability per selected candidate and is never normalized within the fitted window. Actual signal counts in the support, window, training bins and outside support are saved. These candidates are not certified truth-associated signal events.
- Comparison: pole windows are `m +/- 2.25 sigma_nominal(m)`; shifted windows are `c_MC(m) +/- 2.25 sigma_nominal(m)`. The same MC shape is used unchanged. The MC-only center and width convention are inherited from v6.1; centers are fixed for all toys. No event distribution is translated and no mass interpolation is used.
- GP: both mean and correlated covariance are recomputed from each toy's exterior bins, updating `log(n)` targets and `1/n` noise. Archived kernel hyperparameters are fixed. This is conditional per-toy GP retraining, not hyperparameter reoptimization. Both mask definitions update their exterior bins.
- Likelihood: inherited Poisson bin counts plus correlated Gaussian GP-background nuisance penalty. The free yield can be negative when the Poisson means remain positive. Pulls use `(Ahat-N)/sigma_A`, with observed profile-Hessian errors. LR containment evaluates the profiled likelihood at true N against the unconstrained minimum, with thresholds 1 and 3.841459 for nominal 68.27% and 95% sets. Wald endpoints are saved separately and labeled.
- Controls: zero injection on each shared background; clean sidebands trained only on that toy's background component; known true background with no nuisance covariance. The latter two use unavailable truth information and diagnose the simulation machinery. They are not production extraction methods.

The injected signal has an exact total and multinomial covariance, while the inherited extraction likelihood is Poisson. Unit pull width and nominal LR containment are therefore references, even for the oracle. Template statistics, MC-center uncertainty, detector-response uncertainty and background-truth uncertainty are not propagated. The supplied v13 MC/v16 data selections and daughter association are not fully certified. Results concern recovery under this specified toy model, not efficiency, exclusions or discovery.

## Products and reproduction

- `pdf/HPS_GPR_v6p2_MC_Injection_Recovery.pdf`: report with recovery, pull distributions, paired comparisons, controls and all primary-cell summaries.
- `results/toys.csv`: all signed yield/error/pull/LR fits with exact injection bookkeeping.
- `results/summary.csv`: per-cell bias, spread, recovery, pull moments and containment counts with binomial intervals.
- `results/paired.csv`: paired shifted-minus-pole differences and null-subtracted incremental response.
- `results/asimov.csv`: deterministic expected-count control fits.
- `results/checkpoints/`: per-mass raw background/injection arrays, masks, probabilities, fit rows and checksummed resumable completion records.
- `protocol.json`, `provenance/`, `inputs/`: fixed design, copied inputs, upstream ROOT identities and executable dependencies.
- `qa/validation.json`, `qa/independent_review.md`, `qa/independent_validation.json`: numerical and independent checks. Rendered-report checks are separate.

Run from any directory with a Python environment satisfying `requirements.txt`:

```sh
python3 /path/to/v6p2_mc_injection_20260923/scripts/reproduce.py
```

The launcher caps execution at four local workers with one numerical thread each and has a 30-minute process-group watchdog. Each completed mass is reused only when dependency and output checksums match. A `STOP` file at the study root stops work at the next toy boundary. Native histograms and fit matrices were already local; no S3DF login-node or compute-node jobs were needed. Scientific computation finished in a few seconds; no production inputs or v6.1 outputs were modified.
