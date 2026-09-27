# HPS GPR v6.4.4: independently calibrated local-to-global appendix

This revision appends the local-first global-significance comparison to the v6.4.3 study. The original raw-statistic results remain as labeled references. The report explains why choosing the strongest individual-year result is a different question from the joint common-coupling search; that optional dataset-choice test is not used for the combined result.

The 2016 fit and GP-training exclusion remain [-2.5,+2.5]u; 2021 remains [-4,+3]u; 2015 retains its Gaussian. At each generated mass the combined likelihood has exactly one psi=epsilon²/1e-8, with campaign signal expectations psi*K_y(m)*P_yi(m). K_y is the inherited fixed conversion and P_yi is the full selected template probability. Each campaign has independent background nuisance parameters. This is verified numerically by comparing the joint profile to the sum of campaign profiles at the same fixed psi. The search does not sum separate best-fit significances or fit independent signal couplings for each year.

## Two ensembles with different roles

- A: the original 1,024 complete scans, unchanged, fix the local map p_A(q)=(1+number of A values at least q)/1025 at every mass.
- B: 1,024 independent new complete scans with a distinct random-seed namespace. Apply exactly the same frozen maps to observations and B; select the minimum p across the declared mass grid. Count B minima less than or equal to the observed minimum, including ties at the empirical-map floor.
- Only the unchanged raw-maximum statistic is also calibrated with all 2,048 scans pooled. The local-first global result uses 1,024 B trials, conditional on the fixed A maps; B never updates those maps.

Independent-grid Sidak uses N=136 for 2016 and N=181 for 2021 and combined. It is a labeled independence reference, not an assumption that neighboring fits are independent. A second Sidak curve uses one effective N per search, estimated only from A leave-one-out local ranks at fixed probability thresholds 0.005,0.01,0.02,0.03,0.05. Its log-survival least-squares policy was frozen before examining B. It is an approximation checked against B, not a replacement for the direct B calibration. Threshold-dependent effective counts from B are descriptive and include conditional finite-simulation intervals.

The local calibration has resolution 1/1025. In particular, zero A exceedances means a finite rank floor, never a zero underlying tail. The B confidence intervals condition on the chosen A maps and archived null sources; they do not propagate calibration-map uncertainty, source-estimation uncertainty or previous method/window choices. All grids have 1 MeV spacing, with the same campaign availability as v6.4.3.

## Main appendix results

| Search | Selected mass (MeV) | B count | Direct global p | Conditional 95% interval | Global Z | Fitted Sidak p | Independent-grid Sidak p |
|---|---:|---:|---:|---|---:|---:|---:|
| 2016 | 91 | 89/1024 | 0.088 | [0.070, 0.106] | 1.35 | 0.068 | 0.233 |
| 2021 | 67 | 231/1024 | 0.226 | [0.200, 0.252] | 0.75 | 0.152 | 0.412 |
| Combined | 68 | 75/1024 | 0.074 | [0.058, 0.091] | 1.45 | 0.055 | 0.162 |

All three fitted Sidak predictions fall below the corresponding direct B pointwise 95% intervals at the observed minima. The fixed effective count therefore understates these global probabilities; these comparisons do not constitute a simultaneous curve test. All three predictions extrapolate below the A-only fit range. The combined minimum reaches the A-map floor; B includes every tie at that floor. The pooled 2,048-scan raw-maximum p-values are 0.065, 0.232 and 0.191, respectively. They calibrate a different mass ordering and are retained as comparison results.

## Rebuild this revision

Use Python 3.9 and requirements.txt, plus Tectonic with cached article/LatinModern packages. Two local workers and one numerical thread each are enforced. No remote computation or downloads occur.

```bash
STUDY_PYTHON=/path/to/scientific/python3 bash rebuild_appendix.sh
STUDY_PYTHON=/path/to/scientific/python3 bash rebuild_appendix.sh --fresh-validation
```

The default reuses verified B checkpoints, rebuilds all appendix analyses and figures, repeats the shared-coupling audit and numerical validation, and generates pdf/report.pdf. The fresh option regenerates only B. A is deliberately preserved so this remains an independent validation of exactly the same calibration maps. The original rebuild.sh is a historical v6.4.3 workflow; use rebuild_appendix.sh for this report.

## Records

- provenance/validation_protocol.json: independent generation, engine and A-array hashes.
- provenance/calibrated_analysis_protocol.json: local maps, inclusive ties, Sidak fitting policy and uncertainties.
- results/sidak_fit.json: fixed A-only effective counts and fitting points.
- results/global_validation_*.npz: complete new B arrays; results/validation_checkpoints: resumable B mass checkpoints.
- results/calibrated_validation_*.npz: every frozen-map rank, with observed ranks and B minimum ranks.
- results/calibrated_summary.csv: raw-maximum and local-first results, peaks, counts, intervals and Sidak comparisons.
- results/calibrated_curves.csv: all mass-dependent comparison curves.
- results/threshold_comparison.csv: independently measured global tails and Sidak comparisons as functions of local threshold.
- results/validation_minima.csv: every B minimum and its displayed mass.
- qa/common_coupling_audit.json and common_coupling_profiles.csv: common-parameter numerical factorization.
- qa/calibrated_validation.json: independent rank counting, seed regeneration, fit replays and parent preservation.
- source/parent_v643_report.tex and provenance/parent_v643_report.pdf: original report preserved.

The full selected templates and old numerical results are unchanged. Original inputs and methods are documented in provenance/parent_v643_README.md. Verify the distributed package with `shasum -a 256 -c MANIFEST.sha256` before a rebuild, which updates timing records. PDF text and rendered pages are checked for each release.
