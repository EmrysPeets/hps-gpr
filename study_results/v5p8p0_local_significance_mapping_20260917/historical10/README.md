# Historical 2016 10% matched-policy comparison

This is a new v5.8.0 check using the actual historical ROOT histogram, kept distinct from scaling the full-data observation. Source SHA, selection/overlap caveats and engine identities are in `manifest.json`. The parent reviewed kernels, 30–210 MeV support, signal resolution and single-channel template convention are inherited. The GP mean and count-dependent errors are recomputed for each spectrum.

`python3 run_actual10.py` produces 142-point observed, scaled-parent-stress, mass-specific local-GP, and parent-full-stress signed-root scans. All 568 deterministic fits succeeded in 9.9 seconds. The historical 10% observation peaks at signed root 2.054 at 65 MeV. Its measured support-count ratio to full data is 0.10220158. Scaling the parent stress shape by this ratio reduces its root RMS from 4.967 to 1.607; this is a conditional common-shape exposure diagnostic, not proof of matched selection.

`python3 run_local_tests.py` generates 512 independent complete Poisson spectra under each of two truths at inherited diagnostic anchors 42, 66, 76, 90, 92 and 117 MeV. The seeds are `[580201610, mass_MeV, truth_id]`. All 6,144 fits succeeded in 64.6 seconds with one worker and one linear-algebra thread. `local_checkpoints/` retains counts, fitted roots, numerical diagnostics, exact deterministic truth and seed sequence. `actual10_local_tests.csv` includes two-sided 95% Clopper–Pearson binomial intervals.

| Mass [MeV] | Observed signed root | Direct local-GP tail | Direct scaled-stress tail |
|---:|---:|---:|---:|
| 42 | −0.423 | 1 exactly | 1 exactly |
| 66 | +2.011 | 14/512 = 0.02734 | 509/512 = 0.99414 |
| 76 | −0.836 | 1 exactly | 1 exactly |
| 90 | +0.861 | 75/512 = 0.14648 | 24/512 = 0.04688 |
| 92 | +1.259 | 49/512 = 0.09570 | 9/512 = 0.01758 |
| 117 | +0.577 | 157/512 = 0.30664 | 136/512 = 0.26563 |

The statistic is `q0=max(0,r)^2` and its tail includes equality. Consequently a nonpositive observed root has exact tail probability one by construction; the generic binomial interval in the CSV is merely the mechanical interval from 512/512. The nominal asymptotic display convention, recorded separately, instead gives 0.5 for such observations.

At 66 MeV the local-GP interval is [0.01503, 0.04545], whereas the stress interval is [0.98297, 0.99879]. This large difference remains despite the smaller exposure. Stress means at 42, 66, 76 MeV are −3.591, +4.372, −4.623, with widths 0.932, 0.958, 0.980. The raw nominal 5% rule rejects 511/512 stress backgrounds at 66 MeV. Thus near-unit fluctuation widths do not establish a centered discovery null, even at historical 10% counts.

All displayed probabilities condition on their stated generating shapes and frozen policies. Local-GP truths are separately derived at each mass and cannot be concatenated into one global null. There is no scan-global calibration, rare-tail extrapolation, independent selection matching, or claim that either generating construction is the physical continuum. Numerical replay at 92 MeV agrees with the prior v5.5.3 matched-policy result within 2.22×10⁻¹⁶. A separate output audit verified every tail count, all finite roots, and deterministic-anchor agreement with the 142-point scan.
