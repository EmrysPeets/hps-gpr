# Signal-robustness audit for v4.9.7

This self-contained audit product explains the change in the earlier 65 MeV
local excess when the 2021 native-10% GP support moved from 40--300 MeV (v4.2)
to 36--300 MeV (v4.9.5). The builder snapshots the archived ledgers, reconstructs
the four fixed-hyperparameter support x kernel cells, carries the full 201-point
old/new local-Z curves, and makes PDF/PNG figures.

## Result at 65 MeV

| quantity | v4.2 support40 | v4.9.5 support36 | change |
|---|---:|---:|---:|
| fitted amplitude Ahat [events] | 28038.923354 | 17100.853570 | -39.0% |
| amplitude uncertainty [events] | 6609.529989 | 7136.497770 | +8.0% |
| local asymptotic Z | 4.252493 | 2.402877 | -1.849616 |
| local asymptotic p0 | 1.0570175e-05 | 0.0081333211 | x769.5 |
| 90% asymptotic epsilon^2 UL | 1.1718399e-05 | 8.4208872e-06 | -28.1% |
| GP length scale in log(m) | 0.319475 | 0.274866 | -- |
| GP constant | 66.347717 | 52.088109 | -- |
| GP log marginal likelihood | 1648.806087 | 1676.039393 | +27.233 |

The observed ROOT file, mass resolution, density normalization, and conversion
K=A/epsilon^2 are unchanged at 65 MeV. The raw data feature therefore did not
"disappear." Rather, its fitted signal interpretation is not robust to this GP
support prescription.

## Mechanism supported by the audit

The fixed-state 2x2 table shows that this is not the mechanical effect of adding
four MeV of low-side support while holding the old GP state fixed. On the old
support geometry, swapping to the new kernel state already lowers Z from about
4.25 to 2.85. On the support36 geometry, holding the old state instead gives Z
about 4.74, whereas the support36 maximum-LML state gives 2.40. Each support
geometry prefers its corresponding diagonal kernel state in GP marginal
likelihood. This supports a mechanistic statement: the support change migrated
the preferred GP correlation structure, raising the inferred continuum through
the masked region and reducing Ahat.

On an identical 10,001-point grid over the nominal 65 MeV +/-2.25 sigma window,
the new diagonal GP mean is +1,949.4
events per 0.625 MeV target bin at 65 MeV and
+23,484.2 events
(+0.1594%) when integrated across the
common window. This is a fixed-state diagnostic; the profiled amplitude change
is not a one-bin subtraction.

## Scope of the earlier excess

The pre-optimization 2021-only maximum was at
65 MeV with local Z=4.252.
The v4.9.5 maximum is at 78 MeV with local
Z=2.810. The often-quoted pre-optimization
Z=4.657 at 65 MeV is the 2015+2021 pair, not the all-three combination; the old
all-three local value is Z=3.993. Controlled hybrids that replace only the 2021
block lower the pair and all-three values as recorded in
`derived/m65_scope_hybrid_diagnostic.csv`. Those hybrid points are not an official
combined scan.

## Branch and geometry checks

- The raw finalist support40 state and the accepted v4.2 max-LML state differ by
  only 0.091 fitted event and 0.000017 in Z at 65 MeV. The v4.9.5 repair ledger
  repairs 94, 152, and 212 MeV, not 65 MeV. Branch repair is therefore not a
  material explanation for the 1.85-Z shift.
- The exact table uses the dedicated accepted 65 MeV extraction summary
  (Z=4.252493351); the unmodified full-curve ledger gives
  Z=4.252492721 at that point. Their
  6.3e-07
  difference is a harmless ledger/refit-state microdifference, not a physics effect.
- The 2021 native bin width is 0.125 MeV and five-bin rebinning gives 0.625 MeV.
  Moving the lower support edge by 4 MeV shifts the coarse-bin phase by 0.25 MeV.
  The old 65 MeV fit window contains 16 bins spanning 60.000--70.000 MeV; the new
  window contains 15 bins spanning 60.375--69.750 MeV. The accepted v4.9.5 scan
  therefore does not isolate support extent from rebin phase. A fixed-anchor rebin
  audit is the appropriate follow-up if that separation is required.

## Claim boundary

Diagnostic only: p0 and Z are local/asymptotic. Support x kernel and combined-scope swaps are controlled counterfactuals, not a trials-corrected significance, coverage calibration, exclusion, or proof that the feature is signal or background. The v4.9.5 support choice itself did not minimize an observed
amplitude, p-value, or upper limit, but its 0.75 practical criterion was a
documented post-phase-1 amendment. The conditional support toys are source-recovery
diagnostics, not direct coverage.

## Products

- `derived/m65_2021_exact_comparison.csv` and `m65_2021_exact_changes.csv` -- exact accepted old/new values.
- `derived/m65_support_kernel_counterfactual.csv` -- four fixed-state support x kernel cells.
- `derived/m65_gp_common_grid.csv` -- direct old/new GP mean and pointwise standard deviation on a common grid.
- `derived/old_new_2021_local_z_curves.csv` -- complete 201-point archived curve comparison.
- `derived/key_mass_curve_comparison.csv` -- compact key-mass extract.
- `derived/m65_scope_hybrid_diagnostic.csv` -- 2021/pair/all-three single-mass scope check.
- `derived/m65_branch_lineage.csv` -- raw-finalist, accepted-v4.2, and v4.9.5 branch lineage.
- `derived/signal_robustness_summary.json` -- machine-readable interpretation and claim boundaries.
- `figures/*.pdf` and `figures/*.png` -- publication figures.
- `source_snapshots/` -- compact snapshots of every archived ledger used directly.
- `qa/pdf_render_qa.json`, `qa/semantic_validation.json`, and rendered PDF page PNGs -- QA evidence.
- `qa/artifact_manifest_sha256.csv` -- SHA-256 for every artifact except the manifest itself.

## Source provenance

| source | SHA-256 | role |
|---|---|---|
| `study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/derived/individual_observed_limits_reviewed_v4p2.csv` | `1e3e99fb7c0a171d6d496de87ac6664b485928042b2cede242dffab55e0cc410` | accepted v4.2 2021 observed local-Z/limit curve |
| `study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/note_figures/extractions_m065/observed_extraction_m065_fit_summary.csv` | `dc06707637511644e6bad06638451351a9995b9e363b4cfe0aeddcae18bf3c4f` | accepted v4.2 exact 65 MeV fit summary |
| `study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/standalone_pairwise_bands100_fixed/ul_bands_standalone_pairwise_100.csv` | `efa73576adae356d4805b7548a0bc14da4d4a2572fd6a19cc6404e7cd5386e47` | accepted v4.2 standalone and pairwise rows at 65 MeV |
| `study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/derived/combined_bands300_reviewed_v4p2.csv` | `8f4b37ff6a998e236c1ea959db56a76f21ce509c05f24c17675cef676fcbeadd` | accepted v4.2 all-three row at 65 MeV |
| `study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/derived/analysis/observed_2021_10pct_support036_300.csv` | `28e6a10b8633fc69c1bab62d32fe39417c42ac886ef27f74ca0c9aeb7cc620e9` | v4.9.5 2021 observed local-Z/limit curve |
| `study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/observed_scan/final/optimizer_repair_ledger.csv` | `8391d3f7ffbb9e6585f1a43de0bce03c9d27f0c132544201554fe4c3e654fb58` | v4.9.5 observed branch-repair ledger |
| `study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/inputs/frozen_v4p2_analysis_card.yaml` | `5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055` | frozen v4.2 analysis card |
| `study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/inputs/v4p9p5_observed_2021_10pct_support036_300_card.yaml` | `c66bf0debb582c6c868f64475cd4c3595e0a0bc39ad17a09fb450c627e2c7b1f` | v4.9.5 support36 observed card |
| `study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/inputs/source_2021_10pct.root` | `3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4` | common 2021 native-10% observed ROOT source |
| `study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/PROVENANCE.md` | `6bdb27a621a1c05bf1417d287a9348ad71551cf1cf7d9185df731f24d5666636` | v4.9.5 support-selection provenance |
| `study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/runtime_overlay/hps_gpr/gpr.py` | `1c83cae238e87a4e94928c97fb737947c22a3f88b16dfaf955d48ab6b4771dd5` | archived GPR implementation used for reconstruction |
| `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/finalist_k15_2021_10pct_combined100toy_20260803/derived/observed_2021_reviewed.csv` | `4b5d8df6e4e5f3d0cdf4bb21b19fcd5dc9f92c3fdff28d5968662ba6fcabad93` | read-only pre-v4.2-finalization 2021 curve used for the branch-lineage check |
| `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/finalist_k15_2021_10pct_combined100toy_20260803/derived/optimizer_repair_ledger.csv` | `6a28e060687663717d476e79d7d7dd01ac3d6006e917a1ecfedfb478e51d74ca` | read-only pre-v4.2-finalization targeted repair ledger |

The two absolute finalist paths above are read-only lineage inputs recorded by the
audit; the builder does not reread the primary dirty checkout. All substantive
curve and accepted-result inputs are either snapshotted here or available by
`git show HEAD:<path>` from the isolated release revision.

## Rebuild

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 study_results/v4p9p7_2016_support_combined_100toy_20260902/signal_audit/build_signal_robustness_audit.py
```

Semantic QA: `13/13` checks passed.
