# v4.9.16: 2015 low-mass rising-edge side study

The requested 15--20 MeV scan uses a local 12--28 MeV GP support, comparisons with nearby supports, a local exp(Chebyshev-5) background check, and signal extractions at 15, 17, 17.25 and 20 MeV. The 20--22 MeV bridge is context only. This is an exploratory extension, not a newly qualified production search.

The nominal ceiling-eight GP gives its largest upward fluctuation at 17.25 MeV, r = 1.430 and local asymptotic p0 = 0.0764. It reaches a kernel bound. Ceilings 16, 32 and 64 recover stable, interior optima; at 17.25 MeV the ceiling-16 root is 1.577 and p0 = 0.0574. The polynomial cross-check gives p0 = 0.254 there. None establishes a heavy photon. Resolution and accepted signal shape below 19 MeV remain unqualified.

There are 100 conditional Poisson toy fits at each displayed mass for each of ceilings eight and 16 (800 total). Each count includes the first ten pilot fits. Every toy retrains its sidebands and optimizes the GP again. The 17.25 MeV toy tails contain only 2/100 and 1/100 exceedances respectively, with wide binomial intervals and shifted null roots. These are mass-local, plug-in procedural checks; they do not constitute a global p-value, independent physical background validation, or final calibration.

## Deliverables

- `output/pdf/v4p9p16_2015_lowmass_side_study_20260906/HPS_GPR_v4p9p16_2015_LowMass_Section.pdf`: five-page standalone addition.
- `output/pdf/v4p9p16_2015_lowmass_side_study_20260906/HPS_GPR_v4p9p16_with_2015_LowMass_Study.pdf`: augmented v4.9.16 report, including the ongoing presentation/extraction section captured at build time; the low-mass addition is Section 11, pages 24--28 in this snapshot.
- `note/lowmass_section.tex`: embeddable LaTeX section; see `INTEGRATION.md`.
- `figures/`: five reusable PDF/PNG figures.
- `derived/scan.csv`, `derived/kernel_stability.csv`, `derived/fits/`: complete numeric scans and fit arrays.
- `derived/toy_roots*.csv`, `derived/toy_summary*.json`, `derived/toy_truth*.csv`: restartable conditional toy results and their means.
- `derived/display_mapping*.npz`: whole-bin maps and display count/GP covariances.
- `qa/numerical_validation.json`: direct saved-array likelihood, fit, toy and document checks.
- `qa/render_section/`, `qa/render_full/`, `qa/visual_review.json`: rendered-page QA.
- `provenance/`: input, numerical-source and inherited-report identities. Inherited figures are copied into `inherited_figures/`, so this report is insulated from subsequent edits to the active parent.

All writes for this side study are confined to this folder and its own PDF output directory. No source ROOT file, production card, parent report, git index, branch or remote was changed.

## Reproduce

From the repository root, with the existing Python/numpy/scipy/pandas/scikit-learn/uproot/matplotlib/pypdf and Tectonic environment:

```bash
python3 -B study_results/v4p9p16_2015_lowmass_side_study_20260906/run_study.py
python3 -B study_results/v4p9p16_2015_lowmass_side_study_20260906/check_stability.py
python3 -B study_results/v4p9p16_2015_lowmass_side_study_20260906/run_study.py --toys 100
python3 -B study_results/v4p9p16_2015_lowmass_side_study_20260906/run_study.py --toys 100 --upper-factor 16
python3 -B study_results/v4p9p16_2015_lowmass_side_study_20260906/make_report.py
python3 -B study_results/v4p9p16_2015_lowmass_side_study_20260906/verify_study.py
```

`run_study.py --toys N` resumes existing mass/toy IDs; changing N is not a new ensemble. The toy streams are deterministic; the two methods use their own generating means. To run fresh validation after changing the pipeline, use a new derivative and seed namespace instead of mixing results. The checked-in pilot CSVs preserve the original ten-fit stage. Do not overwrite sealed products without first creating a new derivative.

`make_report.py` captures the latest available presentation-extraction report, falling back to the deficit report if necessary. Exact captured sources are listed in `provenance/report_parent.json`; re-running later may intentionally capture a newer parent. Tectonic uses cached resources only. The numerical code imports the attested campaign runtime, not the modified working `hps_gpr` package.

## Next scientific step

Use the actual event selection to validate accepted signal templates, mass scale, resolution and the acceptance turn-on in 15--20 MeV. Compare independent control predictions and smooth alternative generating truths, including signal injections. Freeze support and background rules using predictive diagnostics. A final discovery assessment needs a coherent full-spectrum ensemble and explicit accounting for the mass scan and model choices; this small side study deliberately supplies local diagnostics only.
