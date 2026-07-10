# V3 length-scale closure figures

These figures are the note-facing lower-length-scale scans selected for the
2026-07-09 analysis-note update.  They supersede the older fixed-configuration
closure summaries in the main text; those older summaries remain useful as
historical validation material.

## 2015

- Figure: `fig40_style_2015_lslb_closure_suite_100toy.png`
- Source:
  `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/fig40_style_2015_lslb_closure_suite_100toy.png`
- Data summary: `2015_highstat_ranking.csv`
- Scope: 100 toys per mass/strength cell, profiled-background lower-bound
  scan, primary `fShiftSigPowTail` functional form.
- Interpretation boundary: the scalar aggregate score ranks 1.1 slightly ahead
  of 1.0.  The frozen analysis value is 1.0, so the figure supports a stable
  1.0--1.1 neighborhood; it does not by itself demonstrate that 1.0 is the
  unique score optimum.

## 2016

- Figure: `fig40_style_2016_lslb_closure_suite_25toy.png`
- Source:
  `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2016_100toy_20260630/section_plots/fig40_style_2016_lslb_closure_suite_25toy.png`
- Data summary: `2016_highstat_ranking.csv`
- Scope: 25 toys per mass/strength cell, profiled-background lower-bound scan,
  primary `fShiftSigPowTail` functional form.
- Interpretation: 0.9 has the best aggregate calibration score of the tested
  0.9, 1.0, and 1.1 rows and reduces the coherent mean-pull bias relative to
  the tighter alternatives.  The folder name contains `100toy`, but this
  particular completed 2016 scan and its filename explicitly contain 25 toys.

## 2021

- Figure:
  `fig40_style_2021_search50_project30_refmatched_lslb_closure_suite_25toy.png`
- PDF companion is included.
- Source:
  `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/funcform_2021_search50_project30_fSigPowExpQ_upper9_refmatched_closure_20260702/plots/fig40_style_2021_search50_project30_refmatched_lslb_closure_suite_25toy.png`
- Data summary: `2021_highstat_ranking.csv`
- Scope: the corrected `final_1pct_invM.root` input, retaining the high-psum
  region after rejecting events with psum below 2.8 GeV; 30--250 MeV toy
  support, 50--250 MeV search range, matched background-only refit reference,
  upper length-scale factor 9, and 25 toys per mass/strength cell.
- Interpretation boundary: 0.9 and 1.0 have slightly lower aggregate scores,
  while the frozen 1.1 row has pull widths closest to unity and the strongest
  epsilon-squared extraction proxy.  The frozen 1.1 choice is therefore a
  calibration/reach tradeoff, not a claim that every ranking metric selects it.

All epsilon-squared quantities in these closure scans are extraction proxies,
not final CLs upper limits.  These figures validate estimator closure and the
length-scale choice; they do not replace direct CLs coverage at the frozen
production configuration.
