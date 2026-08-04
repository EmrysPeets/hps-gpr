# Version 4 wide-support combined campaign

## What changed

The 2021 endpoint comparison made a long-standing fit convention visible
again: the interval used to train the Gaussian process should extend beyond
the interval in which resonance hypotheses are tested.  The recent combined
card already did this for 2021, but used the search interval itself as the
2015 and 2016 fit support.

Version 4 keeps every tested mass and all other frozen likelihood settings
unchanged, while using:

| Dataset | Data fraction | Signal-search interval | GP fit support |
|---|---:|---:|---:|
| 2015 | 100% | 19--90 MeV | 14--135 MeV |
| 2016 | 100% | 39--180 MeV | 30--210 MeV |
| 2021 | 10% | 50--250 MeV | 40--300 MeV |

The 2016 lower edge deserves a little explanation.  A nominal 35 MeV lower
support still leaves no rebinned training-bin center below the exclusion at
the 39 MeV search endpoint.  That trial was stopped before it reached the
physics result and is retained under
`aborted_support035_no_low_sideband/`.  Moving the support edge to 30 MeV
restores 20 lower-side training bins at 39 MeV.  The final card has populated
training bins on both sides at every endpoint: the minimum lower/upper counts
are 12/138 for 2015, 20/51 for 2016, and 9/55 for 2021.

The frozen configuration is
`../../study_configs/v4_wide_support_2015full_2016full_2021_10pct_20260803/config_obsUL90_combined_wide_support_v4_observed_only.yaml`.
Its SHA-256 is
`16f686602514c5e156a8da83ed4f5facc1027788e184ef32ff72313f3fadd2a3`.

## Inputs

| Dataset | ROOT input | Histogram | SHA-256 |
|---|---|---|---|
| 2015 | `/Users/emryspeets/research_plots/2015_data/invariant_mass_0pt5mm_full.root` | `invariant_mass` | `58ce717cde753d8566c754a73cb056560ed19e781fe9a43e8634111cc746531f` |
| 2016 | `/Users/emryspeets/root_files/EventSelection_pass4Full.root` | `h_Minv_General_Final_1` | `c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301` |
| 2021 | `/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root` | `preselection/h_invM_8000` | `3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4` |

These run-host paths are provenance, not portable data dependencies committed
to Git.  A rerun should provide the collaboration-controlled ROOT files at
equivalent paths or update a copied configuration, then verify the hashes
above before treating the result as the same analysis state.

## Observed-state review

The same card was scanned three times.  Each attempt contains the complete
415-state grid: 72 fits for 2015, 142 for 2016, and 201 for 2021.  At each
dataset and mass, the exact maximum-log-marginal-likelihood input row was
retained only when another unchanged-card attempt reproduced its LML within
`2e-5`.

All 415 states passed.  Of these, 405 reproduced in all three attempts and 10
reproduced in two; no state is interpolated or unresolved.  The authoritative
coordinates are in `derived/observed_gp_states_reviewed.csv`, and the complete
three-attempt comparison is in `derived/observed_attempt_ledger.csv`.

The review also leaves an important diagnostic in plain sight: all 142
reviewed 2016 fits land on the frozen length-scale upper bound.  The bands
below are therefore conditional results from the declared card, not evidence
that support-matched functional-form closure or direct coverage has already
been established.

## Conditional 300-toy ensemble

The combined campaign covers the full 19--250 MeV grid.  It uses one dataset
from 19--38 MeV, two from 39--49 MeV, all three from 50--90 MeV, two from
91--180 MeV, and 2021 alone from 181--250 MeV.  The production table contains
232 rows and exactly 300 finite background-only pseudoexperiment limits per
row, for 69,600 joint pseudoexperiments in total.

The observed GP hyperparameters are fixed at the reviewed coordinates.  Each
pseudoexperiment draws a conditional GP-posterior background intensity,
Poisson-fluctuates the active spectra with a common toy index, and evaluates
the same asymptotic 90% CL `tilde_q_mu` CLs limit.  The GP is not retrained on
the pseudo-spectra.  The resulting median and central 68%/95% intervals are
conditional expected-limit bands, not a full-procedure coverage study.

The campaign-local profile cache was checked against the ordinary solver at
representative one-, two-, and three-dataset masses.  Every observed and toy
limit in that check agreed bit for bit.  The closure report is
`derived/cached_profile_closure_v4.json`; the final ensemble is
`combined_bands_300toy_cached/ul_bands_combined_all.csv`.

## Empirical limit-consistency tails

The definitions are the same as in version 3:

```text
p_strong = number(toy UL <= observed UL) / 300
p_weak   = number(toy UL >= observed UL) / 300
p_two    = min(1, 2 * min(p_strong, p_weak))
```

The smallest strong-limit tail is 0 of 300 at 59, 71, and 72 MeV.  The
smallest weak-limit tail is 0 of 300 at 21, 65, and 66 MeV.  The smaller
one-sided count, and hence the bounded two-sided value, is 0 of 300 at 21,
59, 65, 66, 71, and 72 MeV.  These are unresolved empirical tails at a
one-toy resolution of `1/300 = 0.00333`; they are not measurements of a zero
probability.

The smallest fixed-mass asymptotic discovery value is
`p0 = 1.76364e-4` (`Z = 3.573`) at 66 MeV.  The separate 2.25-sigma
resolution-spacing estimate gives `N_eff = 35.3814` and an analytic Sidak
value of `0.006221` (`Z = 2.499`) at that local minimum.  This Sidak number is
not derived from the 300 expected-limit toys and is not a toy-calibrated
global discovery probability.

## Effect of widening the supports

The matched narrow-support finalist uses the same full-2015, full-2016, and
2021-10% inputs and the same kernel-bound factors, so its observed curve is a
useful diagnostic comparator.  Across the full grid, the wide/narrow
observed-limit ratio has a median essentially equal to one, but ranges from
0.235 at 19 MeV to 2.257 at 43 MeV.  The changes are concentrated in the
low-mass 2015 and 2015+2016 blocks.  In the 2021-only block, whose support did
not change, the ratio remains within about `6e-5` of unity.

This comparison is not used to choose the support or to claim an improvement:
the v4 domains were declared before the new observed scan.  It shows why an
endpoint support decision is part of the statistical model and why the next
closure tests must use the same widened domains.

## Interpretation boundary

This bundle supports a reviewed observed/asymptotic curve and a conditional
fixed-GP pseudoexperiment ensemble.  It does not yet provide:

- GP-refit pseudoexperiment coverage for the full procedure;
- a scan-wise maximum-`q0` ensemble for a calibrated global discovery
  probability;
- support-matched functional-form closure for the saturated 2016 kernel card;
- the final unblinded 2021 result.

Those statements remain separate in the version 4 note rather than being
inferred from the successful parser, optimizer review, or conditional band
production.

## Review bundle

The mass-by-mass table used in the note is
`derived/combined_bands300_reviewed.csv`.  Its companion
`derived/combined_bands300_summary.json` records the ensemble semantics,
tail-count minima, kernel-bound occupancy, matched-support comparison, and
input hashes in one place.  The five note-ready figure pairs are under
`note_figures/`, and `postprocess_combined_bands300.py` regenerates those
tables and figures from the reviewed ledgers without fitting a GP or throwing
another toy.

The editable note source is
`../../hps_gpr_analysis_note/main.tex`.  From that directory, the review PDF
is built with:

```bash
tectonic --keep-intermediates --keep-logs main.tex
```

The dated review PDF is deliberately kept in the local pre-publication backup
rather than versioned in Git.  The note source, figure assets, campaign
configuration, reviewed tables, and provenance needed to rebuild it are all
part of the publication branch.
