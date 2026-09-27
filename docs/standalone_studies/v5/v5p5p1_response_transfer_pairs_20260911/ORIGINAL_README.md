# HPS-GPR v5.5.1: response transfer and dataset pairs

This six-page extension tests the v5.5.0 residual response at other fixed masses and every supported two-dataset combination. The underlying v5.0.4 and v5.5.0 packages are unchanged. Five vector figure/PNG pairs, complete numerical ledgers, solver/input snapshots and agent reviews accompany the note.

## Findings

- The all-three fits in the selected 90–94 MeV region retain slopes near -3. Other pair combinations within that region can be unstable. At 92 MeV the best beta is -2.904 and its connected Delta(2NLL)<=1 profile span is [-3.422,-2.391]. Nearby masses are correlated, not independent confirmations.
- The original slope does not transfer uniformly. At 66 MeV the all-three best slope is -1.130; the transferred slope has penalty 4.536 versus 2.845 for a common rate. The 2015+2021 pair at 65.5 MeV prefers -1.048; the 2016+2021 pair at 66 MeV prefers +1.370.
- The 2016+2021 pair prefers -4.662/-4.635 at 121/160 MeV, with broad profile shapes. The transferred -2.904 slope costs only 1.022/0.927 in twice NLL relative to independent nonnegative rates.
- Two positive rates determine an amplitude and slope exactly. All 23 applicable positive pairs saturate numerically; small best-fit penalties alone are not evidence for a power law.
- At 92 MeV, the 2016+2021 pair predicts a 2015 equivalent amplitude of 140.74e-6, compared with 77.48+-29.96e-6. Its fixed plug-in penalty is 4.450; a refit using all three datasets gives 0.157. The latter uses the third dataset and is not a held-out prediction.

## Interpretation

Every model acts on the inherited response K_i at one fixed mass: mu>=0 times K_i times (E_i/2.3)^beta. The extra power describes residual response AFTER the existing prompt-density/radiative-fraction conversion, not coupling running or a derived production model. Beam energy remains confounded with detector/target/selection changes.

Candidate selection uses parent pair/all-three local maxima and resolution separation before examining new slope fits. Twelve distinct candidates plus inherited controls give 17 masses, 56 supported scope–mass comparisons, 47 standalone fits and 39 unused-dataset checks. Scopes and neighboring masses overlap; no absolute NLLs are pooled or ranked across different windows. The candidate set is not a new optimized search for a different production law.

Seven fits have zero amplitude and unidentified beta; 17 reach the wide [-12,12] bound; 32 have censored profile spans. Unit-deviance spans are diagnostic shapes, not calibrated confidence intervals. Pair predictions omit pair-parameter uncertainty. The frozen slope already used all three datasets at 92 MeV, so its red prediction controls there are not independent validation. Full-data mass selection further precludes an external-validation interpretation.

Four free pair extrapolations (2016+2021 at 52, 68.5, 90, 90.5 MeV predicting 2015) yield nonpositive profiled backgrounds under the inherited Gaussian constraint model. They are flagged by `plugin_valid_positive_background=False` and retained only for audit. Do not interpret their numerical D values physically. All reported optimized/common/transferred response models and all displayed 92 MeV extrapolations have positive backgrounds and means.

## Files and rebuild

- `pdf/HPS_GPR_v5p5p1_Response_Transfer_and_Pairs.pdf`: final note.
- `figures/response_transfer_examples.pdf`: fitted rates at representative masses.
- `figures/all_pair_candidate_slopes.pdf`: all pair candidates plus 92 MeV.
- `figures/all_three_transfer_summary.pdf`: complete all-three transfer comparisons.
- `figures/beta_profiles_92.pdf`: standalone profile detail for all four scopes at 92 MeV.
- `figures/unused_dataset_predictions_92.pdf`: pair-only predictions and all-data-slope controls.
- `derived/response_fits.csv`, `beta_profiles.csv`, `pair_heldout.csv`: complete fits, profiles and prediction checks.
- `derived/heldout_comparisons.csv`: adds refitted compatibility, explicitly using third-dataset data.
- `derived/candidate_selection.csv`, `selected_pair_summary.csv`: selection and retained candidates.
- `physics_review.md`, `candidate_audit.md`, `qa/agent_reviews.json`: independent consultation and review.

Run `bash scripts/build.sh` from this directory with Python (numpy/scipy/pandas/matplotlib) and Tectonic available. Numerical work uses one thread and no random draws. `python3 scripts/validate.py` extracts/renders the PDF and verifies semantics, source hashes and numerical records. `python3 scripts/package.py` verifies an isolated TeX build, writes the manifest and creates the output PDF/ZIP. These last tools require PyMuPDF and Pillow. The package manifest excludes only itself; final delivery checksums live beside the ZIP.

The copied scopes.json retains the parent's earlier 90 MeV endpoint, which inherited common.py explicitly extends to 100 MeV. No 2015 fits are created above 100 MeV. The parent 2015 extension and 2016 background qualifications remain unchanged. No commit, push or external publication was made.
