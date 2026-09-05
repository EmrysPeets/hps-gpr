# Deterministic reverse-injection check

Follow-up to the 20-spectrum 2021 pilot. No random numbers, new toys, or
reruns of the original toy ensemble are permitted in this script.

Predeclared design:

- Use the exact v4.9.12 2021 rebinned counts, 36–300 MeV support, factor-15
  reviewed per-mass kernel states, and production signal shape/resolution.
- Construct one smooth deterministic generating background from the observed
  data outside **60–86 MeV**, using the reviewed 66 MeV kernel without
  hyperparameter optimization. The larger exclusion protects both candidate
  regions; the first pilot's 60–78 MeV mask did not protect the 80 MeV region.
- Compare background only and three separate positive-only injections at
  **66, 78, and 80 MeV** on that same background. Each injection uses its own
  saved standalone 2021 best-fit full-template yield. These amplitudes and
  locations are data-selected. They are not fitted to this response study.
- Scan each deterministic mean spectrum at integer hypotheses 60–88 MeV,
  refitting the GP mean/covariance with the usual moving ±2.25σ exclusion and
  holding the reviewed kernel coordinates fixed at each tested mass.
- Use the same signed profile-likelihood diagnostic and numerical
  covariance-conditioning / known-feasible-null safeguards as the pilot.
  Retain raw likelihood differences and safeguard flags in the ledger.
- Reconstruct the saved observed fits on this grid as a numerical check.
- The primary comparison is the **change** in signed diagnostic caused by
  the injection relative to the same background-only scan. Also show the
  absolute scans so any generating-background / fitting-model mismatch is
  visible. A wider mask changes the generating truth, so do not silently mix
  the absolute results with the first pilot's toy medians.
- Recompute the 66 MeV deterministic reference on the new common truth for
  an apples-to-apples response comparison. Do not generate 66 MeV toys.
- Preserve all existing pilot artifacts, including the original twenty
  pseudo-spectra, scans, figures, protocol, and script. Write follow-up outputs
  only in this subfolder; the parent README can summarize the new findings.

This is a post-selection, conditional, fixed-kernel mechanism diagnostic.
It cannot establish whether either observed candidate is physical, rank
signal hypotheses statistically, or supply local/global calibrated p-values.
