# Pre-execution numerical-conditioning audit amendment

Frozen before any new final-dataset result was evaluated.  This file supplies
the numerical tolerances referenced by `STATISTICAL_PROTOCOL.md`; it does not
change the conditioning algorithm.

The audit compares the release's explicit deterministic diagonal loading with
the attested runtime's native implicit Cholesky regularization while keeping the
piecewise bounded CLs mapping, observed vectors, GP states, and signal
normalizations identical.

The fixed audit coordinates are:

- standalone 2015 at 19, 50, and 90 MeV;
- standalone 2016 at 39, 65, 102, and 180 MeV;
- standalone 2021 at 50, 78, 150, and 250 MeV;
- every available pair and the all-three combination at 50, 65, and 90 MeV.

A coordinate passes only if both paths converge with valid likelihood nesting
and if:

- the relative difference in the 90% CLs epsilon-squared limit is at most
  `5e-4`; and
- the absolute difference in the one-sided local asymptotic Z value is at most
  `5e-3`.

The complete release also requires every covariance block on the 415-state
grid to select a relative load strictly below `1e-4`.  Any failure stops the
release.  Audit coordinates and tolerances cannot be changed after results are
seen.
