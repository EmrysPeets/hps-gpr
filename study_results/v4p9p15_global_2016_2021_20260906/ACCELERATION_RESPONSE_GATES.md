# Full-response accuracy checks declared before accelerated global analysis

In addition to the per-mass response probes in ACCELERATION_PROTOCOL.md,
compute the complete exact Asimov column at 2016 masses 39, 56, 66, 75, 120,
and 180 MeV and 2021 masses 50, 78, 100, 150, 200 and 250 MeV. These cover support
edges, the disclosed background transition, and the middle/high mass scan.
Retain the exact columns. Require maximum absolute root error <1e-3,
maximum absolute centered-response error <1e-4, and relative L2 response
error <1e-3. A failing coordinate uses the exact backend for all ensembles.

Before global analysis, compare the covariance submatrix built from these
complete exact columns with that from the final columns. Require every
response width to agree within relative 1e-3 and every normalized correlation
entry to agree within absolute 1e-3. Keep these numerical gates separate from
Poisson-vs-Gaussian statistical checks and background qualification. Report
both the maximum discrepancies and exact-fallback coordinates.

Also require the exact baseline-root difference divided by the response width
to be below 1e-3 at every mass, and probe-response differences divided by
that width below 1e-4. This explicitly addresses cancellation in D.
