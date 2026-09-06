# Numerical acceleration amendment, 6 September 2026

The exact ten-scan pilots are complete for both datasets. Exact 2016
validation was paused, retaining every completed checkpoint, after the
measured evaluation ratio projected roughly 37 further minutes plus a
similarly expensive Asimov phase. The statistical definitions, spectra,
seeds, mass grids and physical qualifications in PROTOCOL.md remain fixed.

Benchmark the existing frozen calibration_core.enable_lowrank accelerator at
2016 masses 39, 88, 120 and 180 MeV and 2021 masses 50, 150 and 250 MeV.
This is a numerical implementation decision, using agreement and runtime,
not the favorability of any observed p-value. The accelerator truncates
joint-kernel eigenfeatures at relative 1e-15 and background nuisance modes
at 1e-5 in Poisson-whitened covariance units. Every mass must pass the
parent's two-truth/proposal gates (<1e-3 for mean/covariance/r/q metrics) or
use its exact fallback. Scalar/batch and observed-replay gates remain.

Use a separately contracted runner and global_fast output tree. Reuse exactly
the same named ensemble/seed coordinates and counts. These are paired
numerical implementations of the same experiments, not additional toys.
Compare all ten exact pilot roots at every mass, and every available exact
validation root, to the accelerated roots, with absolute error below 1e-3.
Record all overlaps, fallback decisions and source hashes. Preserve the
incomplete exact cohort and its interruption record.

Before accepting an Asimov covariance, check the exact unperturbed root and
predeclared one-bin responses, including full-support endpoints, a uniform
16-bin grid, and the blind-window endpoints/center at every mass. Require
absolute root error below 1e-3 and response-difference error below 1e-4.
If the response gate fails, compute that mass's Asimov column exactly.
Additional full-column checks and covariance/statistic error gates will be
recorded before inspecting the accelerated global results. The independent
HEP reviewer will assess cancellation risks in the response construction.

This amendment changes the numerical backend only. It does not license a
new background truth, fit policy, asymptotic mapping, scan ordering, mass
grid, exposure, or combination rule. Numerical approximations and their
measured accuracy must be disclosed in the report. No final-discovery
qualification follows from a passed acceleration gate.
