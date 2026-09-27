# Independent magnetic-transport code review

Reviewed `scripts/field_transport.cpp` and `scripts/magnetic_model.py` before the production scan. This was a bounded code/units review plus a direct read of the saved sensor corners; no transport scan was launched. Findings below describe the inspected pre-fix version and must be reconciled with later revisions.

## Findings requiring attention

1. **Map-cache identity is not validated.** `magnetic_model.py:24–46` keys the field cache only by year and reuses it whenever the `.npy` file exists. It records the source hash but never compares that hash, the selected filename, the conversion or the offset on reuse. Changing the 2015 map can silently preserve the previous field. Validate cache identity or invalidate and rebuild the affected cache before scanning. This was reported to the parent immediately.

2. **Sensor root convergence is not checked.** `field_transport.cpp:42–52` brackets a crossing by endpoint signs, applies four clamped Newton iterations to the Hermite interpolant, and then tests only the in-plane coordinates. It does not verify the remaining normal-plane residual. A poorly converged root could be counted as a hit. Add a plane-residual tolerance with a bracketed fallback or explicit failure counter. Most forward crossings should be well behaved, but an unchecked convergence assumption is avoidable.

3. **Forward-only termination must remain explicit.** `field_transport.cpp:61` stops at the first nonpositive longitudinal direction. The crossing prefilter also assumes forward longitudinal progress. This is a declared forward-transport model, not complete propagation of curling tracks through later recrossings. Record this criterion and distinguish it from the path/iteration safety bounds. The current safety flag does not report a longitudinal turning stop. Endpoint-sign tests likewise do not identify two crossings of a plane within one step near a turn; step refinement is needed for the actual selected fractions.

## Checks consistent with the stated model

- The Lorentz sign is `q * (u cross B)`, with `q=+1` for the positron and `q=-1` for the electron. The coefficient `0.299792458` is correct for momentum in MeV/c, path length in mm and field in tesla, with c=1 in the equations.
- RK4 integrates position and direction in path length, with fixed initial momentum magnitude. Normalizing direction after each step enforces the constant-magnitude constraint. This is not an independent demonstration of integration accuracy; the planned uniform-field analytic comparison and step refinement provide that evidence.
- The step cap limits the estimated maximum direction change to 0.01 rad per step using the largest saved field magnitude. Trilinear interpolation cannot exceed the largest corner-vector norm. Field-grid spacing and piecewise derivatives still motivate a step-size comparison.
- The C++ field indexing matches NumPy C-order `(x, y, z, component)`. Global grid origins are table minima plus the stated offset; querying subtracts that origin. The conversion by 1000 agrees with the supplied field-reader/source ledger. The inspected geometry records the same offset `(21.17, 0, 457.2)` mm for all three maps; pin this metadata to the actual selected maps after source corrections.
- Sensor axes and half-widths are selected consistently from the geometry rows. Hole/slot rectangles are unioned within each view. Paired 3D flags require axial and stereo hits in the same half; the 2D result counts individual view flags by station.
- Saved active-sensor corner bounds are contained within X approximately `[-85.35, 128.91]` mm, Y approximately `[-56.74, 56.84]` mm, and Z below `914.31` mm. Thus the current `zstop=940` mm encloses all active faces. The current field grid X range `[-228.83, 271.17]` mm and Y range `[-70, 70]` mm also enclose them. Assert these conditions against the final chosen grid in code.
- Stopping after an outward transverse grid exit is consistent with the explicitly zero exterior field and enclosing sensor bounds: the subsequent straight ray cannot return to those sensors. This is a finite-map model boundary, not a claim about the physical fringe field outside the map.

The independent uniform-field and zero-field hit checks are owned by the parent task. No acceptance calibration or field-source equivalence is established by this review.

## Final resolution

The cache now verifies the pinned source hash, path, conversion factor and offsets before reuse. Sensor crossings require a normal residual <=1e-7 mm, with a bracketed fallback. The forward-only termination is documented in the report. Active sensor corners are checked to lie within transverse field-grid boundaries and below z=940 mm. Final analytic-circle, zero-field-count and step-refinement checks pass.
