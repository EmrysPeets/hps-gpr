# Scientific-scope clarification for the frozen 2016 support protocol

This file does not change the numerical protocol, its inputs, or its selection
rule. It records two provenance qualifications found by an independent static
audit before any support-dependent observed scan was authorized.

## 2016 10% shape source

The frozen protocol and study specification call the 2016 10% development
histogram "independent." No run- or event-level provenance in this release
establishes statistical disjointness from the full-2016 histogram. The 10%
input must therefore be described as the **pre-existing 2016 10% development
sample/subset**, not as an independent sample. It supplies partial observed-
shape information to the source-conditioned stress truth.

Before the support freeze, the full-2016 value array enters the truth builder
only through one scalar count over 26--210 MeV. No support-specific full-data
fitted amplitude, local p-value, or upper limit is used to select an edge. This
is a narrower statement than full-data blindness and is the statement used in
the v4.9.7 note.

## Broad-tail continuation

Above 85 MeV, the hybrid generating mean uses the archived
`fShiftSigPowTail` expectation. Its immutable metadata records
`fit_ok: false`. The static audit nevertheless verifies a finite nonnegative
shape, a Pearson chi-square per degree of freedom of 0.99003897, free
parameters strictly inside their declared bounds, a positive 85--210 MeV
tail, consistency with the archived expected total, and bitwise reproduction
of all 100 stored Poisson toys.

The nonconverged ROOT status is explicitly waived **only** to use this curve as
a smooth broad-tail component of the conditional source-conditioned stress
truth. The waiver does not qualify it as a physical background generator,
coverage ensemble, calibrated expected-sensitivity model, exclusion model, or
significance calibration.

## Interpretation

Any selected support is conditional on this construction and on the declared
pull-recovery rule. The subsequent 100-toy combined bands hold reviewed GP
states fixed and use asymptotic inner limits; they are descriptive conditional
limit quantiles, not direct coverage or toy-calibrated inner confidence limits.
