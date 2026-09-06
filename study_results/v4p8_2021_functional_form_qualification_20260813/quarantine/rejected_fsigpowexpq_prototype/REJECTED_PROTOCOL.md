# Rejected fSigPowExpQ prototype

This directory is quarantined and is not an accepted v4.8 result.
`QUARANTINE_STATUS.json` is the machine-readable authority; any `pass` field in
the copied historical manifest or prototype code is superseded and must not be
interpreted as permission to run or use the product.

The prototype was stopped before production after the user rejected the source
fit quality.  The archived native-10% `fSigPowExpQ` fit has Pearson
chi-square/ndf about 6.17 and its `c1` coefficient is on the declared +50 bound;
the archived 1% fit also places `c1` on that bound.  Both metadata records report
`fit_ok=false`.

The copied prototype driver is also technically nonconforming: it was based on a
v4.6 subset, did not load and attest the archived runtime overlay by itself, and
its copied plotting script remained wired to `fGenGammaThresh`, 100 toys, and a
`sqrt(E/100)` residual.  No output from this prototype may be used for generator
qualification, closure acceptance, kernel-bound choice, Section 5 figures, or a
production-card change.

The directory is retained solely as a provenance record explaining why the
initial route was abandoned.
