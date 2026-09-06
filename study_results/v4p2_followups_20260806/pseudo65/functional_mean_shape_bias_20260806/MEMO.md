# Functional-mean shape-bias diagnostic

## Question and construction

This conditional diagnostic asks whether replacing [60,70) MeV with the stored smooth `fGenGammaThresh` expectation can itself produce a 61--63 MeV positive response when the result is analyzed by the unchanged v4.2 GP card.

The central inputs are fractional deterministic means (Asimov-like only within the replacement window), while the original observed 2021 10% counts are retained bitwise outside. They are not observed datasets, complete Asimov datasets, pseudoexperiments, expected results, or coverage tests.

Each table entry is `Ahat; sigma_A; local p0; local Z`.

| mass | functional mean | functional Poisson draw | GP mean | GP-mean Poisson draw |
|---:|---:|---:|---:|---:|
| 61 MeV | 14299.6; 6301.1; 0.01166; 2.268 | 21474.1; 6265.3; 0.0003008; 3.431 | 6016.9; 6329.8; 0.1707; 0.951 | 9076.1; 6310.7; 0.07505; 1.439 |
| 62 MeV | 12115.2; 6352.0; 0.02816; 1.909 | 23177.5; 6288.4; 0.0001123; 3.690 | 2746.0; 6383.4; 0.3333; 0.431 | 4196.5; 6366.7; 0.2549; 0.659 |
| 63 MeV | 8899.6; 6440.1; 0.08341; 1.383 | 20219.2; 6386.1; 0.0007687; 3.168 | 1487.1; 6467.0; 0.4087; 0.231 | -272.3; 6461.1; 0.5; 0.000 |

## Quantitative answer

Yes, conditionally: the deterministic functional mean produces a positive GP-extraction shoulder in the 61--63 MeV region even without a Poisson draw. Its largest local response in that window is Z=2.268 at 61 MeV. The functional Poisson draw is larger, reaching Z=3.690 at 62 MeV.

At 62 MeV the functional deterministic mean exceeds the GP deterministic mean by 9369.2 fitted events. The particular functional Poisson draw adds another 11062.3 events relative to its deterministic mean response. Thus the reviewed 61--63 MeV shoulder contains both a deterministic truth-model/GP mismatch component and an additional fluctuation component in that one draw.

The GP-mean deterministic lane also has a smaller positive response near the low side of the window, so this construction does not identify every positive fitted event exclusively with the functional interpolation. The relevant functional-specific diagnostic is its excess over the otherwise identical GP-mean lane.

## Statistical boundary

All p-values are local asymptotic responses of one conditional hybrid spectrum. No ensemble was generated. The comparison is not a coverage statement, expected sensitivity, global p-value, or probability that the interpolation will create a shoulder.
