# Authorized 2016 extension, September 24, 2026

The user requested:

> if you were to conduct a similar study for mean offsets for 2016, would you look at 10% dataset offsets based on GP mean , then scale those offsets to account for what that would correspond to in 2016 100%. then see if scaled 100% toys extract appropriately, then incorporate those pull changes in the paired response of 2016 when determining upper limits and signal extraction? if so, conduct this study and add it to the back of this report. if not, identify proper way to handle this and continue

The resulting derivative preserves the original nine-page v6.3.1 2021 report.
This extension tests exposure scaling instead of assuming it, separates additive
offset from paired signal response and pull width, and validates conditional
finite-grid inference on independent evaluation toys. It does not modify observed
2016 production results. The historical 10% sample lacks verified selection and
exposure equivalence, so the controlled low-exposure branch uses 0.1 of the same
pinned full-exposure mean. See the frozen protocol for the precise scientific
scope, seed design, fit settings, and limitations.

The user also authorized optional S3DF offloading in their epeets/src directory.
This bounded study used local workers only.
