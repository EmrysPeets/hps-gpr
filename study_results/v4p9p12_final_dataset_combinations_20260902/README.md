# v4.9.12 final-dataset combinations

This directory builds the observed, fixed-mass asymptotic results for every
nonempty combination of the three current final inputs:

- full 2015;
- full 2016;
- 2021 10%;
- all three pairs; and
- the common all-three interval.

The 2021 1% comparison is not an input or result. The 2016 10% sample appears
only in upstream support-development evidence and is not a result curve. No
pseudoexperiments, expected-limit bands, scan-wide calibration, or global
significance are produced here.

## Frozen analysis choices

- 2015: 14--135 MeV GP support and resolution-scaled upper length factor 8.
- 2016: 30--210 MeV GP support and resolution-scaled upper length factor 12.
- 2021: 36--300 MeV GP support and resolution-scaled upper length factor 15.
- Signal coordinate: one nonnegative `epsilon^2` for each standalone result or
  one shared nonnegative `epsilon^2` for a combination.
- Limit: 90% bounded, piecewise-asymptotic CLs.
- Discovery diagnostic: one-sided, fixed-mass asymptotic `p0`.

The inference is conditional on the frozen GP states and on a partially
unblinded analysis history. It is not an unconditional coverage statement.

## Fail-closed workflow

`assemble_release_inputs.py` first replays the 2015 exact-max selection, the
fit-only 2021 repair requalification, and the full 2016 state certification.
It will not write the final inputs unless the 2016 decision authorizes all 142
states. `validate_release.py` repeats those semantic checks, audits every output
grid and profile, and is the only program allowed to emit
`qa/release_attestation.json` with status `release_complete`.

From the repository root, the complete analysis/export pipeline is:

```bash
python3 study_results/v4p9p12_final_dataset_combinations_20260902/run_release_pipeline.py --workers 6
```

The individual stages are, in order:

1. the 8 tail-mapping, 11 solver/conditioning, and 12 core profile-likelihood
   unit tests;
2. certified input assembly;
3. the exact 415-state prediction and 680-row seven-scope result scan;
4. the predeclared 23-coordinate numerical-conditioning impact audit;
5. extraction at the all-three local-`p0` minimum;
6. PDF/PNG figure generation;
7. fail-closed release validation and SHA-256 manifest creation; and
8. export of only release-attested files to the Harvard writing-sample tree.

The Harvard PDF is built separately from
`study_results/harvard_writing_sample_final_combinations_20260902/source/writing_sample.tex`
after the export succeeds.
