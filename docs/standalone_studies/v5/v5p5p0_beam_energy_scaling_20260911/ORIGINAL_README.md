# HPS-GPR v5.5.0: beam energy and shared structures

This diagnostic extends the frozen v5.0.4 results. The eight-page note, six standalone figure pairs, numerical arrays, physics consultation, and source snapshots are bundled here. No parent files were edited; no new toys or data were used.

## Main results

- The verified 2015/2016/2021 energies are 1.056/2.30/3.74 GeV. Target, detector, and selection differences prevent an energy-only causal interpretation.
- With a 92 MeV 2016 reference, square-root mass scaling maps 2015/2021 to 62.338/117.317 MeV, with signed fitted roots +1.540/-0.450. Linear scaling maps them to 42.240/149.600 MeV, with roots -0.799/+0.030. These simple prescriptions do not preserve a positive three-campaign feature.
- At fixed 92 MeV, all three amplitudes are positive, with strong differences in their preferred equivalent couplings. The parent common-rate likelihood loss is reproduced as 11.552.
- An additional empirical response factor at fixed mass, R=(E/2.3)^beta, fits beta=-2.904 and R=(9.585,1,0.244). It leaves a descriptive likelihood loss of 0.157 relative to separate rates. This nearly saturated, selected-mass fit is not evidence for coupling running or a predicted production law. The inherited response already includes its beam-dependent prompt-density conversion.
- The full signed scan comparison and seven-anchor matrix include other excesses and deficits. No mass exponent is optimized, no cross-window likelihood comparison is made, and no calibrated preference or global significance is assigned.

## Read and reproduce

- `pdf/HPS_GPR_v5p5p0_Beam_Energy_Scaling_Study.pdf`: final note.
- `figures/common_coupling_energy_interpretation_92.pdf`: requested common-coupling-style comparison.
- `figures/fixed_mass_rate_response.pdf`: fixed-mass empirical rate comparison.
- `figures/beam_energy_scan_comparison.pdf`: other structures across all saved supports.
- `derived/anchor_fits.csv`: exact continuous-mass fits; authoritative over preliminary interpolation in the independent audit.
- `physics_review.md` and `references_physics.json`: physics consultation and six primary sources.

From this directory, `bash scripts/build.sh` regenerates the fits, figures, table sources, and PDF. It needs Python with numpy, scipy, pandas and matplotlib, plus Tectonic and its cached LaTeX resources. Numerical work uses one thread. `scripts/validate.py` additionally needs PyMuPDF and Pillow to extract and render the note. Figure PDFs are vector output. PNGs are previews.

The `engine` modules are byte-for-byte copies of the parent solvers and use copied local inputs. The raw scopes input intentionally retains the parent's pre-extension 90 MeV endpoint; inherited `common.py` extends 2015 to 100 MeV. `provenance/inputs.json` pins all snapshots and the original checkout SHA. `protocol.json` covers fixed mass exponents; `provenance/rate_protocol.json` separately records the later empirical rate-slope diagnostic. Unsupported masses are missing, never zero-valued fits.

`derived/interior_extrema.csv` contains 55 unthinned strict interior extrema. The independent `structure_audit.md` contains a distinct 37-feature catalog after resolution separation. Neither is a list of independent trials. The audit's interpolated values were superseded by exact anchor fits. Curvature errors are local, not post-selection intervals. Background-source and 2015 endpoint-extension qualifications remain inherited.

QA records verify convergence, positive means, replay of parent values, unchanged source identities, semantic PDF content, and rendered pages. `MANIFEST.sha256` identifies every delivered file except itself. No commit, push, or external publication was made.
