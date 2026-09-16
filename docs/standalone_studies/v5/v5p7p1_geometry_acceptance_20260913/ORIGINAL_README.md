# HPS v5.7.1: magnetic transport and boosted-decay acceptance

This revision propagates electron and positron trajectories through source-verified HPS field maps and tests them against committed LCDD sensor geometry with the specified run-dependent hit requirements. It replaces the original 70 mrad surrogate. The ordinary LaTeX article presents the decay boost, worked numerical example, magnetic equations, sensor-crossing test, detector inputs and hit selection before the first figure.

## Deliverables

- `pdf/HPS_v5p7p1_Geometry_Acceptance_Study.pdf`: report, with separate landscape geometry and response plates for readable labels.
- `figures/HPS_v5p7p1_acceptance_overview.pdf` and `.png`: updated 3-column, 3-row overview.
- `figures/HPS_v5p7p1_geometry_illustration.pdf`, `.svg`, `.png`: standalone top row showing the three-dimensional sensor faces and accepted curved-track examples, with the nominal 15 mrad mass references.
- `figures/HPS_v5p7p1_response_spectra.pdf`: separate 3-column, 2-row plate of field/zero-field acceptance comparisons and selected data spectra.
- `figures/HPS_v5p7_original_angular_illustration.pdf`, `.svg`, `.png`: the specifically requested original v5.7 top-row illustration, re-rendered as vectors. Its 70 mrad angle remains explicitly illustrative.
- `source/`: portable LaTeX package with an insertable section, figures, table and references. `references.bib` supplies the same citation keys for merging into a note that uses BibTeX; the standalone report uses `references.tex` directly.
- `inputs/geometry/`: pinned LCDD/compact/Java sources, the offline geometry extractor, active rectangles and source hashes. `inputs/geometry/fieldmaps/active_field_config.json` specifies the actual field inputs and reader conventions. `derived/` contains plotted numbers and histogram provenance; `qa/` contains numerical and document checks.

## What changed physically

The selected hps-java revision is `473c732dafaebdd3f58389cd1052a35c60098a23`. It supplies 36/36/40 active sensor rectangles for the representative 2015, 2016 and 2021 configurations. Sensor placement is composed recursively from the LCDD hierarchy, with the GDML rotation convention independently checked. The beam is rotated by +30.52 mrad into global coordinates using the verified hps-mc sign convention.

The main curves implement the run-dependent hit component of the stated selections:

| Run | Positron | Electron | Basis |
| --- | --- | --- | --- |
| 2015 | At least 5 paired 3D points out of 6, including L1 and L2 | At least 5 paired 3D points out of 6 | Archived internal note, PDF pp. 11–12/Table 3, and local v5.0.4 convention |
| 2016 | At least 5 paired 3D points out of 6 | At least 5 paired 3D points out of 6 | Published 2016 selection, Sec. III.2.2 |
| 2021 | At least 10 individual 2D plane hits out of 14 | At least 8 individual 2D plane hits out of 14 | Current user-specified analysis correction |

A paired 3D point requires axial and stereo crossings at the same station in the same detector half. Individual 2D hits are counted separately; their count is not interchangeable with twice the 3D count when a view lacks its partner. Hole and slot sensors are alternatives within each view. Solid and dashed curves apply selected hits with the field present at x=1 and x=0.8. Grey dotted curves are the selected-hit zero-field reference at x=1; faint dash-dotted curves require all station pairs with the field present.

The 2015 published search required both tracks in the first two layers, whereas the implemented local-analysis convention applies that additional condition to the positron. The corrected 2021 counts supersede the older statements in the preserved v5.0.4 source excerpts. The selection provenance and source hashes are recorded in `inputs/selection/`; equivalence to the saved 2021 histogram selection remains unverified.

The active field maps are pinned to hps-fieldmaps revision `feadfcbd17bdee8515cd962d75a90ba1534fa888`:

- 2015 uses the corrected-unfolding v3 map, scale 0.7992, with the original committed sensor transforms. The corrected compact has the same non-field geometry content; a separate corrected LCDD is not supplied by the repository.
- 2016 uses the scale-1.04545 v4 map referenced by the selected Pass2 detector.
- 2021 combines the documented physics-data scale-1.07326 map with the retained Pass2FEE sensors. The data-map calibration energy is 3.742 GeV; this study uses nominal 3.74 GeV. The two detector alignments differ, so the assembly is explicit and is not claimed to match the histogram conditions exactly.

The maps use a 5 mm grid, offset `(21.17, 0, 457.2)` mm and trilinear interpolation. Raw field components are multiplied by 1000 to obtain tesla, consistent with the inspected LCDD/SLIC and Java readers. Components already refer to the global detector axes. The central By values at the map origin are approximately -0.240245, -0.5234 and -0.8596 T; the map origin is not the target.

The main calculation tests full 3D finite faces after RK4 propagation in path length. Maximum steps are 5 mm with an additional 0.01 rad bend limit. Hermite-interpolated plane crossings must satisfy a 1e-7 mm residual before the active half-widths are tested. Transport stops at Z=940 mm, the first nonpositive longitudinal direction, or an outward transverse map exit. All active faces lie inside those geometric bounds. The exterior field is zero, matching the source readers; later recrossings of longitudinally turning tracks are outside this forward model.

The top-row display shows accepted curved trajectories at the first and last nonzero sampled mass for each year. Those are passing examples, not physical endpoints. The report separately retains zero-field projected equal-sharing intervals of approximately 17.27–72.24, 37.54–157.50 and 54.50–348.53 MeV. Their angles are 16.326–68.458, 16.316–68.532 and 14.570–93.324 mrad. These projected references apply the selected-hit criteria and union both upper/lower charge assignments, but discard horizontal seams. They do not bound the full magnetic acceptance.

The nominal 15 mrad equal-sharing references at x=1 are 15.84, 34.50 and 56.10 MeV in the massless-electron limit. The gap alone supplies no upper mass cutoff. Acceptance panels extend to 250, 550 and 950 MeV by year; sampled tails do not establish exact endpoints. The selected-hit magnetic peaks on the saved grid are 40, 90 and 195 MeV, with fractions approximately 26.09%, 25.95% and 40.11%. The plotted fractions, example masses and sampled peaks are saved in `derived/magnetic_acceptance.csv` and `derived/magnetic_summary.json`.

Magnetic transport and the hit requirements are included. Material interactions, hit finding, trigger, reconstruction, other kinematic and analysis cuts, dead channels, beam spot, production energy/direction distributions and displaced decays are omitted. Thus the curves are conditional geometry-and-field fractions, not calibrated signal efficiencies or accepted production rates. The existing data spectra are unchanged raw selected-pair densities with inherited display crops and a common 300 MeV display endpoint; the 2015 input ends at 150 MeV.

## Build

From this directory:

```bash
bash build.sh
```

Requires a C++17 compiler, Python with NumPy, SciPy, matplotlib and uproot, and Tectonic. On macOS the build uses `clang++ -dynamiclib`. The source/data archive contains the three active compressed field maps; `prepare_field_inputs.py` restores and hashes their ASCII tables locally. The pinned geometry and field inputs are local; rebuilding does not require Geant4 or new toys. Tectonic may fetch missing TeX packages on an initially empty installation. Numerical work is serial, nice=10, with BLAS/OMP capped at one. The preceding v5.7.0 package is preserved.

To compile only the already populated LaTeX package:

```bash
cd source
tectonic main.tex
```

## Insert into the full analysis note

Copy `source/` into the note as, for example, `v571/`. In the note preamble load `amsmath`, `amssymb`, `graphicx`, `array`, `booktabs`, `caption`, `pdflscape` and `float`, then define:

```latex
\newcommand{\vFiveSevenRoot}{v571}
```

At the intended location insert:

```latex
\input{v571/acceptance_section.tex}
```

Merge the entries in `v571/references.bib` into the note bibliography. Labels and citation keys use the `v571:` prefix. The entry point includes the equations first, followed by geometry/results and figures. If only the derivation is wanted, `v571/sections/kinematics.tex` is independent of the figure files. The original angular display is also independently available for a simpler introductory schematic.

## Verification

Geometry rotation checks include an explicit base-placement calculation, and the independent boost check conserves the parent four-vector to floating-point precision. Uniform-field propagation is checked against analytic circles for both charges. Zero-field transport reproduces the direct finite-ray hit counts for 1024 orientations per year. Step sizes of 5, 2.5 and 1.25 mm give identical accepted-event classifications for 1024 orientations at one representative mass per year: 58.08, 126.5 and 205.7 MeV. These are sampled checks, not a global integration-error bound.

The magnetic scan uses 4096 deterministic Sobol orientations and 5 MeV mass spacing. Refinement to 8192 points at three masses per year changes the selected-hit or all-station fraction by at most 0.0023706743 absolute acceptance, less than 0.238 percentage point. Details are in `qa/magnetic_scan_validation.json`; this is a sampled check, not a global uncertainty bound. Count-preserving histogram rebinning and hit-requirement ordering are also checked. The report and standalone displays are rendered and visually reviewed before delivery; final source hashes and text checks are saved alongside the package.
