# Magnetic-field source and convention review

## Adopted fields after source review

The final active choice is recorded in `inputs/geometry/fieldmaps/active_field_config.json`. Use its per-year paths and hashes:

- 2015: corrected-unfolding `v3/125acm2_3kg_corrected_unfolded_scaled_0.7992_v3.dat`, central By=-0.2402448 T. All non-info/non-field XML content of its detector compact is structurally identical to the original compact; the repository contains no separately generated v3 LCDD. The original committed sensor transforms are retained without claiming a new geometry build. Corrected Bz at the nominal y=0 target is zero.
- 2016: selected Pass2 `209acm2_5kg_corrected_unfolded_scaled_1.04545_v4.dat`, central By=-0.5234 T.
- 2021: documented physics-data `data2021/334acm3_8kg_corrected_unfolded_scaled_1.07326.dat`, central By=-0.8596 T, combined explicitly with the retained Pass2FEE sensor geometry. The catalogue's calibration energy is 3.742 GeV and the study retains the nominal rounded 3.74 GeV beam. Exact alignment or selected-histogram condition equivalence is not asserted.

The 2021 Pass1_v5 compact and LCDD provide a concrete configuration using the adopted 1.07326 map. Compared with retained Pass2FEE sensors, all 40 sensor names and dimensions agree; maximum center-component difference is 0.7311343 mm, rotation-matrix element difference 0.00689214, and corner-component difference 0.7463879 mm. Thus these are different alignment snapshots, and the chosen assembly must be stated rather than presented as identical configurations. The differences are recorded in `data2021/sensor_comparison.json`.

Legacy map copies below remain source provenance, not active inputs. Redundant uncompressed maps need not be included in the delivery archive.

## Original detector references inspected

The three archives referenced by the selected detector compact files were downloaded from JeffersonLab/hps-fieldmaps commit `feadfcbd17bdee8515cd962d75a90ba1534fa888` and extracted into `inputs/geometry/fieldmaps/`. `download_manifest.json` records URLs, archive hashes, extracted filenames, sizes and hashes. `field_parameters.json` records full-grid validation and numerical checks. No transport simulation was run for this source review.

| Geometry year | Exact selected map | B_y at map origin [T] | Extracted size [bytes] |
|---|---|---:|---:|
| 2015 | 125acm2_3kg_corrected_unfolded_scaled_0.7992.dat | -0.2400045552 | 129364594 |
| 2016 | 209acm2_5kg_corrected_unfolded_scaled_1.04545_v4.dat | -0.5234 | 88954596 |
| 2021 | 334acm3_8kg_corrected_unfolded_scaled_1.0508.dat | -0.8417 | 88964694 |

## Format, units, interpolation and signs

Each map contains one blank line, a dimensions line `101 29 601`, six column declarations, a `0 End of Header` line, then 1,760,329 six-column rows. The columns are x, y, z, Bx, By, Bz. Full-grid validation confirms C-order storage with x outermost, y next, and z varying fastest. Each axis is uniformly spaced by 5 mm: x=-250..250 mm, y=-70..70 mm, z=-1500..1500 mm.

The raw headers explicitly label field components `BX(1000T)`, `BY(1000T)`, `BZ(1000T)`. **Multiply raw field values by 1000 to obtain tesla.** Do not interpret the raw numerical field values as tesla merely because the XML element has `funit="tesla"`. The SLIC/LCDD field-map subscriber ignores that optional unit field; the C++ map reader consumes Geant4 internal field units directly. Independently, the hps-lcsim Java reader explicitly multiplies each raw component by 1000 and explains that Geant4 represents one tesla as 0.001 internally.

The XML offsets, after evaluating units, are `(21.17, 0, 457.2)` mm for all three selected maps. Both readers query **r_table = r_global - offset**. They interpolate the eight neighboring samples independently in each field component. They apply no additional vector rotation, charge sign, reflection, unfolding or field scaling beyond the conversion to the consumer's unit convention. The table components are world-aligned Bx, By, Bz. The full tables have already been unfolded. Outside the defined grid the reader returns zero field. An implementation should treat points exactly on a maximum grid face safely by clipping the lower interpolation-cell index to n-2 while retaining the endpoint weight.

The map origin is therefore world position `(21.17,0,457.2)` mm, not the target. At the nominal world vertex `(0,0,0)` the selected map values are respectively `(-0.0000170754,-0.1976737092,-0.0730969727)` T, `(0,-0.430612,0)` T, and `(0,-0.690734896,0)` T.

For positive charge moving downstream in negative By, q(v cross B) points toward positive global x. A charge-sign QA check should therefore show the positron bending toward +x relative to its initial straight path and the electron toward -x. With momenta in MeV/c and path length in mm, the magnetic equation can use `d p / ds = 0.299792458 q (p/|p| cross B)` with B in tesla. No beam-coordinate rotation should be applied to B again.

## Exact source locators

Reader source files and SHA-256 hashes are saved under `inputs/geometry/fieldmaps/reader_source/` and `reader_source_manifest.json`.

- slaclab/lcdd commit `36a4805e2ff82ec04f7e5332532ef751fddb5057`, `src/lcdd/bfield/Cartesian3DMagneticFieldMap.cc`: reads x/y/z/Bx/By/Bz in nested x/y/z loops; `GetFieldValue` subtracts each offset, then trilinearly interpolates raw components; outside-grid branch returns zero.
- Same commit, `src/lcdd/subscribers/field_map_3dSubscriber.cc`: evaluates offset expressions and passes them to the map constructor; `lunit` and `funit` lookups are commented out.
- JeffersonLab/hps-lcsim commit `391d46492257c5ab155e22ecf74734efd7aaa27a`, `detector-framework/src/main/java/org/lcsim/geometry/field/FieldMap3D.java`, lines 182--204: explicit conversion factor 1000; lines 291--294: subtraction of global offsets.
- Selected LCDD files: 2015 line 8341, 2016 line 8377, 2021 line 10067 specify the map filenames and numerical offsets.

## Source discrepancies and their resolution

1. The 2021 compact comment claims -1.022 T, but the referenced raw map has central By=-0.8417 T. Use the actual map values, not that stale comment.
2. The selected 2015 legacy map has nonzero Bz on y=0 near the target. A separate hps-java detector, `HPS-EngRun2015-Nominal-v6-0-fieldmap_v3`, explicitly states that it uses a map with corrected unfolding and points to `125acm2_3kg_corrected_unfolded_scaled_0.7992_v3.tar.gz`. Its XML filename remains the old basename. That corrected archive must therefore be extracted into a separate directory if adopted, to preserve the original snapshot. The source XML is saved as `2015_corrected_unfolding_compact.xml`.
3. The pinned hps-fieldmaps README associates the 2021 physics data at 3.742 GeV with scale 1.07326; it lists scale 1.0508, the one referenced by HPS_Run2021Pass2FEE, under the 3.7 GeV proposal detector. The selected compact remains authoritative for its own map, but exact calibration equivalence to the 2021 data is not established. The pinned README is saved as `fieldmaps_README.md`.

The adopted-model declaration above resolves these source choices. No raw map is silently rescaled, and the resulting map-conditioned calculation is not a verified reconstruction efficiency for the data.
