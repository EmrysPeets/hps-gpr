# Detector geometry sources for v5.7.1

Fetched small files from JeffersonLab/hps-java at commit `473c732dafaebdd3f58389cd1052a35c60098a23`. The authoritative geometric input for this study is the committed LCDD snapshot, not an independently rebuilt compact.xml. Thus compact constants or alignment parameters are context; no claim is made that rebuilding a compact.xml today gives byte-identical LCDD output.

- 2015: `HPS-EngRun2015-Nominal-v6-0-fieldmap` (representative nominal engineering-run geometry; no exact selected-data run/alignment mapping verified).
- 2016: `HPS-PhysicsRun2016-Pass2` (used in local HPSTR `plotUtils/reach/simps22/2016_reach/makeComponents/makeTotRadAcc.py`; exact correspondence to this study's selected data histogram not established).
- 2021: `HPS_Run2021Pass2FEE` (local HPSTR `plotUtils/reach/simps22/2021_reach/gen_mc/gen_rad_mc/slic/jobs.json` explicitly selects this detector, run 14166, `run_params=3pt7`, `target_z=0`).

`extract_sensors.py` recursively composes the GDML world placement tree and extracts 36 / 36 / 40 active silicon rectangles. JSON fields provide full local dimensions, local-to-world axes, centers, and corners. The last three old stations have hole/slot sensor alternatives in each axial and stereo plane. A station hit requires at least one active sensor in its axial plane and one in its stereo plane; a ray need not hit both hole and slot sensors.

GDML rotation verification comes from Geant4 `G4GDMLReadDefine.cc:153-163` (`rotateX`, `rotateY`, `rotateZ`) and `G4GDMLReadStructure.cc:476` (inverse for local-to-parent transform), pinned commit `62f62ecae238a7c304c52af4affbe70795475590`. The parser uses `(Rz Ry Rx)^T`, then composes parent transforms. QA checks orthonormality, determinant, plane residuals, top/bottom centers and plausible station locations.

Beam rotation sign is verified in hps-mc `tools/stdhep-tools/src/beam_coords.cc:29-30`, commit `308e27be7821f1aad406cb8060602a10fca2d37a`: `px'=px cos(alpha)+pz sin(alpha)`, `pz'=pz cos(alpha)-px sin(alpha)`. The utility default is 0.0305 rad, while these compact files specify 0.03052 rad; this study uses the compact value. Global x is horizontal, y vertical, z downstream. No physical target volume appears in these LCDD files. Origin `(0,0,0)` is a documented model assumption (only z=0 receives direct support from the local 2021 job).

Normal active sensors are 38.3399 x 98.33 x 0.32 mm. The first two 2021 stations use 14.025 x 30 x 0.2 mm active sensors. 2021 module L1 is physical L0; module L2 is upgraded L1. First sensor centers lie at z~38 mm for 2021 and z~88 mm for 2015/2016; nominal station centers are 50 / 100 mm because the axial/stereo and top/bottom planes are staggered. The full hierarchy matters.

This extraction file describes silicon only. The final study applies the active magnetic maps in `fieldmaps/active_field_config.json`; original LCDD field references are retained as provenance, with source-backed updates documented in `../../qa/field_source_review.md`. Finite straight-ray silicon intersections are geometric conditional fractions, not HPS A-prime reconstruction/trigger/selection efficiencies. Material scattering, bending, dead channels, beam spot, production-angle/energy distributions, and displaced vertices require additional modeling or simulation.

Run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 extract_sensors.py`. This regenerates JSON geometry, QA, and the source SHA-256 manifest without network access or simulation.
