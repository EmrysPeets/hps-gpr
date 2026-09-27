# Independent geometry and boost review

Historical review before the user supplied the final hit thresholds. The geometry transforms and boost findings below remain applicable. The former all-station/one-missed curve definitions and `station_counts` implementation are superseded by `hit_selection_review.md`, the saved selection ledger, and the final `hit_features`/`selection_mask` implementation.

Reviewed `scripts/build_study.py` against `inputs/geometry/active_sensors.json` and the committed LCDD. No actionable implementation bug was found in beam direction, plane intersection, finite rectangle bounds, hole/slot unions, axial/stereo requirements, or station counting. The original assumption that the exact vertical beam plane has a nonempty all-station aperture is false: zero-field rays can hit horizontal hole/slot seams. The parent identified this while running the actual finite-rectangle check and is replacing the top-row annotation with an explicitly projected vertical-envelope interval.

## Rotation sign check beyond orthogonality

The LCDD base placement is `(21.336, 0, 349.9358)` mm and its declared rotation is +pi/2 about x. Its local-to-world map must therefore be `(x,y,z) -> (x+21.336,z,349.9358-y)`. For the first eight 2021 sensor centers (the two short-sensor stations), the explicit map agrees with the recursive parser to **8.79e-14 mm**. This is an independent simple-placement constraint, not an orthogonality identity. Using the opposite rotation sign instead displaces those centers by at least **477.179 mm** and swaps their vertical half. The top first axial sensor is at `(1.15655146, 7.87627550, 38.38076418)` mm; the bottom first axial sensor is at `(1.85963797, -7.86832224, 61.50499320)` mm. These also demonstrate the top/bottom and axial/stereo staggering hidden by nominal 50-mm station labels.

## Boost and ray interpretation

The positive beam rotation in `daughters` matches hps-mc: beam-frame +z becomes `(sin(alpha),0,cos(alpha))`. A five-orientation check at 90 MeV parent mass and 2.30 GeV parent energy, including asymmetric rest-frame decays and different azimuths, gives maximum parent invariant-mass residual **1.03e-11 MeV**, total-energy residual **4.55e-13 MeV**, and total-momentum-component residual **4.55e-13 MeV**. The sign flip is applied to all rest-frame daughter momentum components before the boost; total parent momentum points along the declared global beam direction.

`prepared` correctly subtracts the nominal vertex from each sensor center. `station_counts` correctly obtains the ray parameter from the plane normal and tests both local planar coordinates against half the full sensor sizes. It unions hole/slot sensors within each station/half/view, then requires axial AND stereo in one detector half. Comparing the minimum of the two daughter station counts against the requested count correctly requires both daughters to satisfy the count independently.

`draw_geometry` displays axial faces projected onto beam-distance versus global vertical position. It is a projection: overlapping hole/slot faces and omitted stereo faces must not be read as the complete 3D aperture. The original `vertical_edges` routine tests both symmetric rays against the complete 3D rectangles, but its nonempty-aperture assertion fails: the nominal horizontal beam line crosses hole/slot seams in the downstream stations. For example, the 2015 station-4 beam x is near 15 mm, between the axial hole-sensor endpoint near 13.7 mm and slot-sensor start near 16.3 mm. A projected y-z envelope is therefore the appropriate top-row illustration if explicitly labeled as a projection; it must not be called a full 3D accepted mass interval. The middle-row finite-rectangle calculation retains the seams. The exact electron-mass symmetric-mass relation in `symmetric_mass` agrees with the Lorentz boost.

## Boundaries to retain in the report

- All-station means six stations for 2015/2016 and seven for 2021. One missing station therefore means at least five or six respectively; it is a geometric counterfactual and not a documented reconstruction track requirement.
- The transverse-vector angular weight is the relativistic `1+cos^2(theta*)` model, while the boost retains electron mass. At low mass the exact massive-vector decay angular law differs; these curves must retain the stated angular-model convention.
- Committed LCDD snapshots are the geometric input. Exact conditions equivalence to each existing selected mass histogram is unverified. The plotted data spectra do not validate these signal geometric fractions.
- No magnetic field, material, trigger, dead-channel, reconstruction, production-angle, or production-energy integration is applied. Parent `x=1` and `x=0.8` are fixed-energy illustrative scenarios. Nominal vertex is the origin; only the 2021 local job directly supports target z=0.
- The projected vertical-envelope endpoints apply to a 2D projection of the detector, not full 3D acceptance or absolute mass support across all decay orientations. A 2.5 MeV mass grid and finite quadrature do not establish a mathematical zero-support boundary.

Only these small algebraic/placement checks were run during the review; no scan, simulation, or numerical worker was launched.

## Field-map availability lookup

The exact map archives referenced by these compact files are available in JeffersonLab/hps-fieldmaps at commit `feadfcbd17bdee8515cd962d75a90ba1534fa888`. GitHub tree metadata gives compressed sizes of 20,412,210 bytes (2015 `125acm2_3kg_corrected_unfolded_scaled_0.7992.tar.gz`), 15,064,259 bytes (2016 `209acm2_5kg_corrected_unfolded_scaled_1.04545_v4.tar.gz`), and 15,077,298 bytes (2021 `334acm3_8kg_corrected_unfolded_scaled_1.0508.tar.gz`). Offsets are `(21.17, 0, 457.2)` mm for all three. No map was downloaded or transport approximation implemented; bending remains an explicit omitted effect.
