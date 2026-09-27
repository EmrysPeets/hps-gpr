# Bounded remote production-selection check

22 September 2026. The named remote hpstr checkout is commit `751bc138c5160dfdf7a0f323f2d9dc3fdbf155e4` (27 June 2024, “2015 related changes”). A bounded text-source pass found no occurrence of the requested v13/v16 production directory names or the `ele_p_smear_ratio` / `pos_p_smear_ratio` branches. Its processor names also expose no preselection or smearing production module. The requested remote data directories exist, but their names do not establish matching cuts or smearing.

The three files here are exact text snapshots from that historical checkout, with remote paths and SHA-256 identities in `manifest.json`. They provide schema/history context only and are **not** promoted to authoritative v13 or v16 production configurations. The named checkout does not resolve whether the MC reconstructed vertex mass already incorporates the stored momentum smearing, whether further mass transformation is required, or whether MC and observed selections match. Use the actual production records and the separate ROOT schema/cutflow audit to resolve those questions; do not automatically apply smearing twice.

This audit did not read MC ROOT payloads, run fits, install software, or modify cluster files. The source search remained within the named hpstr checkout; no wider search was undertaken.
