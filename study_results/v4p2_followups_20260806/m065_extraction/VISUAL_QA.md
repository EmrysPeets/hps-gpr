# Visual QA

Status: **PASS**

The final PNGs were inspected at their original rendered resolution on
2026-08-06, and the corresponding PDFs were checked to be single-page vector
artifacts.

| Figure | PNG dimensions | PDF pages | Checks |
| --- | ---: | ---: | --- |
| `figure61_common_0p5MeV` | 4564 x 2275 | 1 | Title, three dataset columns, shared legend, blind-window boundaries, axes, and lower-panel uncertainties are legible and unclipped. |
| `figure61_common_0p5MeV_profiled` | 4534 x 2275 | 1 | Exact-window curves and count-residual markers remain distinguishable; labels do not describe residuals as per-bin significances. |
| `figure62_profiled_residuals_physical68` | 4358 x 2335 | 1 | All three native residual panels are retained; the physical coefficient panel, nominal/asymptotic qualifier, interval endpoints, and shared-fit row are unobstructed. |
| `figure62_coefficients_physical68` | 2725 x 1566 | 1 | Native/common markers, physical boundary, asymmetric intervals, legend, and non-coverage-calibrated qualifier are legible and unclipped. |

No overlapping annotations, cropped labels, missing panels, or negative-axis
physical interval endpoints were found in the final renders.
