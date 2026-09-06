# Reader-facing validation-figure QA

Date: 2026-09-02

Scope: the three validation figures included in the Harvard writing-sample
derivative. The source numerical ledgers and all statistical calculations are
unchanged; only reader-facing labels and titles were revised.

## Checks

- Recomputed numerical tables match the source-study tables exactly in every
  numeric column: 24 rows x 15 numeric columns for the focused 65 MeV study,
  80 x 15 for the baseline summary, and 80 x 16 for the consolidated summary.
- Text extracted with `pypdf` from all three PDFs contains no occurrence of
  `Table-17`, `Table 17`, or an internal `v4` release label.
- The plotting source contains no occurrence of those same reader-facing
  anti-patterns.
- Visual inspection of all three regenerated PNGs found no clipped titles,
  legends, axes, or annotations.

## Reader-facing vocabulary

- `baseline smooth-threshold model`
- `threshold-refined 65 MeV model`
- `extended-support 65 MeV model`
- `development subset`
- `independent continuation`

## PDF SHA-256

- `figure64_spurious_signal_consolidated_full100_90cl.pdf`:
  `4ab85717c42398e83f112316600afaf7abf47cd423b846446b6e0b8b26ce1f74`
- `figure65_onepctx10_65mev_table17_vs_historical_90cl.pdf`:
  `f5cdd89a68963de3439e37d9b1dfb3ccb04927270a98dbf8566a20b9ccd5b47f`
- `pull_means_widths_consolidated_2x4_full100_90cl.pdf`:
  `366d080923c72ad0886c9c7292feadbe4f5ef7f1446cd2cafed399c3da1acb87`

The legacy wording that remains in one filename is retained only to preserve
the existing TeX include path. It is not visible anywhere in the figure.
