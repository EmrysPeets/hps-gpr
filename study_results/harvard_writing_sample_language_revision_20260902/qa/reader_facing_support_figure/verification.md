# Reader-facing support figure verification

- Source data: `v4p9p5_2021_gp_support_edge_optimization_20260820/derived/analysis/confirmation_cell_summary.csv`
- Source-data SHA-256: `2500ea319390bdf87bb6afefafa7e1707c19cb137c5382f86b7e7f5cdd69fbdd`
- Adapted script: `scripts/make_reader_facing_support_figure.py`
- PDF SHA-256: `414a7b949afbb93270f969ea88922bae7902f58c1f5f8b7ac05b2c9bcc5f4af5`
- PNG SHA-256: `947cbfa1ae169d8e79a9324a19ba29b251611731274f4d603dcac1d3c9449086`
- Original/derivative Matplotlib artist-data SHA-256: `4565e8204db2d358c48ab741c133c85d7483d852ad953ba2abe8cec60b294bfd`
- Compared numeric artists: 3 axes, 36 lines, 36 error-bar segments, and 72 path vertices; exact match.
- PDF text check: new title present; old title absent; one page.
- Visual check: the 180-dpi PDF rendering has no clipping, overlap, or unreadable labels.

The only intended plot-content change is the main title, now
`Support comparison using 100 pseudoexperiments`.
