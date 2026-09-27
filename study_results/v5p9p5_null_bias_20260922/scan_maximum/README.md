# Scan-maximum audit for HPS-GPR v5.9.5

Read `findings.md` for the interpretation. All plots and tables retain the raw observed scan. New simulations are source-conditional Gaussian controls; no likelihood refit or data change was made.

With Python, NumPy, SciPy and Matplotlib installed, run from any directory:

```sh
python3 scripts/audit.py
python3 scripts/make_figures.py
```

When the parent repository exists, `audit.py` verifies bundled inputs against it. When copied elsewhere, it uses the bundled NPZ files and verifies their saved hashes. A parent-free rebuild reproduced all three numerical CSV ledgers byte for byte; see `provenance/portable_rebuild.json`.

`provenance/validation.json` records 13 passed checks, including source identities, the meaning of K and D, slide-50 counts, the saved/new nominal-tail comparison and exact independent/perfect-correlation control formulas. Figure PNGs were visually reviewed; vector PDF companions are supplied. Monte Carlo intervals describe finite simulation precision only.
