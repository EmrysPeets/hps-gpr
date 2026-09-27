# Preserved delivery archive

The payload of `HPS_GPR_v5p8p2_Plots.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p2_nominal_gp_significance_20260917/HPS_GPR_v5p8p2_Plots.zip \
  --output /tmp/HPS_GPR_v5p8p2_Plots.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `f63690a2bc93283da1fba7c0cb099be9cabc01e8c2f22946aba48e65d2288279`. Full manifest: `publication/recent_studies_20260927/archives/f63690a2bc93283da1fba7c0cb099be9cabc01e8c2f22946aba48e65d2288279.json`.
