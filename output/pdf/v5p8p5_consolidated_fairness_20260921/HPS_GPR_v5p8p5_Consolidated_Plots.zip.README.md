# Preserved delivery archive

The payload of `HPS_GPR_v5p8p5_Consolidated_Plots.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p5_consolidated_fairness_20260921/HPS_GPR_v5p8p5_Consolidated_Plots.zip \
  --output /tmp/HPS_GPR_v5p8p5_Consolidated_Plots.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `64fc4b8e0bb8e99ca1ca4d018ee66873f78a197a7c22f8a8409b4dc88ce76936`. Full manifest: `publication/recent_studies_20260927/archives/64fc4b8e0bb8e99ca1ca4d018ee66873f78a197a7c22f8a8409b4dc88ce76936.json`.
