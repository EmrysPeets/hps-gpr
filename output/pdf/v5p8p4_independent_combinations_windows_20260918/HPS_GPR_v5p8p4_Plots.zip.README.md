# Preserved delivery archive

The payload of `HPS_GPR_v5p8p4_Plots.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p4_independent_combinations_windows_20260918/HPS_GPR_v5p8p4_Plots.zip \
  --output /tmp/HPS_GPR_v5p8p4_Plots.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `b95f55515362076bf107aa3be865aed621ead93bb02a669a62d46f26a1f55c99`. Full manifest: `publication/recent_studies_20260927/archives/b95f55515362076bf107aa3be865aed621ead93bb02a669a62d46f26a1f55c99.json`.
