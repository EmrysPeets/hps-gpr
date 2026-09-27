# Preserved delivery archive

The payload of `HPS_GPR_v5p6p1_2021_Upper_Limit_Echoes_Source_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p6p1_2021_upper_limit_echoes_20260913/HPS_GPR_v5p6p1_2021_Upper_Limit_Echoes_Source_Data.zip \
  --output /tmp/HPS_GPR_v5p6p1_2021_Upper_Limit_Echoes_Source_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `a5f14c236b33fcb2a0b808a413064ea52940839b7ea2ed327cb702a5c133a725`. Full manifest: `publication/recent_studies_20260927/archives/a5f14c236b33fcb2a0b808a413064ea52940839b7ea2ed327cb702a5c133a725.json`.
