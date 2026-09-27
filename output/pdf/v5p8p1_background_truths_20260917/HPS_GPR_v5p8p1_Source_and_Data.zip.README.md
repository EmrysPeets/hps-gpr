# Preserved delivery archive

The payload of `HPS_GPR_v5p8p1_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p1_background_truths_20260917/HPS_GPR_v5p8p1_Source_and_Data.zip \
  --output /tmp/HPS_GPR_v5p8p1_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `a4b67c4d5dafcca3104ebf0a32a3198c41a0dbfead2d34d0f1d59aec12eba4d7`. Full manifest: `publication/recent_studies_20260927/archives/a4b67c4d5dafcca3104ebf0a32a3198c41a0dbfead2d34d0f1d59aec12eba4d7.json`.
