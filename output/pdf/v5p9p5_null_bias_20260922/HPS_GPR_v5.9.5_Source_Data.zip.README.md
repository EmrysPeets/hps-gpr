# Preserved delivery archive

The payload of `HPS_GPR_v5.9.5_Source_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p9p5_null_bias_20260922/HPS_GPR_v5.9.5_Source_Data.zip \
  --output /tmp/HPS_GPR_v5.9.5_Source_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `ada084c41f9758aa1f8d8651dd80b2462f115eb0b87b76ad9889b03d54b28a73`. Full manifest: `publication/recent_studies_20260927/archives/ada084c41f9758aa1f8d8651dd80b2462f115eb0b87b76ad9889b03d54b28a73.json`.
