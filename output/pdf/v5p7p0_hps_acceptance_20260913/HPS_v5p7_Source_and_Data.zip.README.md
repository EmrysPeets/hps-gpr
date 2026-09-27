# Preserved delivery archive

The payload of `HPS_v5p7_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p7p0_hps_acceptance_20260913/HPS_v5p7_Source_and_Data.zip \
  --output /tmp/HPS_v5p7_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `cbc1fec92e58de7f822cfcc3d00526f29e524f1f91f0bd920c7183aa74693977`. Full manifest: `publication/recent_studies_20260927/archives/cbc1fec92e58de7f822cfcc3d00526f29e524f1f91f0bd920c7183aa74693977.json`.
