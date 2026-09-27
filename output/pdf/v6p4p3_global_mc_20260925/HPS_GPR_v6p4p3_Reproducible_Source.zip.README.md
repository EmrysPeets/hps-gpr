# Preserved delivery archive

The payload of `HPS_GPR_v6p4p3_Reproducible_Source.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p4p3_global_mc_20260925/HPS_GPR_v6p4p3_Reproducible_Source.zip \
  --output /tmp/HPS_GPR_v6p4p3_Reproducible_Source.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `5c114ae562b6b064351618b1c7eb16b292b5d12b8a19db8f37b0e5d09f5d422f`. Full manifest: `publication/recent_studies_20260927/archives/5c114ae562b6b064351618b1c7eb16b292b5d12b8a19db8f37b0e5d09f5d422f.json`.
