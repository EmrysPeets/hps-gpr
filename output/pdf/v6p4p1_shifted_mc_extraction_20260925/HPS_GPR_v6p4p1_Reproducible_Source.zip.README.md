# Preserved delivery archive

The payload of `HPS_GPR_v6p4p1_Reproducible_Source.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p4p1_shifted_mc_extraction_20260925/HPS_GPR_v6p4p1_Reproducible_Source.zip \
  --output /tmp/HPS_GPR_v6p4p1_Reproducible_Source.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `78554d1de10e86c68707c8162f5037c14bb372b88e31a6afaa9899a18096e14f`. Full manifest: `publication/recent_studies_20260927/archives/78554d1de10e86c68707c8162f5037c14bb372b88e31a6afaa9899a18096e14f.json`.
