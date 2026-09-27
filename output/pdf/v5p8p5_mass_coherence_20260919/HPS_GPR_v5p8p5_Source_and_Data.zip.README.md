# Preserved delivery archive

The payload of `HPS_GPR_v5p8p5_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p5_mass_coherence_20260919/HPS_GPR_v5p8p5_Source_and_Data.zip \
  --output /tmp/HPS_GPR_v5p8p5_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `e441c2cdb2bd14634b19b18f72acf75546d0761ed2290143481301a8df55e40b`. Full manifest: `publication/recent_studies_20260927/archives/e441c2cdb2bd14634b19b18f72acf75546d0761ed2290143481301a8df55e40b.json`.
