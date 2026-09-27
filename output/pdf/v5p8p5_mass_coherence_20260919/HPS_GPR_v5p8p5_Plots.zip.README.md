# Preserved delivery archive

The payload of `HPS_GPR_v5p8p5_Plots.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p5_mass_coherence_20260919/HPS_GPR_v5p8p5_Plots.zip \
  --output /tmp/HPS_GPR_v5p8p5_Plots.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `57c324eddcd2fc567bcf885647b1b56103a5f41af55b59210a2b0cd168298c4e`. Full manifest: `publication/recent_studies_20260927/archives/57c324eddcd2fc567bcf885647b1b56103a5f41af55b59210a2b0cd168298c4e.json`.
