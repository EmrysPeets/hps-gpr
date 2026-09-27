# Preserved delivery archive

The payload of `HPS_GPR_v6p4p5_Mass_Correlation_LEE_Source.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p4p5_mass_correlation_lee_20260926/HPS_GPR_v6p4p5_Mass_Correlation_LEE_Source.zip \
  --output /tmp/HPS_GPR_v6p4p5_Mass_Correlation_LEE_Source.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `cb9ef7b447927fdb6682c67738d7b9d5c37963d74061f431d4d5aeda9419b76f`. Full manifest: `publication/recent_studies_20260927/archives/cb9ef7b447927fdb6682c67738d7b9d5c37963d74061f431d4d5aeda9419b76f.json`.
