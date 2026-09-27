# Preserved delivery archive

The payload of `HPS_GPR_v5p8p3_Summary_Plots.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p3_global_interpretation_20260917/HPS_GPR_v5p8p3_Summary_Plots.zip \
  --output /tmp/HPS_GPR_v5p8p3_Summary_Plots.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `49d42880d9c966e54c209b5836240f2bb8108a5d93f8b76751dfde03bcd80a2c`. Full manifest: `publication/recent_studies_20260927/archives/49d42880d9c966e54c209b5836240f2bb8108a5d93f8b76751dfde03bcd80a2c.json`.
