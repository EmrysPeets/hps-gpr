# Preserved delivery archive

The payload of `HPS_GPR_External_Statistician_Source.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p5_external_statistician_20260921/HPS_GPR_External_Statistician_Source.zip \
  --output /tmp/HPS_GPR_External_Statistician_Source.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `d19ca0c29b75e1907946aaf3da0ea9874a40de4d513fb5748d7912fd614435c8`. Full manifest: `publication/recent_studies_20260927/archives/d19ca0c29b75e1907946aaf3da0ea9874a40de4d513fb5748d7912fd614435c8.json`.
