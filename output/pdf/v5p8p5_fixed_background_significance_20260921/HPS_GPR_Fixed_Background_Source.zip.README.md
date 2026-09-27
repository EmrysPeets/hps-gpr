# Preserved delivery archive

The payload of `HPS_GPR_Fixed_Background_Source.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p5_fixed_background_significance_20260921/HPS_GPR_Fixed_Background_Source.zip \
  --output /tmp/HPS_GPR_Fixed_Background_Source.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `781c3e35adbed7254ba3631333ffaabc82667eb7153300b283901c7857ddd316`. Full manifest: `publication/recent_studies_20260927/archives/781c3e35adbed7254ba3631333ffaabc82667eb7153300b283901c7857ddd316.json`.
