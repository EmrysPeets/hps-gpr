# Preserved delivery archive

The payload of `HPS_GPR_v6p4p2_Reproducible_Source.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p4p2_2016_window_comparison_20260925/HPS_GPR_v6p4p2_Reproducible_Source.zip \
  --output /tmp/HPS_GPR_v6p4p2_Reproducible_Source.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `d5b2de028d6c7110777e36a41c65e96096e3490d966e0595b5ffb5cffa2adf30`. Full manifest: `publication/recent_studies_20260927/archives/d5b2de028d6c7110777e36a41c65e96096e3490d966e0595b5ffb5cffa2adf30.json`.
