# Preserved delivery archive

The payload of `HPS_GPR_v6p3p8_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p8_observed_morph_20260925/history/before_blind_panel_combination/HPS_GPR_v6p3p8_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p8_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `2cf0e5a4031c4d044680efb491c745de228704747062ec84161f999cf263e74f`. Full manifest: `publication/recent_studies_20260927/archives/2cf0e5a4031c4d044680efb491c745de228704747062ec84161f999cf263e74f.json`.
