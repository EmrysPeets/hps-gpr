# Preserved delivery archive

The payload of `HPS_GPR_v6p1_MC_Signal_Templates_source.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p1_mc_signal_templates_20260922/HPS_GPR_v6p1_MC_Signal_Templates_source.zip \
  --output /tmp/HPS_GPR_v6p1_MC_Signal_Templates_source.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `3880264ea1d77c5a8adf3ade0cda0b390f4691c48b970003c479c90893cb409b`. Full manifest: `publication/recent_studies_20260927/archives/3880264ea1d77c5a8adf3ade0cda0b390f4691c48b970003c479c90893cb409b.json`.
