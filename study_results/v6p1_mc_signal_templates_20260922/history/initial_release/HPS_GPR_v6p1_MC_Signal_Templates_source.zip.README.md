# Preserved delivery archive

The payload of `HPS_GPR_v6p1_MC_Signal_Templates_source.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive study_results/v6p1_mc_signal_templates_20260922/history/initial_release/HPS_GPR_v6p1_MC_Signal_Templates_source.zip \
  --output /tmp/HPS_GPR_v6p1_MC_Signal_Templates_source.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `99069667daacbb28625f52b325b30b5d0e46e4cff4c9b44a9ff82dff38814172`. Full manifest: `publication/recent_studies_20260927/archives/99069667daacbb28625f52b325b30b5d0e46e4cff4c9b44a9ff82dff38814172.json`.
