# Preserved delivery archive

The payload of `HPS_GPR_v6p3p7_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p7_template_windows_20260925/HPS_GPR_v6p3p7_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p7_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `707264018ce36a554a25f62acd8e9d01037d795c89ecedd13e91fe8c192875ba`. Full manifest: `publication/recent_studies_20260927/archives/707264018ce36a554a25f62acd8e9d01037d795c89ecedd13e91fe8c192875ba.json`.
