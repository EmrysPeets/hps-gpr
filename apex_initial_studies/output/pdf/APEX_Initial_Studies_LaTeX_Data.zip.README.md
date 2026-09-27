# Preserved delivery archive

The payload of `APEX_Initial_Studies_LaTeX_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive apex_initial_studies/output/pdf/APEX_Initial_Studies_LaTeX_Data.zip \
  --output /tmp/APEX_Initial_Studies_LaTeX_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `bece6bbea4cbd90c718e5fbb10ed41be04c84f2c5b3bf6521740bb97189b1ff4`. Full manifest: `publication/recent_studies_20260927/archives/bece6bbea4cbd90c718e5fbb10ed41be04c84f2c5b3bf6521740bb97189b1ff4.json`.
