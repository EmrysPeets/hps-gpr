# Preserved delivery archive

The payload of `parent_2021_reproducible_study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive study_results/v6p3p2_2016_offset_transfer_20260924/inputs/parent_2021_reproducible_study.zip \
  --output /tmp/parent_2021_reproducible_study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `500b4927da8f8aa04ec92bf783935ad6837acc4f8a3f3f3fc259ad07ad636990`. Full manifest: `publication/recent_studies_20260927/archives/500b4927da8f8aa04ec92bf783935ad6837acc4f8a3f3f3fc259ad07ad636990.json`.
