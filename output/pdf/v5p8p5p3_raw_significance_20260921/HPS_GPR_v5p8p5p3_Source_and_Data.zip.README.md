# Preserved delivery archive

The payload of `HPS_GPR_v5p8p5p3_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p5p3_raw_significance_20260921/HPS_GPR_v5p8p5p3_Source_and_Data.zip \
  --output /tmp/HPS_GPR_v5p8p5p3_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `0b3e51552b972dfac0d3e76a15a7258ac75b2a7b6ce38c2f40ef4aea691306aa`. Full manifest: `publication/recent_studies_20260927/archives/0b3e51552b972dfac0d3e76a15a7258ac75b2a7b6ce38c2f40ef4aea691306aa.json`.
