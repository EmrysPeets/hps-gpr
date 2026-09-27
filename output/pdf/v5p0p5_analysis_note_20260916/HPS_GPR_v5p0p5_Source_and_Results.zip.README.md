# Preserved delivery archive

The payload of `HPS_GPR_v5p0p5_Source_and_Results.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p0p5_analysis_note_20260916/HPS_GPR_v5p0p5_Source_and_Results.zip \
  --output /tmp/HPS_GPR_v5p0p5_Source_and_Results.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `97ada95130ae5c1999b07d952940c3ea0eb8020ed151f6598b61849eaa5fcf13`. Full manifest: `publication/recent_studies_20260927/archives/97ada95130ae5c1999b07d952940c3ea0eb8020ed151f6598b61849eaa5fcf13.json`.
