#!/usr/bin/env python3
"""Build the report table from the portable historical-row audit."""
from pathlib import Path
import csv
ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "audit/native2021_10pct_summary.csv").open() as handle:
    rows = list(csv.DictReader(handle))
lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
         r"Mass [MeV] & Mean $s_t$ & SD$(s_t)$ & CV [\%] & $\operatorname{Corr}(s_t,u_t)$\\",
         r"\midrule"]
for row in rows:
    values = [f"{float(row['mass_MeV']):.0f}", f"{float(row['sigma_ref_mean']):,.1f}",
              f"{float(row['sigma_ref_sd']):.1f}", f"{100*float(row['sigma_ref_cv']):.3f}",
              f"{float(row['corr_sigma_ref_Ahat0']):+.3f}"]
    lines.append(" & ".join(values) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "source/archive_table.tex").write_text("\n".join(lines)+"\n")
