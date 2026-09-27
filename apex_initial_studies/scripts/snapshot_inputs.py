"""One-time snapshot of the existing, explicitly selected HPS result ledgers."""
from pathlib import Path
import hashlib,json,shutil,subprocess,platform
import numpy as np

B=Path(__file__).resolve().parents[1]; R=B.parent
F=R/'study_results/v5p0p4_figure2_publication_20260913'
P=R/'study_results/v5p6p1_2021_upper_limit_echoes_20260913'
files={
 'hps_combined_v504.csv':F/'inputs/v504_union.csv',
 'hps_figure2_contours.csv':F/'derived/projected_contours.csv',
 'hps_2021_released.csv':P/'inputs/parent/released_2021_asymptotic.csv',
 'hps_source_scan.csv':P/'inputs/source_scan.csv',
 'hps_projection_catalogue.csv':P/'inputs/catalogue.csv',
 'hps_lane_manifest.json':P/'inputs/lane_manifest.json',
 'hps_figure2_README.md':F/'README.md',
 'hps_v561_README.md':P/'README.md',
}
for scenario in ['ten_160','ten_210','one_210','one_extra244']:
    for spectrum in ['background_asimov','matched_asimov','yield_asimov']:
        files[f'{scenario}_{spectrum}.csv']=P/'derived/scans'/scenario/f'{spectrum}.csv'
ledger=[]
for name,src in files.items():
    dst=B/'inputs'/name; shutil.copy2(src,dst)
    ledger.append(dict(source=str(src),snapshot=str(dst.relative_to(B)),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
coeff={}
for year in ['2016','2021']:
    p=F/'inputs'/f'spectrum_{year}.npz'
    with np.load(p) as d:coeff[year]=dict(sigma_polynomial_GeV=d['sigma_coeffs'].tolist(),source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(B/'inputs/hps_resolution.json').write_text(json.dumps(coeff,indent=2))
(B/'provenance/hps_inputs.json').write_text(json.dumps(ledger,indent=2))
(B/'provenance/environment.json').write_text(json.dumps(dict(
    date='2026-09-16',python=platform.python_version(),platform=platform.platform(),
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
    git_status_at_snapshot=subprocess.check_output(['git','status','--short'],cwd=R,text=True),
    scope='New standalone derivative; parent inputs are read only; no new toys or GPR fits.'),indent=2))
print('Pinned',len(ledger),'HPS files')
