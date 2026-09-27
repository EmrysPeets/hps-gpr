"""Copy the earlier reader's report without changing its released artifacts."""
from pathlib import Path
import shutil, hashlib, json
B=Path(__file__).resolve().parents[1]
parent=B.parent/'v5p8p5_external_statistician_20260921'
for folder in ['source','figures','qa','results','reviews','inputs']:
    (B/folder).mkdir(exist_ok=True)
ledger=[]
for p in sorted(parent.rglob('*')):
    if p.is_file():
        ledger.append({'path':str(p.relative_to(B.parents[1])), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(B/'qa/parent_manifest.json').write_text(json.dumps(ledger,indent=2)+'\n')
shutil.copytree(parent/'inputs',B/'inputs/earlier_review',dirs_exist_ok=True)
shutil.copy2(parent/'figures/observed_evidence.pdf',B/'figures/observed_evidence.pdf')
shutil.copy2(parent/'reviews/fixed_background_audit.md',B/'reviews/earlier_fixed_background_audit.md')
t=(parent/'source/report.tex').read_text()
t=t.replace('This is a review of saved results; no new fits are performed.', 'I now add a direct calculation with the estimated background held fixed in each signal likelihood, using the same event inputs and $\\pm2.25\\sigma_m$ masks. The earlier profiled results remain the comparison.')
t=t.replace('The answer depends on what is fixed.', 'The earlier results in this section profile background uncertainty. The new fixed-background calculation follows on the next pages. The distinction matters.')
t=t.replace('I did not recover a verified fixed-covariance combined discovery scan, so I do not assign it a peak significance here.', 'The new calculation below supplies the previously missing fixed-covariance discovery scans and checks their raw statistics against complete null toy scans.')
t=t.replace('\\clearpage\n\\section*{What additional 2021 events could clarify}', '\\clearpage\n\\input{fixed_background.tex}\n\\clearpage\n\\section*{What additional 2021 events could clarify}')
t=t.replace('No new fit, significance calibration or exposure forecast was produced.', 'This extension adds fixed-background fits and conditional null checks, with no new exposure forecast. Reproduction inputs and code accompany the report.')
(B/'source/report.tex').write_text(t)
print('Copied report and pinned',len(ledger),'earlier artifacts.')
