#!/usr/bin/env python3
"""Build the plain-language LaTeX companion from verified numerical products."""
from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NOTE=HERE/'note'
OUTPUT=ROOT/'output/pdf/v4p9p14_interpretation_global_20260906'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prob(p):
    if p==0:return '0'
    if p>=.001:return f'{p:.4f}'
    mantissa,exponent=f'{p:.2e}'.split('e')
    return r'$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
def tail_text(t):
    return ('fewer than '+prob(t['interval95'][1])+' (95\\% MC upper bound)') if t['k']==0 else prob(t['p'])

def main():
    source=HERE/'global/2015/analysis/summary.json'
    s=json.loads(source.read_text())
    assert s['ensemble_contracts_and_all_point_checks_verified']
    sections=[]
    for method,label in [('profiled','Profiled'),('fixed','Fixed')]:
        x=s['methods'][method];g=x['global_gp'];d=x['global_direct']
        line=(rf'\textbf{{{label} statistic.}} The most extreme point for this new local-probability rule is {x["peak_mass_MeV"]} MeV. '
            'Its common-background local probability is '+prob(x['local_common_truth_p'])+', and the GP estimate of the scan-wide probability is '+tail_text(g)+'. '
            f'Independent direct scans exceed the same threshold in {d["k"]}/{d["n"]} cases; their 95\\% interval is '
            +prob(d['interval95'][0])+'--'+prob(d['interval95'][1])+'.')
        sections.append(line)
        if g['k']<20:
            sections.append(f'Only {g["k"]} of {g["n"]:,} GP fields exceed this threshold. Its 95\\% Monte Carlo interval is '+prob(g['interval95'][0])+'--'+prob(g['interval95'][1])+'. The direct scans do not resolve this rare tail; this is an extrapolative diagnostic, not a validated discovery probability.')
    sections.append('These numbers condition on one archived background. They cannot be used to choose the more favorable statistic or to replace the earlier envelope over different backgrounds. The GP Monte Carlo interval measures sampling precision within the approximation, not its physical validity.')
    (NOTE/'global_results_text.tex').write_text('\n\n'.join(sections)+'\n')
    rows=[]
    for method in ('profiled','fixed'):
        x=s['methods'][method]
        rows.append(method.capitalize()+' & '+f'{x["valid_z_mean_range"][0]:+.3f} to {x["valid_z_mean_range"][1]:+.3f}'+' & '+f'{x["valid_z_sd_range"][0]:.3f} to {x["valid_z_sd_range"][1]:.3f}'+' & '+str(x['marginal_normality_holm_flags'])+r' \\')
    text=(r'\begin{table}[H]\centering\small'+'\n'+r'\begin{tabular}{lrrr}\toprule'+'\n'+r'Method & Standardized mean range & Spread range & Flagged masses\\\midrule'+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n'+r'\caption{Independent 1,000-toy checks after the deterministic centering and scaling. Flags use a Holm-adjusted normality test over 72 masses per method. Lack of a flag does not establish exact Gaussian tails.}\end{table}'+'\n\n')
    text+='The root-mean-square correlation discrepancies between direct scans and the Asimov matrices are '+f'{s["methods"]["profiled"]["correlation_rms_difference"]:.3f}'+' (profiled) and '+f'{s["methods"]["fixed"]["correlation_rms_difference"]:.3f}'+' (fixed). '
    for method in ('profiled','fixed'):
        x=s['methods'][method]
        if x['global_direct']['k']==0:
            text+='The '+method+' tail remains unresolved by direct scans. '
        else:
            text+='For the '+method+' statistic, the GP global estimate '+('lies inside' if x['gp_global_inside_direct_interval95'] else 'lies outside')+' the direct-scan 95\\% interval at the observed threshold. '
    text+='These checks assess the approximation under the chosen spectrum; they do not qualify that spectrum as the physical background.\n'
    coarse=s['methods']['profiled']['coarse_2MeV_global_at_fine_peak']
    text+='The two 2 MeV subgrids give profiled global estimates of '+prob(coarse[0]['p'])+' and '+prob(coarse[1]['p'])+' at the same fine-grid threshold, compared with '+prob(s['methods']['profiled']['global_gp']['p'])+' on the 1 MeV grid. This substantial change illustrates why a continuous-search claim requires a finer-grid check.\n'
    (NOTE/'global_validation_text.tex').write_text(text)
    (NOTE/'generated_values.tex').write_text(r'\newcommand{\GlobalPlotFloorText}{Downward triangles at $10^{-8}$ identify smaller analytic probabilities; zero Monte Carlo tails are instead shown at their 95\% upper bounds.}'+'\n')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(['tectonic','--keep-logs','--outdir',str(OUTPUT),str(NOTE/'reader_report.tex')],cwd=NOTE,capture_output=True,text=True)
    (NOTE/'build.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stderr)
    pdf=OUTPUT/'reader_report.pdf'
    target=OUTPUT/'HPS_GPR_v4p9p14_Calibration_Explained_and_Global_Study.pdf'
    target.write_bytes(pdf.read_bytes());pdf.unlink()
    inputs=[source,Path(__file__),*NOTE.glob('*.tex'),*HERE.joinpath('figures').glob('*.pdf')]
    (HERE/'provenance/report_build.json').write_text(json.dumps({'pdf':str(target),'pdf_sha256':sha(target),'inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs}},indent=2)+'\n')
    print(result.stdout+result.stderr)
    print(target)
if __name__=='__main__':main()
