#!/usr/bin/env python3
"""Generate tables from accepted combined products and compile the analysis note."""
from pathlib import Path
import hashlib,json,subprocess
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NOTE=HERE/'note'
OUT=ROOT/'output/pdf/v4p9p16_combined_global_20260906'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prob(x):
    if x==0:return '0'
    if x>=.001:return f'{x:.4f}'.rstrip('0').rstrip('.')
    a,b=f'{x:.2e}'.split('e')
    return '$'+a+r'\times10^{'+str(int(b))+'}$'
def estimate(t):
    return '$<$'+prob(t['upper95_one_sided']) if t['k']==0 else prob(t['p'])
def table(header,rows,caption,spec):
    return ('\\begin{table}[H]\\centering\\small\n\\begin{tabular}{'+spec+'}\\toprule\n'
        +header+r'\\\midrule'+'\n'
        +'\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
        +'\n'+r'\bottomrule\end{tabular}'+'\n'+r'\caption{'+caption+'}\\end{table}\n')
def write(name,text):(NOTE/name).write_text(text+'\n')
def main():
    folder=HERE/'global'
    s=json.loads((folder/'analysis/summary.json').read_text())
    qa=json.loads((HERE/'qa/numerical_validation.json').read_text())
    execution=json.loads((folder/'summary.json').read_text())
    assert qa['passed'] and s['numerical_audit_passed']
    d=pd.read_csv(folder/'analysis/pvalue_curves.csv')
    x=s['methods']['profiled'];g=x['global_gp'];t=x['global_direct']
    if t['k']==0:
        headline=('The principal profiled scan has no exceedance in 1,000 direct joint experiments. '
            'Its conditional probability is unresolved; the one-sided 95\\% upper bound is '
            +prob(t['upper95_one_sided'])+'. The Gaussian field also '
            +('has no sampled exceedance' if g['k']==0 else 'has only '+str(g['k'])+' sampled exceedances')
            +'. An extreme stress-centered score is not an established particle significance.')
    else:
        headline=('For the principal profiled scan, the GP global estimate is '+estimate(g)
            +f", compared with {t['k']}/1,000 direct joint exceedances (central 95\\% interval "
            +prob(t['interval95'][0])+'--'+prob(t['interval95'][1])+'). '
            +'This comparison tests the GP approximation under the stated joint background.')
    write('headline.tex',headline)
    rows=[]
    for method in ('profiled','fixed'):
        q=s['methods'][method]
        rows.append([method.capitalize(),q['peak_mass_MeV'],prob(q['local_common_truth_p']),
            estimate(q['global_gp']),f"{q['global_direct']['k']}/1000"])
    write('summary_table.tex',table(r'Statistic & Mass [MeV] & Local $p$ & GP global $p$ & Direct count',
        rows,r'Principal minimum-local-p ordering under the common stress background. Less-than signs are one-sided 95\% Monte Carlo upper bounds, not measured zero probabilities. Fixed-background results are a separate diagnostic method.','lrrrr'))
    write('scope_table.tex',table('Mass [MeV] & Active data',
        [['19--38','2015 full'],['39--49','2015 full + 2016 full'],
         ['50--90',r'2015 full + 2016 full + 2021 10\%'],
         ['91--180',r'2016 full + 2021 10\%'],['181--250',r'2021 10\%']],
        'The complete search uses 1 MeV spacing and established dataset ranges.','ll'))
    rows=[];texts=[]
    for method in ('profiled','fixed'):
        q=s['methods'][method]
        rows.append([method.capitalize(),f"{q['valid_z_mean_range'][0]:+.3f} to {q['valid_z_mean_range'][1]:+.3f}",
            f"{q['valid_z_sd_range'][0]:.3f} to {q['valid_z_sd_range'][1]:.3f}",
            q['marginal_normality_holm_flags']])
        texts.append(method.capitalize()+': the maximum-distribution KS diagnostic gives nominal p='
            +prob(q['principal_maximum_KS']['pvalue'])+' for the principal ordering and '
            +prob(q['raw_maximum_KS']['pvalue'])+' for the raw ordering. The correlation RMS difference is '
            +f"{q['correlation_rms_difference']:.3f}"+'.')
    write('marginal_table.tex',table('Statistic & Standardized mean range & Spread range & Flags',
        rows,'Normality flags use a Holm adjustment across 232 masses within each method. Non-rejection has finite power and cannot certify extreme Gaussian tails.','lrrr'))
    texts.append('These diagnostics compare simulation methods, not the data significance. Two methods are checked per ordering. Direct Poisson counts calibrate the declared score under the generating spectrum without requiring a jointly Gaussian field; the GP estimate has that additional approximation.')
    if min(q[k]['pvalue'] for q in s['methods'].values() for k in ('principal_maximum_KS','raw_maximum_KS'))<.05:
        texts.append('At least one nominal diagnostic is below 0.05. This approximation warning is retained; it is neither a multiplicity-adjusted conclusion nor a far-tail validation.')
    write('tail_diagnostics.tex','\n\n'.join(texts))
    rows=[]
    for m in s['representative_masses_MeV']:
        v=d[(d.method=='profiled') & (d.mass_MeV==m)].iloc[0]
        gp=dict(k=int(v.gp_k),p=v.p_global_gp,upper95_one_sided=v.p_global_gp_upper95)
        raw=dict(k=int(v.raw_gp_k),p=v.p_global_raw_ordering,upper95_one_sided=v.raw_gp_upper95)
        rows.append([m,v.dataset_set,prob(v.p_asymptotic),prob(v.p_local_common_truth),estimate(gp),
                     f'{int(v.direct_k)}/1000',estimate(raw)])
    write('representative_table.tex',table(r'Mass & Active years & Asymp. local & GP local & GP global & Direct & Raw global',
        rows,r'Profiled representative masses: the principal and raw extrema, plus 30, 65, 120 and 220 MeV. GP local uses the common-background Gaussian response. Both global columns use the full union at the threshold observed at that mass. Raw global is the separate raw-peak test.','rlrrrrr'))
    rows=[]
    for method in ('profiled','fixed'):
        q=s['methods'][method]
        rows.append([method.capitalize(),q['peak_mass_MeV'],f"{q['observed_raw_r']:+.3f}",
            f"{q['asimov_r']:+.3f}",f"{q['response_sd']:.3f}",f"{q['observed_standardized_r']:+.3f}"])
    write('peak_table.tex',table(r'Statistic & Mass & Raw $r$ & Offset $a$ & Width $s$ & $(r-a)/s$',
        rows,'Principal peak decomposition. The final column is a conditional score, not an independently established discovery significance.','lrrrrr'))
    raw=x['raw_ordering']
    interpretation=('At the profiled principal peak, the unadjusted asymptotic local probability is '
        +prob(x['local_asymptotic_p'])+', while centering and scaling against the stress background gives '
        +prob(x['local_common_truth_p'])+'. ')
    if abs(x['observed_raw_r'])<1 and abs(x['asimov_r'])/x['response_sd']>5:
        interpretation+='The observed raw root is close to zero; the extreme centered score arises mainly because the stress background predicts a large negative root. '
    before=d[(d.method=='profiled') & d.mass_MeV.isin([x['peak_mass_MeV']-2,x['peak_mass_MeV']-1])]
    if len(before)==2 and (before.observed_r<=0).all():
        interpretation+='\n\nAt '+str(int(before.mass_MeV.iloc[0]))+' and '+str(int(before.mass_MeV.iloc[1]))+' MeV, the centered scores are '
        interpretation+=f"{before.z_standardized.iloc[0]:.2f} and {before.z_standardized.iloc[1]:.2f}, but the raw roots are negative. The declared boundary therefore assigns local p=1 there. "
        interpretation+='The next positive raw root becomes the principal peak. This abrupt change comes from the signal boundary together with the stress offset; it is not a numerical sign flip. '
    interpretation+='\n\nThe raw maximum occurs at '+str(raw['peak_mass_MeV'])+' MeV. '
    if raw['global_gp']['k']==raw['global_gp']['n'] and raw['global_direct']['k']==raw['global_direct']['n']:
        interpretation+='All 200,000 GP fields and all 1,000 direct experiments exceed it. '
    else:
        interpretation+='It gives GP global '+estimate(raw['global_gp'])+' and direct '+str(raw['global_direct']['k'])+'/1000. '
    interpretation+='These differences reflect the specified orderings and background offsets; they are not a free choice of significance.'
    write('peak_interpretation.tex',interpretation)
    rows=[]
    for method in ('profiled','fixed'):
        q=s['methods'][method]
        rows.append([method.capitalize(),estimate(q['global_gp']),
                     *[estimate(v) for v in q['coarse_2MeV_global_at_fine_peak']]])
    write('grid_table.tex',table('Statistic & Full 1 MeV grid & First 2 MeV subgrid & Second subgrid',
        rows,r'GP probabilities at the same original peak threshold. Zero-count bounds do not establish fine-tail convergence.','lrrr'))
    comparison=s['v12_comparison']
    write('limit_comparison.tex',
        f"The largest absolute relative change from the corresponding v4.9.12 limit is {100*comparison['max_absolute_relative_limit_difference']:.3f}\\%. "
        f"The largest difference in the nonnegative discovery root is {comparison['max_absolute_bounded_root_difference']:.4f}. "
        'Fresh dense fits reproduce v4.9.13 dense references where the same scope exists. '
        'The old optimizer does not select a preferred answer. Above '+f'{2*105.6583745:.3f} MeV, displayed limits include '
        r'$1+\sqrt{1-4m_\mu^2/m^2}(1+2m_\mu^2/m^2)$ once; the CSV retains the electron-channel values.')
    records=qa['numerical']
    maxroot=max(v['pilot_max_root_error'] for v in records)
    maxresp=max(v.get('relative_l2_error',0) for v in records)
    maxcorr=max(v['max_absolute_error'] for v in qa['sentinel_correlations'].values())
    fallback=execution['exact_fallback_masses']
    write('numerical_summary.tex',
        f"All {qa['check_count']} numerical and product-identity checks passed. "
        "There are 142 newly fitted joint coordinates and 90 reused single-dataset coordinates. "
        f"Summed coordinate execution took {execution['coordinate_seconds']/60:.1f} minutes with one worker and one BLAS thread. "
        f"The exact backend was retained at {len(fallback)} new coordinates"
        +(' ('+', '.join(map(str,fallback))+' MeV)' if fallback else '')+'. '
        'The largest paired pilot-root difference is '+prob(maxroot)
        +', the largest complete-response relative vector error is '+prob(maxresp)
        +', and the largest sentinel-correlation error is '+prob(maxcorr)
        +'. No paired pilot changed its raw-positive classification.')
    OUT.mkdir(parents=True,exist_ok=True)
    run=subprocess.run(['tectonic','--keep-logs','--outdir',str(OUT),str(NOTE/'analysis_note.tex')],
        cwd=NOTE,capture_output=True,text=True)
    (NOTE/'build.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError(run.stderr)
    rawpdf=OUT/'analysis_note.pdf'
    pdf=OUT/'HPS_GPR_Analysis_Note_v4p9p16_Combined_Global_Search.pdf'
    pdf.write_bytes(rawpdf.read_bytes());rawpdf.unlink()
    inputs=[Path(__file__),HERE/'PROTOCOL.md',folder/'observed.csv',
        folder/'summary.json',folder/'analysis/summary.json',folder/'analysis/pvalue_curves.csv',
        HERE/'qa/numerical_validation.json',HERE/'provenance/figure_build.json',
        *NOTE.glob('*.tex'),*HERE.joinpath('figures').glob('*.pdf')]
    (HERE/'provenance/report_build.json').write_text(json.dumps(dict(
        pdf=str(pdf),pdf_sha256=sha(pdf),inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs}),indent=2)+'\n')
    print(run.stdout+run.stderr);print(pdf)

if __name__=='__main__':main()
