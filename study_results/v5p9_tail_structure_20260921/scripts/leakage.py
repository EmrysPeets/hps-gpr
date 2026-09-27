"""Deterministic signal-in-training response, not an ensemble or coverage test."""
from tail_model import *

def main():
    scans=pd.read_csv(B/'derived/scans.csv',dtype={'scope':str})
    rows=[]
    for year,anchors in {'2015':[51,92],'2016':[90,92],'2021':[78,92]}.items():
        d=DATA[year]
        for mass in anchors:
            nominal=scans.query("scope == @year and mass_MeV == @mass and window == 'primary' and family == 'gaussian'").iloc[0]
            amplitude=float(nominal.A90);scale=signal_scale(year,mass)
            for window,padding in WINDOWS.items():
                p=context(year,[mass],padding=padding,anchor=mass);mask=p['mask']
                # Smooth, observed-conditioned reference spectrum from exterior data.
                n0,_=predict(d['x'],d['n'],mask,p['const'],p['ls'],query=d['x'])
                b0,C0=predict(d['x'],n0,mask,p['const'],p['ls']);L0,diag0=factor_cov(C0,b0)
                for family,k in SCENARIOS:
                    w,meta=weights(year,mass,k,family);s=scale*w
                    b1,C1=predict(d['x'],n0+amplitude*s,mask,p['const'],p['ls']);L1,diag1=factor_cov(C1,b1)
                    # Center the uninjected fitted bins on their own reconstructed GP.
                    n=b0+amplitude*s[mask]
                    for extraction in ('matched','gaussian'):
                        fit_w=w if extraction=='matched' else weights(year,mass,1.,'gaussian')[0]
                        for training,b,L in [('clean',b0,L0),('contaminated',b1,L1)]:
                            model=OneSignalProfile(b,L,scale*fit_w[mask]);f=model.fit(n);f0=model.fit(n,0.)
                            r=float(np.sign(f['A'])*np.sqrt(max(0.,2*(f0['nll']-f['nll']))))
                            rows.append(dict(scope=year,mass_MeV=mass,window=window,family=family,kappa=k,
                                extraction=extraction,training=training,injected_A=amplitude,recovered_A=f['A'],
                                recovery=f['A']/amplitude,signed_r=r,max_score=max(f['score'],f0['score']),
                                min_lambda=min(f['min_lambda'],f0['min_lambda']),
                                injected_full_counts=amplitude*scale,
                                injected_training_counts=float(amplitude*s[~mask].sum()),
                                gp_shift_fit_counts=float((b1-b0).sum()),
                                gp_shift_over_injected_window=float((b1-b0).sum()/(amplitude*s[mask].sum()))))
            print('leakage',year,mass,flush=True)
    f=pd.DataFrame(rows);f.to_csv(B/'derived/leakage_response.csv',index=False,float_format='%.17g')
    check=f.query("extraction=='matched' and training=='clean'")
    write(B/'qa/leakage_validation.json',dict(rows=len(f),maximum_clean_matched_recovery_error=float(abs(check.recovery-1).max()),
          maximum_score=float(f.max_score.max()),minimum_lambda=float(f.min_lambda.min()),
          meaning='Deterministic conditional mean-response at six anchors. No Poisson sampling, no coverage or global calibration.'))

if __name__=='__main__':main()
