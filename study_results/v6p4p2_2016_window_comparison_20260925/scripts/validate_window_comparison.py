"""Independent geometry, GP, seed, summary and parent-preservation checks."""
from run_window_comparison import *
def main():
    obs=pd.read_csv(R/'observed_scan.csv',float_precision='round_trip');g=pd.read_csv(R/'geometry.csv',float_precision='round_trip')
    toys=pd.read_csv(R/'toy_rows.csv',float_precision='round_trip',keep_default_na=False);resp=pd.read_csv(R/'signal_response.csv',float_precision='round_trip');nc=pd.read_csv(R/'null_checks.csv',float_precision='round_trip')
    checks=0
    def check(x):
        nonlocal checks
        assert x;checks+=1
    parent=json.loads((B/'provenance/parent_v641_hashes.json').read_text())
    for p,h in parent.items():
        if p=='pdf/report.pdf':continue
        if p=='results/summary.json':
            old=json.loads((B/'provenance/parent_summary_v641.json').read_text());new=json.loads((B/p).read_text())
            old.pop('runtime_seconds',None);new.pop('runtime_seconds',None);check(old==new)
        else:check(sha(B/p)==h)
    for line in (B/'provenance/input_manifest.sha256').read_text().splitlines():
        h,p=line.split('  ',1);check(sha(B/p)==h)
    check(len(obs)==634);check(obs.valid.all());check(len(toys)==8000);check(toys.valid.all());check(set(toys.study)=={'null','injection'})
    for m in range(40,176):
        a=WindowContext(m,3.5);b=WindowContext(m,2.)
        check(np.array_equal(a.categories,b.categories));check(np.all(~b.fit|a.fit));check(np.array_equal(b.fit,b.guard))
        direct=np.abs((b.data['x']*1000-b.center)/b.width)<=2
        check(np.array_equal(direct,b.fit));check(abs(b.probability[b.fit].sum()+b.probability[~b.fit].sum()+b.categories[0]+b.categories[-1]-1)<1e-12)
    replay=[]
    for m in (40,60,69,91,100,160,175):
        c=WindowContext(m,2.);b,v=c.prediction(c.data['n']);bb,vv=C.predict(c.data['x'],c.data['n'],c.guard,c.const,c.ls)
        check(np.array_equal(b,bb));check(np.array_equal(v,vv))
        row=solve([c.part()],m,'mc_2u','2016');z=obs[(obs.scope=='2016')&(obs.mass_MeV==m)&(obs.half_width_u==2.)].iloc[0]
        check(abs(row['A90']/z.A90-1)<1e-10);check(abs(row['signed_root']-z.signed_root)<1e-9);replay.append(m)
    a=obs[obs.scope=='combined'].pivot(index='mass_MeV',columns='half_width_u',values='epsilon2_90_visible_legacy');check(np.array_equal(a.loc[176:,2.],a.loc[176:,3.5]))
    check((toys.groupby(['study','mass_MeV','toy','injected']).counts_hash.nunique()==1).all())
    check((toys.realized_full_signal==toys.realized_signal_in_fit+toys.realized_signal_in_training+toys.realized_signal_outside).all())
    truth=np.load(B/'inputs/null_2016.npz')['truth'];replays=0
    lookup={(int(m),h):WindowContext(int(m),h) for m in toys.mass_MeV.unique() for h in WIDTHS}
    for (study,m,toy),q in toys.groupby(['study','mass_MeV','toy']):
        m=int(m);toy=int(toy);signal=np.zeros(len(C.DATA['2016']['n'])+2,dtype=np.int64)
        seed=[641250925,2016,toy] if study=='null' else [642250925,1,2016,toy]
        bg=np.random.default_rng(np.random.SeedSequence(seed)).poisson(truth)
        if study=='injection':
            E=float(q.A_expected.max());cats=lookup[m,2.].categories
            signal=np.random.default_rng(np.random.SeedSequence([642250925,2,m,2016,toy])).poisson(E*cats)
        for injected,s in q.groupby('injected'):
            sig=signal if injected else signal*0;counts=bg+sig[1:-1];h=hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest()
            check((s.counts_hash==h).all());check((s.realized_full_signal==sig.sum()).all())
            for r in s.itertuples():
                c=lookup[m,r.half_width_u]
                check(r.realized_signal_in_fit==sig[1:-1][c.fit].sum());check(r.realized_signal_in_training==sig[1:-1][~c.guard].sum())
                if toy==0 and m in (69,91) and r.half_width_u==2.:
                    z=solve([c.part(counts)],m,'mc_2u','2016',limit=False);check(abs(z['Ahat']-r.Ahat)<1e-7);replays+=1
    for r in nc.itertuples():
        q=toys[(toys.study=='null')&(toys.mass_MeV==r.mass_MeV)&(toys.half_width_u==r.half_width_u)]
        z=obs[(obs.scope=='2016')&(obs.mass_MeV==r.mass_MeV)&(obs.half_width_u==r.half_width_u)].iloc[0]
        k=int((q.q0>=z.q0).sum());check(k==r.exceedances);check(abs((1+k)/1001-r.p_rank)<1e-15)
        check(abs(beta.ppf(.025,k,1001-k)-r.cp95_low)<1e-15);check(abs(beta.ppf(.975,k+1,1000-k)-r.cp95_high)<1e-15)
    for r in resp.itertuples():
        q=toys[(toys.study=='injection')&(toys.mass_MeV==r.mass_MeV)&(toys.half_width_u==r.half_width_u)]
        a=q[q.injected].set_index('toy');z=q[~q.injected].set_index('toy');delta=a.Ahat-z.Ahat
        check(len(a)==len(z)==200);check(abs(delta.mean()/r.A_expected-r.response)<1e-13)
        check(abs(delta.std(ddof=1)/np.sqrt(200)/r.A_expected-r.response_mc_se)<1e-13)
        check(abs(z.Ahat.std(ddof=1)/r.response-r.response_adjusted_null_sd)<1e-8)
    out=dict(passed=True,checks=checks,parent_numerical_and_source_hashes_unchanged=True,all_inputs_verified=True,
        observed_replay_masses=replay,targeted_toy_fit_replays=replays,all_toy_draws_regenerated=True,
        independent_GP_prediction_matches=True,all_categories_and_masks_verified=True,
        observed_csv_sha256=sha(R/'observed_scan.csv'),toy_rows_sha256=sha(R/'toy_rows.csv'))
    write(B/'qa/window_comparison_validation.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
