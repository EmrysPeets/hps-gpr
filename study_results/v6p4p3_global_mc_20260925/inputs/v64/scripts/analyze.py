"""2016 selected-MC shape study. No data fits, extra smearing or GP runs."""
from pathlib import Path
import json, hashlib, platform, sys
import numpy as np
import pandas as pd
import scipy, uproot
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.special import ndtr, xlogy

B = Path(__file__).resolve().parents[1]
KEY = 'h_MinvScSm_GeneralLargeBins_Final_1'
COEFF = json.loads((B/'inputs/reference_resolution.json').read_text())['sigma_coeffs_GeV']
NBOOT = 64
SEED = 640925

def write(path, obj):
    (B/path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def savecsv(name, rows):
    pd.DataFrame(rows).to_csv(B/'results'/name, index=False, float_format='%.12g')

def sigma(m):
    return 1000*np.polynomial.polynomial.polyval(m/1000, COEFF)

def locate(m, e, y, span=1.5, pedestal=True):
    """Same bin-integrated local Gaussian+affine definition as saved v6.1."""
    x = (e[1:]+e[:-1])/2
    sig = sigma(m)
    smooth = gaussian_filter1d(y, .2*sig/np.diff(e)[0])
    search = np.flatnonzero(abs(x-m) <= 3*sig)
    peak = search[np.argmax(smooth[search])]
    mode = x[peak]
    keep = abs(x-mode) <= span*sig
    yy = y[keep]
    ul, uh = (e[:-1][keep]-mode)/sig, (e[1:][keep]-mode)/sig
    u = (x[keep]-mode)/sig
    mix = (u-u.min())/(u.max()-u.min())
    scale = max(yy.max(), 1.)
    yn = yy/scale
    def expected(p):
        out = p[0]*(ndtr((uh-p[1])/p[2])-ndtr((ul-p[1])/p[2]))
        if pedestal:
            out = out+p[3]*(1-mix)+p[4]*mix
        return out
    def residual(p):
        lam = np.maximum(expected(p), 1e-15)
        term = lam-yn+xlogy(yn, yn/lam)
        return np.sign(lam-yn)*np.sqrt(np.maximum(2*term, 0))
    start = [max(yn.sum()*.8, 1.), 0., .85]
    low, high = [0, -1, .25], [np.inf, 1, 2.5]
    if pedestal:
        start += [max(yn[0]*.5, .0001), max(yn[-1]*.5, .0001)]
        low += [0, 0]; high += [np.inf, np.inf]
    fit = least_squares(residual, start, bounds=(low, high), max_nfev=400,
                        ftol=1e-11, xtol=1e-11, gtol=1e-10)
    p = fit.x
    active = bool(abs(p[1]) > .999 or p[2] < .251 or p[2] > 2.499)
    edge = bool(peak in (search[0], search[-1]))
    center, width = float(mode+p[1]*sig), float(p[2]*sig)
    out = dict(mass_MeV=m, center_MeV=center, shift_MeV=center-m,
               sigma_core_MeV=width, sigma_ref_MeV=sig, smoothed_mode_MeV=float(mode),
               span=span, pedestal=pedestal, valid=bool(fit.success and not active and not edge),
               fit_success=bool(fit.success), shape_bound_hit=active, search_edge=edge,
               fit_bins=int(keep.sum()), deviance=float(np.sum(residual(p)**2)*scale),
               nominal_ndof=int(keep.sum()-len(p)), fit_low_MeV=float(e[:-1][keep][0]),
               fit_high_MeV=float(e[1:][keep][-1]), gaussian_area=float(p[0]*scale),
               pedestal_left=float(p[3]*scale) if pedestal else 0.,
               pedestal_right=float(p[4]*scale) if pedestal else 0.)
    return out

def cdf(e, y, xx):
    return np.interp(xx, e, np.r_[0., np.cumsum(y)]/y.sum(), left=0., right=1.)

def design(m, kind):
    m = np.atleast_1d(m).astype(float)
    x = (m-100)/100
    return {'constant': np.ones((len(m), 1)), 'proportional': (m/100)[:, None],
            'affine': np.column_stack([x*0+1, x]),
            'quadratic': np.column_stack([x*0+1, x, x*x]),
            'logarithmic': np.column_stack([x*0+1, np.log(m/100)])}[kind]

def main():
    hist, inventory, rows, sensitivity, replicas = {}, [], [], [], []
    for path in sorted((B/'inputs/root').glob('*.root'), key=lambda p:int(p.stem.split('_')[-2])):
        m = int(path.stem.split('_')[-2])
        with uproot.open(path) as f:
            h = f[KEY]
            e, y, flow = h.axis().edges()*1000, h.values().astype(float), h.values(flow=True)[[0,-1]]
            v = h.variances()
            assert len(y)==400 and np.all(y>=0) and np.all(y==np.round(y))
            assert np.all(flow==0) and np.allclose(v, y) and h.member('fEntries')==y.sum()
            assert np.allclose(np.diff(e), .625) and np.all(np.isfinite(y))
            meta = dict(mass_MeV=m, input_file=path.name, histogram_key=KEY,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        entries=int(h.member('fEntries')), sumw=float(y.sum()),
                        sumw2=float(v.sum()), underflow=float(flow[0]), overflow=float(flow[1]),
                        input_edges_unit='GeV', analysis_unit='MeV', bins=len(y),
                        bin_width_MeV=.625, support_MeV=[float(e[0]),float(e[-1])],
                        title=h.title, axis_title=h.axis().member('fTitle'),
                        embedded_fit_count=len(h.member('fFunctions')),
                        unsmeared_present='h_Minv_GeneralLargeBins_Final_1' in f)
        inventory.append(meta)
        hist[m] = e, y
        np.savez_compressed(B/'histograms'/f'm{m:03d}.npz', edges_MeV=e, counts=y,
                            variance=v, probability=y/y.sum(), flow_counts=flow)
        r = locate(m, e, y)
        q = np.interp([.025,.16,.5,.84,.975], np.r_[0,np.cumsum(y)]/y.sum(), e)
        x = (e[:-1]+e[1:])/2
        mean = float(np.dot(x,y)/y.sum())
        r.update(entries=int(y.sum()), mean_MeV=mean, mean_shift_MeV=mean-m,
                 median_MeV=float(q[2]), median_shift_MeV=float(q[2]-m),
                 q025_MeV=float(q[0]), q16_MeV=float(q[1]), q84_MeV=float(q[3]),
                 q975_MeV=float(q[4]), halfwidth68_MeV=float((q[3]-q[1])/2),
                 rms_MeV=float(np.sqrt(np.dot((x-mean)**2,y)/y.sum())),
                 mean_mc_se_MeV=float(np.sqrt(np.dot((x-mean)**2,y)/y.sum()**2)),
                 raw_mode_MeV=float(x[y.argmax()]), primary_domain=bool(m>=40),
                 core_fraction_2=float(np.diff(cdf(e,y,r['center_MeV']+r['sigma_core_MeV']*np.array([-2.,2.])))[0]),
                 left_tail_2=float(cdf(e,y,r['center_MeV']-2*r['sigma_core_MeV'])),
                 right_tail_2=float(1-cdf(e,y,r['center_MeV']+2*r['sigma_core_MeV'])))
        for label, center, wid in [('pole_ref',m,sigma(m)), ('core_ref',r['center_MeV'],sigma(m)),
                                    ('core_fitted',r['center_MeV'],r['sigma_core_MeV'])]:
            bounds = cdf(e,y,center+2.25*wid*np.array([-1,1]))
            r[f'{label}_fraction_2p25'] = float(np.diff(bounds)[0])
            r[f'{label}_left_tail_2p25'] = float(bounds[0])
            r[f'{label}_right_tail_2p25'] = float(1-bounds[1])
        for span, ped in [(1.25,True),(2.,True),(1.5,False),(2.,False)]:
            alt = locate(m,e,y,span,ped)
            alt['center_delta_MeV'] = alt['center_MeV']-r['center_MeV']
            sensitivity.append(alt)
        rng = np.random.default_rng(np.random.SeedSequence([SEED,m]))
        boot = []
        for j in range(NBOOT):
            a = locate(m,e,rng.poisson(y).astype(float))
            a.update(replica=j)
            replicas.append(a)
            if a['valid']:
                boot.append([a['center_MeV'],a['sigma_core_MeV']])
        boot = np.asarray(boot)
        r.update(bootstrap_attempts=NBOOT, bootstrap_valid=len(boot),
                 center_mc_sd_MeV=float(boot[:,0].std(ddof=1)),
                 sigma_mc_sd_MeV=float(boot[:,1].std(ddof=1)))
        r['definition_spread_MeV'] = max(abs(a['center_delta_MeV']) for a in sensitivity if a['mass_MeV']==m and a['valid'])
        rows.append(r)
        print(f"{m:3d} N={int(y.sum()):6d} c-m={r['shift_MeV']:+.4f} sd={r['center_mc_sd_MeV']:.4f} sigma={r['sigma_core_MeV']:.3f} F2={r['core_fraction_2']:.3f} valid={r['valid']} boot={len(boot)}/{NBOOT}",flush=True)
    d = pd.DataFrame(rows).set_index('mass_MeV')
    masses = d.index[d.primary_domain & d.valid].to_numpy()
    assert len(masses)==27
    # Smooth laws summarize a location convention, with equal weight per mass.
    laws, law_predictions = [], []
    for target, values in [('shift_MeV',d.loc[masses,'shift_MeV'].to_numpy()),
                           ('log_sigma_core',np.log(d.loc[masses,'sigma_core_MeV'].to_numpy()))]:
        for kind in ['constant','proportional','affine','quadratic','logarithmic']:
            X = design(masses,kind)
            beta = np.linalg.lstsq(X,values,rcond=None)[0]
            loo=[]
            for i,m in enumerate(masses):
                take=masses!=m
                pred=float((design([m],kind)@np.linalg.lstsq(X[take],values[take],rcond=None)[0])[0])
                loo.append(pred)
                law_predictions.append(dict(target=target,model=kind,mass_MeV=int(m),observed=float(values[i]),fitted=float((X@beta)[i]),heldout=pred))
            err=np.array(loo)-values
            laws.append(dict(target=target,model=kind,coefficients=beta.tolist(),
                             training_rms=float(np.sqrt(np.mean((X@beta-values)**2))),
                             loo_rms=float(np.sqrt(np.mean(err**2))),loo_max_abs=float(abs(err).max())))
    # Full-support CDF averaging conserves probability. Each mass has equal weight.
    # Native bin densities are piecewise uniform; no smoothing of empirical shapes.
    u=np.linspace(-200,200,8001)
    aligned=np.array([cdf(*hist[m], d.loc[m,'center_MeV']+d.loc[m,'sigma_core_MeV']*u) for m in masses])
    assert np.allclose(aligned[:,[0,-1]], [0,1])
    pooled=aligned.mean(axis=0)
    np.savez_compressed(B/'histograms/aligned_shapes.npz',u_edges=u,masses_MeV=masses,
                        cdf=aligned,common_cdf=pooled,probability=np.diff(aligned),common_probability=np.diff(pooled))
    metrics=[]
    uc=np.linspace(-2,2,401)
    for m in masses:
        r=d.loc[m];e,y=hist[m];F=np.r_[0,np.cumsum(y)]/y.sum()
        ev=(e-r.center_MeV)/r.sigma_core_MeV
        other=masses[masses!=m]
        common=np.mean([cdf(*hist[j],d.loc[j,'center_MeV']+d.loc[j,'sigma_core_MeV']*ev) for j in other],axis=0)
        core=cdf(e,y,r.center_MeV+r.sigma_core_MeV*uc);core=(core-core[0])/(core[-1]-core[0])
        C=np.mean([cdf(*hist[j],d.loc[j,'center_MeV']+d.loc[j,'sigma_core_MeV']*uc) for j in other],axis=0)
        C=(C-C[0])/(C[-1]-C[0])
        G=(ndtr(uc)-ndtr(-2))/(ndtr(2)-ndtr(-2))
        out=dict(mass_MeV=int(m),gaussian_full_cdf_distance=float(abs(F-ndtr(ev)).max()),
                 moment_gaussian_full_cdf_distance=float(abs(F-ndtr((e-r.mean_MeV)/r.rms_MeV)).max()),
                 common_loo_full_cdf_distance=float(abs(F-common).max()),
                 gaussian_core_cdf_distance=float(abs(core-G).max()),
                 common_loo_core_cdf_distance=float(abs(core-C).max()))
        lower=masses[masses<m];upper=masses[masses>m]
        if len(lower) and len(upper):
            a,b=lower[-1],upper[0];t=(m-a)/(b-a)
            cc=(1-t)*d.loc[a,'center_MeV']+t*d.loc[b,'center_MeV']
            ww=np.exp((1-t)*np.log(d.loc[a,'sigma_core_MeV'])+t*np.log(d.loc[b,'sigma_core_MeV']))
            uu=(e-cc)/ww
            morph=(1-t)*cdf(*hist[a],d.loc[a,'center_MeV']+d.loc[a,'sigma_core_MeV']*uu)+t*cdf(*hist[b],d.loc[b,'center_MeV']+d.loc[b,'sigma_core_MeV']*uu)
            out.update(morph_lower_MeV=int(a),morph_upper_MeV=int(b),morph_center_error_MeV=float(cc-r.center_MeV),
                       morph_width_ratio=float(ww/r.sigma_core_MeV),morph_full_cdf_distance=float(abs(F-morph).max()))
        metrics.append(out)
    s=pd.DataFrame(metrics)
    savecsv('centers_and_shapes.csv',rows);savecsv('fit_definition_sensitivity.csv',sensitivity)
    savecsv('bootstrap_fits.csv',replicas);savecsv('location_law_predictions.csv',law_predictions)
    savecsv('shape_comparisons.csv',metrics)
    write('results/location_width_laws.json',laws);write('results/input_inventory.json',inventory)
    write('results/summary.json',dict(version='6.4',histograms=len(rows),masses_MeV=d.index.tolist(),
          missing_grid_masses_MeV=[150],primary_masses_MeV=masses.tolist(),total_entries=int(d.entries.sum()),
          primary_shift_range_MeV=[float(d.loc[masses,'shift_MeV'].min()),float(d.loc[masses,'shift_MeV'].max())],
          primary_fraction_2_range=[float(d.loc[masses,'core_fraction_2'].min()),float(d.loc[masses,'core_fraction_2'].max())],
          primary_definition_spread_max_MeV=float(d.loc[masses,'definition_spread_MeV'].max()),
          shape_metric_ranges={k:[float(s[k].min()),float(s[k].max())] for k in s if 'distance' in k},
          bootstrap_attempts=len(replicas),bootstrap_valid=sum(r['valid'] for r in replicas),
          invalid_nominal=[r['mass_MeV'] for r in rows if not r['valid']],
          scope='Selected smeared and scaled 2016 MC shape diagnostics; no data-based calibration or GP extraction'))
    write('provenance/runtime.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,uproot=uproot.__version__,platform=platform.platform()))

if __name__=='__main__':
    main()
