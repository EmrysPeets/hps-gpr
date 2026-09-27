"""Independent numerical audit of the saved conditional observed extraction.

Checks pinned inputs, inherited references, probability/geometry bookkeeping,
selected regions and representative likelihood replays. No toys or calibration.
"""
from pathlib import Path
import json
import sys
import traceback
import extraction as E
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve
from scipy.special import ndtr, xlogy

B = Path(__file__).resolve().parents[1]
REPORT = dict(passed=False, scope='Numerical reproducibility and bookkeeping; no coverage or global-significance calibration.', checks=0)


def check(condition, message):
    REPORT['checks'] += 1
    if not bool(condition):
        raise AssertionError(message)


def one(frame, scope, method, mass):
    rows = frame[(frame.scope == scope) & (frame.method == method) & (frame.mass_MeV == mass)]
    check(len(rows) == 1, f'Expected one {scope}/{method}/{mass} row')
    return rows.iloc[0]


def row_agreement(row, ref):
    limit_rel = abs(float(row.psi90 / ref.psi90 - 1.))
    root_abs = abs(float(row.signed_root - ref.signed_root))
    check(limit_rel < 2e-7 and root_abs < 2e-6, f'Reference mismatch: {limit_rel}, {root_abs}')
    return dict(limit_relative_difference=limit_rel, signed_root_absolute_difference=root_abs)


def mask(year, mass, kind):
    d = E.C.DATA[year]
    if kind == 'gaussian':
        center = (mass if year != '2021' else mass-3.2243308692909953-2.213992811446465*np.log(mass/150.))
        width = E.C.sigma(year, mass) * 1000.
        lo, hi = center-2.25*width, center+2.25*width
    else:
        center, width = E.BANKS[year].parameters(mass)
        left, right = (-3.5, 3.5) if year == '2016' else (-4., 3.)
        lo, hi = center+left*width, center+right*width
    return (d['x']*1000 >= lo) & (d['x']*1000 <= hi), lo, hi


def selected_masks(scope, mass, method):
    if scope == '2016':
        return {'2016': mask('2016', mass, 'mc')[0]}
    return {y: mask(y, mass, 'mc' if y != '2015' else 'gaussian')[0] for y in E.years(mass)}


def main():
    frame = pd.read_csv(B/'results/observed_scan.csv', float_precision='round_trip')
    geometry = pd.read_csv(B/'results/template_geometry.csv', dtype={'year':str}, float_precision='round_trip')
    regions = json.loads((B/'results/selected_regions.json').read_text())['regions']
    summary = json.loads((B/'results/summary.json').read_text())
    protocol = json.loads((B/'provenance/protocol.json').read_text())
    hashes = []
    for line in (B/'provenance/input_manifest.sha256').read_text().splitlines():
        digest, relative = line.split(None, 1)
        path = B/relative.lstrip('*')
        check(path.is_file() and E.sha(path) == digest, f'Input hash mismatch: {relative}')
        hashes.append(relative)
    check(len(hashes) == 168, 'Unexpected frozen input count')
    check(protocol['frozen_input_manifest_sha256'] == E.sha(B/'provenance/input_manifest.sha256'), 'Protocol manifest mismatch')
    for name, digest in protocol['scripts'].items():
        check(E.sha(B/'scripts'/name) == digest, f'Frozen script changed: {name}')
    REPORT['pinned_inputs'] = dict(files=len(hashes), protocol_hash=E.sha(B/'provenance/protocol.json'))

    check(len(frame) == 1354 and summary['rows'] == len(frame), 'Unexpected result count')
    check(not frame.duplicated(['scope','method','mass_MeV']).any(), 'Duplicate scan keys')
    check(frame.valid.all(), 'Invalid fit rows')
    check(np.isfinite(frame[['psi90','psi_hat','sigma_psi','signed_root','p0_asymptotic','max_score','min_lambda','cls']].to_numpy()).all(), 'Nonfinite output')
    check((frame.psi90 > 0).all() and (frame.sigma_psi > 0).all(), 'Invalid limits or errors')
    check((frame.min_lambda > 0).all() and (frame.max_score < 3e-5).all(), 'Fit stationarity or positivity failure')
    check(np.max(abs(frame.cls-.1)) < 2e-6, 'CLs root failure')
    check(np.allclose(frame.p0_asymptotic, ndtr(-np.maximum(frame.signed_root,0)), atol=2e-15, rtol=0), 'Local p-value mismatch')
    check(np.allclose(frame.q0, np.maximum(frame.signed_root,0)**2, atol=2e-12, rtol=0), 'q0 mismatch')
    check(np.allclose(frame.epsilon2_90_ee_proxy, frame.psi90*1e-8, atol=0, rtol=2e-15), 'Raw conversion mismatch')
    multiplier = np.array([E.branch(m) for m in frame.mass_MeV])
    check(np.allclose(frame.epsilon2_90_visible_legacy, frame.psi90*1e-8*multiplier, atol=0, rtol=2e-15), 'Display multiplier mismatch')
    REPORT['scan'] = dict(rows=len(frame), fresh_rows=int((frame.origin=='fresh_fit').sum()),
        reused_rows=int((frame.origin!='fresh_fit').sum()), maximum_score=float(frame.max_score.max()),
        minimum_expectation=float(frame.min_lambda.min()), maximum_CLs_residual=float(abs(frame.cls-.1).max()))

    expected_domains = {'2016':(40,175,3), '2015':(60,100,1), '2021':(60,240,2), 'combined':(60,240,3)}
    for scope, (lo,hi,nmethods) in expected_domains.items():
        a = frame[frame.scope==scope]
        check(len(a)==(hi-lo+1)*nmethods, f'Incomplete {scope} domain')
        check(set(a.mass_MeV)==set(range(lo,hi+1)), f'Incorrect {scope} masses')
    for mass, rows in frame[frame.scope=='combined'].groupby('mass_MeV'):
        expected = '+'.join(y for y in ('2015','2016','2021') if y=='2021' or y=='2015' and mass<=100 or y=='2016' and mass<=175)
        check(len(rows)==3 and set(rows.method)==set(E.METHODS), f'Missing combined method at {mass}')
        check(set(rows.campaigns)=={expected}, f'Campaign membership mismatch at {mass}')
        if mass > 175:
            for method, kind in [('all_gaussian','gaussian'),('mc2021','mc'),('mc2016_2021','mc')]:
                row_agreement(one(frame,'combined',method,mass),one(frame,'2021',kind,mass))
    REPORT['membership'] = dict(matched_between_all_three_methods=True, year2015_last_mass=100, year2016_last_mass=175,
        above175_reduces_to_2021=True, excluded_from_old_joint_equality=list(range(176,181)))

    old = pd.read_csv(B/'inputs/v639_combined_scan.csv', float_precision='round_trip')
    matches = []
    for row in frame[frame.scope.isin(['2015','2016','combined'])].itertuples():
        if row.scope=='combined':
            if row.method=='mc2016_2021' or 176<=row.mass_MeV<=180:
                continue
            policy = 'gaussian_baseline' if row.method=='all_gaussian' else 'morph_starter'
        elif row.method=='gaussian' and row.mass_MeV>=60:
            policy = 'unchanged'
        else:
            continue
        refs = old[(old.scope==row.scope)&(old.policy==policy)&(old.mass_MeV==row.mass_MeV)]
        check(len(refs)==1, 'Missing inherited comparison point')
        matches.append(dict(scope=row.scope,method=row.method,mass_MeV=int(row.mass_MeV),**row_agreement(row,refs.iloc[0])))
    REPORT['inherited_reference_agreement'] = dict(points=len(matches),
        combined_points=sum(r['scope']=='combined' for r in matches),
        maximum_limit_relative_difference=max(r['limit_relative_difference'] for r in matches),
        maximum_root_absolute_difference=max(r['signed_root_absolute_difference'] for r in matches))

    old2021 = pd.read_csv(B/'inputs/v638_observed_scan.csv',float_precision='round_trip')
    for row in frame[frame.scope=='2021'].itertuples():
        policy = 'gaussian_baseline' if row.method=='gaussian' else 'morph_starter'
        ref = old2021[(old2021.policy==policy)&(old2021.mass_MeV==row.mass_MeV)].iloc[0]
        factor = E.conversion('2021',row.mass_MeV)
        check(abs(row.psi90*factor/ref.A90-1)<2e-15, 'Reused 2021 limit conversion mismatch')
        check(abs(row.psi_hat*factor-ref.Ahat)<1e-9 and abs(row.sigma_psi*factor-ref.sigma_A)<1e-9, 'Reused 2021 yield conversion mismatch')
        check(row.signed_root==ref.signed_r and row.p0_asymptotic==ref.p0_fixed_mass, 'Reused 2021 test statistic changed')

    max_identity=0.; max_grid_error=0.; max_norm_error=0.; probability_checks=0
    for year, bank in E.BANKS.items():
        e = E.C.DATA[year]['edges']*1000
        for m in range(int(bank.anchors[0]),int(bank.anchors[-1])+1):
            cat = bank.categories(m,e)
            check(np.min(cat)>=0 and abs(cat.sum()-1)<1e-12, f'Invalid {year}/{m} categories')
            max_norm_error=max(max_norm_error,abs(float(cat.sum())-1.))
            pairs=bank.neighbors(m);center=sum(w*bank.samples[a]['center'] for a,w in pairs)
            width=sum(w*bank.samples[a]['width'] for a,w in pairs)
            U=(e-center)/width
            expected=sum(w*np.interp(bank.samples[a]['center']+bank.samples[a]['width']*U,
                bank.samples[a]['edges'],bank.samples[a]['cdf'],left=bank.samples[a]['cdf'][0],
                right=bank.samples[a]['cdf'][-1]) for a,w in pairs)
            ec=np.r_[expected[0],np.diff(expected),1-expected[-1]]
            delta=float(np.max(abs(cat-ec)));max_grid_error=max(max_grid_error,delta)
            check(delta<1e-12, f'Independent interpolation disagreement {year}/{m}')
            if m in bank.anchors:
                direct=bank.categories(m,e,kind='direct');delta=float(np.max(abs(cat-direct)))
                max_identity=max(max_identity,delta)
                check(delta<1e-12, f'Direct anchor disagreement {year}/{m}')
            probability_checks+=1
    grid=np.load(B/'results/template_grid_2016.npz')
    check(np.array_equal(grid['masses_MeV'],np.arange(40,176)), 'Saved template masses mismatch')
    expected=np.array([E.BANKS['2016'].categories(m,E.C.DATA['2016']['edges']*1000) for m in range(40,176)])
    check(np.max(abs(grid['full_MC_categories']-expected))<1e-12,'Saved template categories mismatch')
    REPORT['probabilities']=dict(grid_points=probability_checks, maximum_category_normalization_error=max_norm_error,
        maximum_independent_interpolation_error=max_grid_error, maximum_native_identity_error=max_identity)

    for row in geometry.itertuples():
        fit,lo,hi=mask(row.year,row.mass_MeV,row.kind);d=E.C.DATA[row.year];idx=np.flatnonzero(fit)
        check(fit.sum()==row.fit_bins and (~fit).sum()==row.training_bins,'Geometry bin counts mismatch')
        check(abs(lo-row.requested_low_MeV)<1e-11 and abs(hi-row.requested_high_MeV)<1e-11,'Requested geometry mismatch')
        check(abs(d['edges'][idx[0]]*1000-row.actual_low_MeV)<1e-11 and abs(d['edges'][idx[-1]+1]*1000-row.actual_high_MeV)<1e-11,'Actual geometry mismatch')
        check(row.left_training_bins>=3 and row.right_training_bins>=3,'Insufficient sideband support')
        check(abs(row.fit_fraction+row.training_fraction+row.below_support+row.above_support-1)<1e-12,'Signal probability lost or renormalized')
    for y,m,k in [('2016',40,'mc'),('2016',91,'mc'),('2016',175,'mc'),('2021',60,'mc'),('2021',240,'mc')]:
        ctx=E.Context(y,m,k)
        check(np.array_equal(ctx.fit,ctx.guard),'Fit and GP exclusion differ')
        check(np.array_equal(ctx.fit,mask(y,m,k)[0]),'Runtime mask mismatch')
    REPORT['geometry']=dict(rows=len(geometry), fit_equals_GP_exclusion=True, independent_bin_mask_reconstruction=True)

    selected_verified=[]
    for scope,method in [('2016','mc'),('combined','mc2016_2021')]:
        q=frame[(frame.scope==scope)&(frame.method==method)].sort_values('mass_MeV')
        masses=q.mass_MeV.to_numpy();values=q.q0.to_numpy();candidates=[]
        for i in range(len(q)):
            if values[i]>0 and (i==0 or values[i]>values[i-1]) and (i==len(q)-1 or values[i]>=values[i+1]):
                candidates.append((int(masses[i]),float(values[i])))
        accepted=[];used=[]
        for m,v in sorted(candidates,key=lambda pair:(-pair[1],pair[0])):
            masks=selected_masks(scope,m,method)
            if any(any(y in other and np.any(a & other[y]) for y,a in masks.items()) for other in used):
                continue
            accepted.append(m);used.append(masks)
            if len(accepted)==2:break
        stored=[int(r['mass_MeV']) for r in regions if r['scope']==scope]
        check(accepted==stored and len(stored)==2,f'Selected top two mismatch: {scope}')
        selected_verified.append(dict(scope=scope,masses_MeV=accepted,nonoverlapping_actual_fit_bins=True))
    REPORT['selected_regions']=selected_verified

    targets=set()
    for r in regions:
        for method in E.KINDS2016 if r['scope']=='2016' else E.METHODS:
            targets.add((r['scope'],method,int(r['mass_MeV'])))
    for m in [40,150,175]:targets.add(('2016','mc',m))
    for m in [100,101,175,176,240]:targets.add(('combined','mc2016_2021',m))
    for m in [60,68,80,94,240]:
        for kind in ['gaussian','mc']:targets.add(('2021',kind,m))
    replays=[]
    for scope,method,m in sorted(targets):
        parts=E.parts_for_combined(m,method) if scope=='combined' else [E.Context(scope,m,method).part()]
        for part in parts:
            ctx=part['context']
            check(np.array_equal(ctx.fit,ctx.guard), 'Replay fit/GP mask differs')
            independent_b,independent_cov=E.C.predict(ctx.data['x'],part['counts'],ctx.fit,ctx.const,ctx.ls)
            check(np.allclose(part['b'],independent_b,rtol=1e-12,atol=1e-9),'GP mean differs from inherited engine')
            check(np.allclose(part['cov'],independent_cov,rtol=1e-11,atol=1e-7),'GP covariance differs from inherited engine')
        mod,n=E.model(parts);fit=mod.limit(n,details=True);free=fit['free'];null=fit['null']
        ref=one(frame,scope,method,m)
        check(abs(fit['A90']/ref.psi90-1)<2e-7 and abs(fit['signed_r']-ref.signed_root)<2e-6,'Replay changed result')
        check(abs(free['A']-ref.psi_hat)<2e-7*max(1,abs(ref.psi_hat)),'Replay yield mismatch')
        _,_,H,_=mod._objective(free['z'],n,mod.Jfree,mod.b,mod.penfree)
        unit=np.zeros(len(H));unit[0]=1
        sd=mod.scale*np.sqrt(cho_solve((cholesky(H,lower=True),True),unit)[0])
        check(abs(sd/ref.sigma_psi-1)<2e-8,'Observed-Hessian uncertainty mismatch')
        factor_errors=[]
        for chosen,A in [(free,free['A']),(null,0.),(mod.fit(n,fixed=ref.psi90),ref.psi90)]:
            offset=0;parts_nll=0.
            for part in parts:
                rank=part['L'].shape[1];theta=chosen['theta'][offset:offset+rank];offset+=rank
                lam=part['b']+part['L']@theta+A*part['S']
                check(np.min(lam)>0,'Nonpositive independently reconstructed mean')
                pos=part['n']>0
                parts_nll+=float(np.sum(lam-part['n'])+np.sum(xlogy(part['n'][pos],part['n'][pos]/lam[pos])))+.5*float(theta@theta)
            discrepancy=abs(parts_nll-chosen['nll']);factor_errors.append(discrepancy)
            check(discrepancy<2e-6,'Independent campaign likelihood factorization mismatch')
        replays.append(dict(scope=scope,method=method,mass_MeV=m,
            limit_relative_difference=float(abs(fit['A90']/ref.psi90-1)),
            root_absolute_difference=float(abs(fit['signed_r']-ref.signed_root)),
            Hessian_error_relative_difference=float(abs(sd/ref.sigma_psi-1)),
            likelihood_factorization_max_absolute_error=max(factor_errors)))
        print(f'Validated replay {scope}/{method}/{m}',flush=True)
    REPORT['replays']=replays
    REPORT['files']=dict(observed_scan_sha256=E.sha(B/'results/observed_scan.csv'),
        template_geometry_sha256=E.sha(B/'results/template_geometry.csv'),
        selected_regions_sha256=E.sha(B/'results/selected_regions.json'),script_sha256=E.sha(__file__))
    REPORT['passed']=True


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        REPORT['failure']=f'{type(exc).__name__}: {exc}'
        REPORT['traceback']=traceback.format_exc()
        raise
    finally:
        E.write(B/'qa/extraction_validation.json',REPORT)
        print(json.dumps({k:REPORT[k] for k in ('passed','checks')},indent=2),flush=True)
