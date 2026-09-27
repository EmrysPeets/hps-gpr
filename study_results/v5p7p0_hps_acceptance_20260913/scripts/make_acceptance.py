#!/usr/bin/env python3
"""v5.7: conditional angular acceptance and separately labelled selected data.

Single-process deterministic integration. No production or detector simulation.
Run from any directory; input ROOT snapshots and output paths are study-relative.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import csv
import hashlib
import json
import math
import time
import warnings
import numpy as np
import scipy
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import uproot

BASE = Path(__file__).resolve().parents[1]
YEARS = [
    dict(year='2015', energy_GeV=1.056, color='#2166ac',
         file='2015.root', key='invariant_mass', crop=15., endpoint=150., sample='2015 full'),
    dict(year='2016', energy_GeV=2.300, color='#8051a6',
         file='2016.root', key='h_Minv_General_Final_1', crop=30., endpoint=300., sample='2016 full'),
    dict(year='2021', energy_GeV=3.740, color='#13876e',
         file='2021.root', key='preselection/h_invM_8000', crop=36., endpoint=1000., sample='2021 10%'),
]
INNER, OUTER = .015, .070

def acceptance(mass, energy, x=1., cap=True, eps=2e-9):
    """Fraction of all massless transverse-vector decays passing angular cuts.

    Parent along +z, E_A=x*energy. Fold c=cos(theta*) to [0,1], with
    normalized density 3/4*(1+c*c). Integrate phi analytically. Both daughters
    must be forward, |atan(py/pz)|>=15 mrad; optionally polar theta<=70 mrad.
    """
    r = mass / (x*energy)
    if not 0 < r < 1:
        return 0.
    beta = math.sqrt(1-r*r)
    t = math.tan(INNER)
    # Largest |c| for which even phi=pi/2 clears the vertical dead zone.
    c_inner = (r*r*math.sqrt(1+t*t) - t*t*beta)/(r*r+t*t)
    hi = min(beta, c_inner)
    if cap:
        tmax = math.tan(OUTER)
        c_outer = (tmax*tmax*beta-r*r*math.sqrt(1+tmax*tmax))/(r*r+tmax*tmax)
        hi = min(hi, c_outer)
    if hi <= 0:
        return 0.
    def integrand(c):
        q = t*(beta+c)/(r*math.sqrt(max(1e-30, 1-c*c)))
        phi_fraction = 1 - 2/math.pi*math.asin(min(1., max(0., q)))
        return .75*(1+c*c)*phi_fraction
    return quad(integrand, 0., hi, epsabs=eps, epsrel=eps, limit=80)[0]

def direct_check(mass, energy, x, cap, n=800):
    # Independent midpoint integration of boosted daughter momenta in (c,phi).
    c = (np.arange(n)+.5)/n
    phi = (np.arange(n)+.5)*(math.pi/2)/n
    r = mass/(x*energy)
    beta = math.sqrt(1-r*r)
    pt = mass/2*np.sqrt(1-c*c)
    pzp = x*energy/2*(beta+c)
    pzm = x*energy/2*(beta-c)
    py = pt[:,None]*np.sin(phi)[None,:]
    keep = ((pzm[:,None]>0) &
            (np.arctan2(py,pzp[:,None]) >= INNER) &
            (np.arctan2(py,pzm[:,None]) >= INNER))
    if cap:
        keep &= ((np.arctan2(pt,pzp)<=OUTER) &
                 (np.arctan2(pt,pzm)<=OUTER))[:,None]
    return float(np.mean(keep.mean(axis=1)*.75*(1+c*c)))

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def main():
    start = time.perf_counter()
    for d in ('derived','figures','qa'):
        (BASE/d).mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.titlesize':13,'axes.labelsize':11,
                         'xtick.labelsize':10,'ytick.labelsize':10,
                         'pdf.fonttype':42,'ps.fonttype':42})
    masses = np.arange(1.,300.01,.5)
    edges = np.arange(0.,301.,1.)
    curves, spectra, endpoints, provenance = [], [], [], []
    plots = []
    for cfg in YEARS:
        energy = 1000*cfg['energy_GeV']
        gap = np.array([acceptance(m,energy,cap=False) for m in masses])
        cap = np.array([acceptance(m,energy) for m in masses])
        cap80 = np.array([acceptance(m,energy,x=.8) for m in masses])
        for m,g,a,b in zip(masses,gap,cap,cap80):
            curves.append(dict(year=cfg['year'], mass_MeV=m,
                gap_only_x1_fraction=g, gap_and_polar70_x1_fraction=a,
                gap_and_polar70_x0p8_fraction=b))
        path = BASE/'inputs'/cfg['file']
        with uproot.open(path) as f:
            raw, raw_edges = f[cfg['key']].to_numpy()
        raw = np.asarray(raw, dtype=np.float64)
        raw_edges = raw_edges*1000
        centers = (raw_edges[:-1]+raw_edges[1:])/2
        mask = ((raw_edges[:-1]>=cfg['crop']-1e-8) &
                (raw_edges[1:]<=300+1e-8))
        counts,_ = np.histogram(centers[mask], bins=edges, weights=raw[mask])
        assert np.isclose(counts.sum(),raw[mask].sum(),rtol=0,atol=1e-5)
        for lo,hi,n in zip(edges[:-1],edges[1:],counts):
            available = lo>=cfg['crop'] and hi<=cfg['endpoint']
            spectra.append(dict(year=cfg['year'], low_MeV=lo, high_MeV=hi,
                selected_count=float(n) if available else '',
                status='displayed' if available else
                    ('below_display_crop' if lo<cfg['crop'] else 'outside_source_histogram')))
        endpoints.append(dict(year=cfg['year'], energy_GeV=cfg['energy_GeV'],
            reference_m15_MeV=energy*math.sin(INNER),
            illustrative_m70_MeV=energy*math.sin(OUTER),
            reference_m15_x0p8_MeV=.8*energy*math.sin(INNER),
            illustrative_m70_x0p8_MeV=.8*energy*math.sin(OUTER),
            curve_peak_x1_MeV=float(masses[np.argmax(cap)]),
            curve_peak_x1_fraction=float(cap.max())))
        provenance.append(dict(year=cfg['year'], input=str(path.relative_to(BASE)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), histogram=cfg['key'],
            native_bins=len(raw), native_range_MeV=[float(raw_edges[0]),float(raw_edges[-1])],
            native_total=float(raw.sum()), plotted_total=float(counts.sum()),
            display_crop_MeV=cfg['crop'], upper_display_MeV=300,
            display_bin_width_MeV=1., scaling='none', sample=cfg['sample']))
        plots.append((cfg, gap,cap,cap80,counts))

    fig = plt.figure(figsize=(16.5,13.5),facecolor='white')
    fig.suptitle("HPS A′ angular acceptance and selected mass spectra  |  v5.7",x=.515,y=.982,
                 fontsize=21,weight='semibold')
    fig.text(.062,.945,'A  |  Angular scales: equal-energy daughters, prompt on-axis parent, E(A′) = E(beam)',
             fontsize=12.4,weight='semibold')
    fig.text(.062,.642,'B  |  Fraction of A′ decays passing an idealized angular window',
             fontsize=12.4,weight='semibold')
    fig.text(.062,.349,'C  |  Selected pair counts in the existing data samples',
             fontsize=12.4,weight='semibold')
    lefts = [.062,.386,.710]
    axs=[]
    for i,(cfg,gap,cap,cap80,counts) in enumerate(plots):
        color=cfg['color']; energy=cfg['energy_GeV']*1000
        ax=fig.add_axes([lefts[i],.729,.270,.175]); axs.append(ax)
        ax.set_title(f"HPS {cfg['year']}  |  {cfg['energy_GeV']:.3f} GeV",pad=10,color=color,weight='bold')
        ax.fill_between([0,1],[0,15],[0,-15],color='#d9dfe6',alpha=.75)
        for s in [-1,1]:
            ax.fill_between([0,1],[0,s*15],[0,s*70],color=color,alpha=.09)
            ax.plot([0,1],[0,s*15],color=color,lw=2)
            ax.plot([0,1],[0,s*70],color=color,lw=1.5,ls='--')
        ax.annotate('',xy=(1.04,0),xytext=(0,0),arrowprops={'arrowstyle':'->','color':'#495460','lw':1.2})
        ax.plot(0,0,'o',color='#202c38',ms=5)
        ax.text(.05,6,'target',fontsize=9)
        ax.text(.77,3,'beam',fontsize=9,color='#495460')
        ax.text(.61,29,'15 mrad',color=color,fontsize=10,rotation=7)
        ax.text(.49,61,'70 mrad (illustrative)',color=color,fontsize=9.5,rotation=0)
        ax.set_ylim(-82,82);ax.set_xlim(-.025,1.05)
        ax.set_yticks([-70,-15,0,15,70]);ax.set_xticks([])
        if i==0: ax.set_ylabel('Vertical position / reference distance [mrad]',fontsize=10)
        ax.spines[['top','right','bottom']].set_visible(False)
        fig.text(lefts[i],.680,
            f"15 mrad mass scale: {energy*math.sin(INNER):.2f} MeV\n"
            f"70 mrad model upper edge: {energy*math.sin(OUTER):.2f} MeV",
            fontsize=11.4,weight='semibold',color=color,linespacing=1.5)
        ax=fig.add_axes([lefts[i],.416,.270,.195]);axs.append(ax)
        ax.plot(masses,100*gap,color='#8b959f',lw=1.65,ls=':',label='15 mrad gap only; x = 1')
        ax.plot(masses,100*cap,color=color,lw=2.1,label='+ 70 mrad polar cap; x = 1')
        ax.plot(masses,100*cap80,color=color,lw=1.75,ls='--',label='+ 70 mrad polar cap; x = 0.8')
        ax.axvline(energy*math.sin(INNER),color=color,lw=.7,alpha=.4)
        ax.axvline(energy*math.sin(OUTER),color=color,lw=.7,alpha=.4)
        ax.set_ylim(0,100);ax.set_xlim(0,300);ax.set_xticks(np.arange(0,301,50))
        ax.set_xlabel('True A′ mass [MeV]')
        if i==0:ax.set_ylabel('Angular acceptance [%]')
        ax.grid(alpha=.2)
        ax.legend(loc='upper left',fontsize=8.7,framealpha=.94,borderpad=.45)
        ax.text(.5,-.255,'x = E(A′) / E(beam); transverse-vector decay',
                transform=ax.transAxes,ha='center',fontsize=9.5,color='#56616d')
        ax=fig.add_axes([lefts[i],.145,.270,.175]);axs.append(ax)
        keep = (edges[:-1]>=cfg['crop'])&(edges[1:]<=min(300,cfg['endpoint']))
        shown=np.where(keep&(counts>0),counts,np.nan)
        ax.stairs(shown,edges,color=color,lw=1.6)
        ax.axvspan(0,cfg['crop'],color='#d9dfe6',alpha=.45,lw=0)
        if cfg['endpoint']<300:
            ax.axvspan(cfg['endpoint'],300,color='#d9dfe6',alpha=.45,lw=0)
            ax.text(.75,.48,'No source bins\nabove 150 MeV',transform=ax.transAxes,
                    ha='center',fontsize=10,color='#56616d')
        ax.set_xlim(0,300);ax.set_ylim(50,3e6);ax.set_yscale('log')
        ax.set_xticks(np.arange(0,301,50));ax.grid(alpha=.2,which='major')
        ax.set_xlabel('Reconstructed pair mass [MeV]')
        if i==0:ax.set_ylabel('Selected pairs / MeV')
        ax.text(.96,.94,cfg['sample'],transform=ax.transAxes,ha='right',va='top',
                fontsize=11.5,color=color,weight='semibold')
        ax.text(.96,.84,f"{counts.sum()/1e6:.2f} million displayed pairs",transform=ax.transAxes,
                ha='right',va='top',fontsize=9.8,color='#56616d',
                bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
        ax.text(.5,-.31,f"Display crop: {cfg['crop']:.0f} MeV; no exposure scaling",
                transform=ax.transAxes,ha='center',fontsize=9.5,color='#56616d')
    fig.text(.062,.026,
        'A–B: geometric illustration, not calibrated HPS efficiency. A 15 mrad vertical gap alone gives no upper mass cutoff.\n'
        'The 70 mrad polar cap is a historical-design-inspired assumption; the actual 2021 tracker and trigger were upgraded.\n'
        'C: recorded background-dominated spectra, not A′ production rates. Display crops and histogram endpoints are not acceptance boundaries.',
        fontsize=10.8,color='#3e4954',linespacing=1.45)
    for ax in axs:
        ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
    figure=BASE/'figures'/'HPS_v5p7_Aprime_acceptance'
    fig.savefig(str(figure)+'.pdf',metadata={'Title':'HPS v5.7 conditional angular acceptance and selected spectra',
               'Subject':'Illustrative angular acceptance; not calibrated detector efficiency'})
    fig.savefig(str(figure)+'.png',dpi=170)
    plt.close(fig)
    write_csv(BASE/'derived'/'angular_acceptance.csv',curves)
    write_csv(BASE/'derived'/'selected_mass_spectra.csv',spectra)
    write_csv(BASE/'derived'/'mass_scales.csv',endpoints)
    (BASE/'derived'/'input_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    checks=[]
    for cfg in YEARS:
        e=1000*cfg['energy_GeV']
        for x in [1.,.8]:
            assert acceptance(x*e*math.sin(INNER)*.999,e,x)==0
            assert acceptance(x*e*math.sin(OUTER)*1.001,e,x)==0
        assert acceptance(e*math.sin(OUTER)*1.05,e,cap=False)>0
        m=.04*e
        for outer in [False,True]:
            value=acceptance(m,e,cap=outer)
            direct=direct_check(m,e,1.,outer)
            checks.append(dict(year=cfg['year'],mass_MeV=m,cap=outer,
                analytic=value,direct_midpoint=direct,absolute_difference=abs(value-direct)))
            assert abs(value-direct)<.001
    arr=np.array([[v[k] for k in ['gap_only_x1_fraction','gap_and_polar70_x1_fraction',
                                  'gap_and_polar70_x0p8_fraction']] for v in curves])
    assert np.all((arr>=0)&(arr<=1))
    assert np.all(arr[:,1]<=arr[:,0]+1e-12)
    scaled=[acceptance(.04*1000*c['energy_GeV'],1000*c['energy_GeV']) for c in YEARS]
    assert np.ptp(scaled)<1e-12
    qa=dict(status='passed',checks=checks,
        checks_description=['1 MeV rebin preserves displayed count totals',
          'Fractions bounded in [0,1]; polar cap cannot increase gap-only acceptance',
          'Zero outside analytic inner/outer endpoints for fixed x',
          'Gap-only curve remains nonzero above illustrative upper edge',
          'Independent two-dimensional boosted-momentum midpoint integration agrees within 0.001 absolute',
          'Common model depends on mass / parent energy'],
        single_process=True,threads=1,elapsed_seconds=time.perf_counter()-start,
        versions=dict(numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,uproot=uproot.__version__))
    (BASE/'qa'/'numerical_validation.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(dict(figure=str(figure),elapsed_seconds=qa['elapsed_seconds'],mass_scales=endpoints),indent=2))

if __name__=='__main__':
    main()
