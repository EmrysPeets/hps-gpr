#!/usr/bin/env python3
"""Display derivatives for v5.0.1; no new HPS fits, data, or pseudoexperiments.

Run with /usr/bin/python3 (local scientific environment). Frozen numerical inputs
are copied byte-for-byte into provenance/figure_inputs on the initial run. Later
runs use those snapshots. The only GP calculation repeated here is the explicitly
synthetic, deterministic four-panel methodology illustration from the old note.
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, hashlib, shutil, importlib.util, re
os.environ.setdefault('MPLCONFIGDIR', '/tmp/v501-figures-mpl')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

B = Path(__file__).resolve().parents[1]
ROOT = B.parents[1]
ARCH = Path('/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5-analysis-note-20260908')
F, E, P, D = B/'figures', B/'editorial', B/'provenance/figure_inputs', B/'derived'
for directory in (F,E,P,D): directory.mkdir(parents=True, exist_ok=True)
INPUT_MAP_PATH = P/'input_map.json'
INPUT_MAP = json.loads(INPUT_MAP_PATH.read_text()) if INPUT_MAP_PATH.exists() else {}
RECORDS, CHECKS = [], {}
BLUE, ORANGE, GREEN, RED, PURPLE = '#236d9b','#c48124','#358157','#b64038','#80599b'
LABEL = {'2015':'2015 full', '2016':'2016 full','2021':'2021 10%'}
SCOPES = {'2015':'individual_2015_full','2016':'individual_2016_full','2021':'individual_2021_10pct'}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def snapshot(rel):
    if rel in INPUT_MAP:
        p = B/INPUT_MAP[rel]['snapshot']
        assert sha(p) == INPUT_MAP[rel]['sha256'], rel
        return p
    src = next((r/rel for r in (ROOT,ARCH) if (r/rel).exists()),None)
    if src is None: raise FileNotFoundError(rel)
    digest = sha(src)
    target = P/(digest[:12]+'_'+src.name)
    shutil.copy2(src,target)
    INPUT_MAP[rel] = dict(original=str(src),snapshot=str(target.relative_to(B)),sha256=digest)
    INPUT_MAP_PATH.write_text(json.dumps(INPUT_MAP,indent=2)+'\n')
    return target
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.labelsize':10,'legend.fontsize':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42,
        'savefig.dpi':175})
def save(fig,name,sources,caption,derivation):
    for ext in ('pdf','png'): fig.savefig(F/(name+'.'+ext),bbox_inches='tight',pad_inches=.12,dpi=300 if ext=='pdf' else 175)
    plt.close(fig)
    RECORDS.append(dict(name=name,caption=caption,derivation=derivation,
        inputs=[dict(path=str(x.relative_to(B)),sha256=sha(x)) for x in sources],
        outputs=[dict(path=str((F/(name+'.'+e)).relative_to(B)),sha256=sha(F/(name+'.'+e))) for e in ('pdf','png')]))

def correlations():
    rels = ['study_results/v4p9p14_interpretation_global_20260906/global/2015/analysis/covariance.npz',
      'study_results/v4p9p15_global_2016_2021_20260906/global_fast/2016/analysis/covariance.npz',
      'study_results/v4p9p15_global_2016_2021_20260906/global_fast/2021/analysis/covariance.npz',
      'study_results/v4p9p16_combined_global_20260906/global/analysis/covariance.npz']
    sources = [snapshot(r) for r in rels]
    fig, axes = plt.subplots(2,2,figsize=(9.4,8.2),layout='constrained')
    rows = []
    for ax, name, source in zip(axes.flat,['2015 full','2016 full','2021 10%','Combined search'],sources):
        a=np.load(source); m=a['masses_MeV']; k=a['profiled_K']
        assert k.shape == (len(m),len(m)) and np.allclose(k,k.T,atol=1e-12)
        im=ax.imshow(k,origin='lower',extent=[m[0]-.5,m[-1]+.5]*2,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.grid(False); ax.set(title=name,xlabel='Mass hypothesis [MeV]',ylabel='Mass hypothesis [MeV]')
        if name=='Combined search':
            for boundary in (38.5,49.5,90.5,180.5):
                ax.axvline(boundary,color='.35',ls=':',lw=.65)
                ax.axhline(boundary,color='.35',ls=':',lw=.65)
        rows.append(dict(panel=name,masses=len(m),minimum_mass=float(m[0]),maximum_mass=float(m[-1]),
                         matrix_sha256=hashlib.sha256(np.asarray(k,dtype='<f8').tobytes()).hexdigest()))
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.84,pad=.025,label='Correlation of local fit fluctuations',ticks=[-1,-.5,0,.5,1])
    save(fig,'v501_profiled_correlations_2x2',sources,
       'Profiled-response correlation matrices under the archived generating backgrounds: 2015 full (top left), 2016 full (top right), 2021 10% (bottom left), and the combined search (bottom right). All panels share a dimensionless color scale. Positive and negative entries describe same-direction and opposite-direction statistical fluctuations, respectively. Dotted lines in the combined panel mark changes in the active datasets. The matrices quantify dependence; they do not assign a probability to an observed echo or describe correlations of physical signal production.',
       'The exact four saved profiled_K matrices are displayed unchanged on one common scale; this replaces v5.0.0 Figures 57--60.')
    CHECKS['correlation_panels'] = rows

def bounds():
    card=snapshot('study_results/v4p9p12_expanded_snapshot_20260905/inputs/analysis_card.yaml')
    # Pin the implementation used to interpret the frozen card as well.
    implementation=[snapshot('hps_gpr/'+x) for x in ('gpr.py','dataset.py','config.py')]
    import types
    pkg=types.ModuleType('v501_pinned');pkg.__path__=[];sys.modules['v501_pinned']=pkg
    for name,p in zip(('gpr','dataset','config'),implementation):
        sp=importlib.util.spec_from_file_location('v501_pinned.'+name,p)
        mod=importlib.util.module_from_spec(sp);sys.modules[sp.name]=mod;sp.loader.exec_module(mod)
    from v501_pinned.config import load_config
    from v501_pinned.dataset import make_datasets
    from v501_pinned.gpr import compute_kernel_ls_bounds
    cfg=load_config(str(card));datasets=make_datasets(cfg)
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.15),sharey=True)
    for ax,(year,ds),color in zip(axes,datasets.items(),[BLUE,ORANGE,GREEN]):
        masses=np.linspace(ds.m_low,ds.m_high,350)
        vals=[compute_kernel_ls_bounds(ds,cfg,mass=float(m)) for m in masses]
        lo=np.array([v['ls_lo'] for v in vals]);hi=np.array([v['ls_hi'] for v in vals]);sx=np.array([v['sigma_x'] for v in vals])
        ax.fill_between(masses*1000,lo,hi,color=color,alpha=.17)
        ax.plot(masses*1000,lo,color=color,lw=1.5)
        ax.plot(masses*1000,hi,color=color,lw=1.5)
        ax.plot(masses*1000,sx,color='.25',ls=':',lw=1.5)
        ax.set(title=LABEL[year],xlabel='Mass hypothesis [MeV]',xlim=(masses[0]*1000,masses[-1]*1000),ylim=(0,.65))
        kmin=cfg.kernel_ls_res_lower_factor_by_dataset[year];kmax=cfg.kernel_ls_res_upper_factor_by_dataset[year]
        ax.text(.5,1.015,rf'$k_{{\min}}={kmin:g},\quad k_{{\max}}={kmax:g}$',transform=ax.transAxes,ha='center',va='bottom',fontsize=9)
        ax.set_title(LABEL[year],pad=28)
        for m,l,h,s in zip(masses,lo,hi,sx):
            rows.append(dict(dataset=year,mass_MeV=m*1000,sigma_x=s,ell_lower=l,ell_upper=h,
                             k_min=kmin,k_max=kmax,upper_floor_active=bool(h>kmax*s*(1+1e-12))))
    axes[0].set_ylabel(r'Length scale $\ell$ in $x=\log m$')
    fig.legend(handles=[Patch(facecolor=BLUE,alpha=.2,label='Allowed GP length-scale region'),Line2D([],[],color='.25',ls=':',label=r'Detector resolution $\sigma_x(m)$')],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.005))
    fig.tight_layout(rect=(0,.1,1,1))
    out=pd.DataFrame(rows);out.to_csv(D/'v501_lengthscale_regions.csv',index=False,float_format='%.17g')
    CHECKS['lengthscale_bounds']={'rows':len(out),'dataset_stat_upper_floor_active_points':int(out.upper_floor_active.sum()),
         'exact_implementation':'compute_kernel_ls_bounds with frozen analysis card and pinned module snapshots',
         'upper_floor_mode':cfg.kernel_ls_local_hi_floor_mode,'upper_floor_factor':cfg.kernel_ls_local_hi_floor_factor}
    save(fig,'v501_lengthscale_allowed_regions',[card]+implementation,
      'Allowed RBF length scales in the GP input coordinate x=log m, evaluated from the frozen analysis card across each search range. Shaded regions lie between the configured lower and upper bounds; dotted curves show the local detector-resolution scale sigma_x(m). The lower/upper factors are 1.0/8 for 2015, 0.9/12 for 2016, and 1.1/15 for 2021. The implementation also retains its configured upper-bound floor of 0.8 times the dataset-median sigma_x times k_max; it is nonbinding over the plotted search ranges. These are allowed numerical domains, not fitted length scales or detector-resolution uncertainties.',
      'Deterministic evaluation of the exact bound function and resolution parameterizations; no spectra or fits are used.')

def clean_kernel():
    src=snapshot('output/pdf/hps_gpr_analysis_note_v4p1_hyperparameters_20260805/source/make_gpr_hyperparameter_explainer.py')
    text=src.read_text().replace('figsize=(7.35, 6.65)','figsize=(10.5, 8.5)').replace('hspace=0.58','hspace=0.82')
    mod=types_module('kernel_schematic',text,src)
    def intercepted(fig,_output,_name):
        axes=fig.axes[:4]
        removed=[]
        for ax in axes:
            for item in list(ax.texts): removed.append(item.get_text());item.remove()
            leg=ax.get_legend()
            if leg: leg.remove()
            ax.set_title('',loc='center')
            ax.title.set_fontsize(11)
            ax.xaxis.label.set_fontsize(10);ax.yaxis.label.set_fontsize(10)
            ax.tick_params(labelsize=9)
        names=['(a) Interpolation through excluded bins','(b) Joint covariance-parameter selection',
               '(c) Moving the test mass moves the mask','(d) Interior and boundary optima']
        for ax,name in zip(axes,names):ax.set_title(name,loc='left',pad=11)
        # Rasterize only the dense colored mesh to prevent PDF cell-edge seams;
        # contours, star, labels, and all other panels remain vector artwork.
        from matplotlib.collections import QuadMesh
        for artist in axes[1].collections:
            if isinstance(artist,QuadMesh): artist.set_rasterized(True)
        axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=1,frameon=False,fontsize=8)
        axes[1].legend(handles=[Line2D([],[],marker='*',color=ORANGE,ls='',markersize=10,label='Bounded maximum'),
                       Line2D([],[],color=RED,ls='--',label='Configured lower / upper bound')],
                       loc='upper center',bbox_to_anchor=(.5,-.23),frameon=False,fontsize=8)
        axes[2].set_ylim(-.35,1.35)
        axes[2].set_yticks([0,1],labels=['96 MeV','72 MeV']);axes[2].tick_params(axis='y',labelsize=8)
        axes[2].legend(handles=[Line2D([],[],marker='o',color='#263442',ls='',label='Training bin'),
                       Line2D([],[],marker='o',markerfacecolor='white',color=BLUE,ls='',label='Excluded bin')],
                       loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,frameon=False,fontsize=8)
        axes[3].legend(handles=[Patch(color='#E8F0F7',label='Allowed domain'),Line2D([],[],marker='o',color=GREEN,ls='',label='Interior optimum'),
                       Line2D([],[],marker='^',color=RED,ls='',label='Upper-bound contact')],loc='upper center',bbox_to_anchor=(.5,-.23),frameon=False,fontsize=8)
        fig.suptitle('How the GP selects its smoothness at each test mass',fontsize=14,fontweight='semibold',y=.98)
        # The remaining explanatory interpretation is in the caption, outside data areas.
        CHECKS['kernel_schematic']={'source_synthetic_calculation_unchanged':True,'removed_in_plot_text':removed,'HPS_observations_used':False,'likelihood_mesh_rasterized_dpi':300}
        save(fig,'v501_kernel_optimization_clean',[src],
          'Deterministic illustration of per-mass GP hyperparameter selection. (a) One synthetic sideband sample is interpolated using short, interior-optimum, and long RBF length scales; the shaded central interval is excluded from training. (b) The joint log marginal likelihood selects the ConstantKernel value and RBF length scale within the bounded domain. (c) Moving the test mass changes the excluded bins and local resolution. (d) Interior optima and upper-bound contacts have different interpretations: contact means that the numerical domain is active and motivates a controlled range study. The curves and likelihood surface are unchanged from the earlier schematic; annotations have been moved into legends and the caption. No HPS data or measured fit results enter this figure.',
          'Reproduce the original deterministic synthetic illustration, remove all in-panel explanatory text and contour-label clutter, expand layout, and place keys below each panel.')
    mod.save_pair=intercepted
    mod.configure_style();mod.make_mass_hypothesis_optimization(F);style()
def types_module(name,text,path):
    import types
    mod=types.ModuleType(name);mod.__file__=str(path);exec(compile(text,str(path),'exec'),mod.__dict__);return mod

def observed_and_projection():
    curves=snapshot('study_results/v4p9p12_final_dataset_combinations_20260902/derived/final_dataset_result_curves.csv')
    bands=snapshot('study_results/v4p9p12_expanded_snapshot_20260905/derived/expected_band_summary_dimuon_300toys.csv')
    total=snapshot('study_results/v4p9p12_expanded_snapshot_20260905/derived/final_total_search_window_dimuon_300toys.csv')
    dens=snapshot('study_results/v4p9p12_combination_expected_bands_20260904/derived/prediction_state_ledger.csv')
    babar=snapshot('study_results/combined_observed_2015full_2016full_2021_10pct_asymptotic_v3_20260802/babar_comparison/derived/babar_Lees2014xha_eps2_90.csv')
    original=snapshot('study_results/v4p9p7_2016_support_combined_100toy_20260902/note/source/sections/06_results.tex')
    card=snapshot('study_results/v4p9p12_expanded_snapshot_20260905/inputs/analysis_card.yaml')
    df=pd.read_csv(curves,float_precision='round_trip');b=pd.read_csv(bands,float_precision='round_trip')
    checks=[]
    for year,key in SCOPES.items():
        q=df[df.scope_key.eq(key)].sort_values('mass_MeV').copy()
        q=q.merge(b[b.scope_key.eq(key)][['mass_MeV','eps2_observed','dimuon_factor']],on='mass_MeV',validate='one_to_one')
        # The inherited v5 band CSV rounded its one large 2015 endpoint at 90 MeV
        # to 13 decimal digits; retain that displayed value exactly, while the
        # yield panel retains the higher-precision original fit value.
        rounding_delta=float(np.max(np.abs(q.eps2_90*q.dimuon_factor/q.eps2_observed-1)))
        assert rounding_delta < 2e-12
        assert np.array_equal(q.mass_MeV.to_numpy(),np.arange({'2015':19,'2016':39,'2021':50}[year],{'2015':90,'2016':180,'2021':250}[year]+1))
        fig,axes=plt.subplots(3,1,figsize=(8.9,7.4),sharex=True,gridspec_kw={'height_ratios':[1,1,1.1]})
        axes[0].plot(q.mass_MeV,q.A90_full_template_events,color=BLUE,lw=1.65)
        axes[1].plot(q.mass_MeV,q.eps2_observed,color=BLUE,lw=1.65)
        axes[2].plot(q.mass_MeV,q.p0_local_asymptotic,color=BLUE,lw=1.65)
        axes[0].set_ylabel(r'$A_{90}$ [events]');axes[1].set_ylabel(r'90% CL$_s$ limit on $\epsilon^2$')
        axes[2].set_ylabel('Nominal local $p_0$');axes[2].set_xlabel('Mass hypothesis [MeV]')
        axes[2].axhline(.05,color='.5',ls='--',lw=1,label='$p_0=0.05$');axes[2].legend(loc='lower right',frameon=False)
        for ax in axes: ax.set_yscale('log');ax.set_xlim(q.mass_MeV.min(),q.mass_MeV.max())
        pmin=q.loc[q.p0_local_asymptotic.idxmin()]
        axes[2].set_ylim(min(q.p0_local_asymptotic.min()*.5,.0001),1.)
        support={'2015':'14--135','2016':'30--210','2021':'36--300'}[year]
        fig.suptitle(f'{LABEL[year]}: observed yield, coupling limit, and local probability',fontsize=12,y=.98,fontweight='semibold')
        fig.text(.5,.935,f'GP support {support} MeV; minimum local $p_0$ = {pmin.p0_local_asymptotic:.3g} at {pmin.mass_MeV:g} MeV ($Z$ = {pmin.Z_local_asymptotic:.2f})',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,0,1,.9))
        keep=['mass_MeV','A90_full_template_events','A90_fitted_window_events','eps2_90','dimuon_factor','eps2_observed','p0_local_asymptotic','Z_local_asymptotic']
        q[keep].to_csv(D/f'v501_observed_scan_{year}.csv',index=False,float_format='%.17g')
        checks.append(dict(dataset=year,n_points=len(q),A90_full_template_unchanged=True,
           epsilon2_matches_v500=True,p0_unchanged=True,inherited_csv_conversion_max_relative_rounding=rounding_delta,minimum_local_p0=float(pmin.p0_local_asymptotic),minimum_local_p0_mass_MeV=int(pmin.mass_MeV),minimum_local_Z=float(pmin.Z_local_asymptotic)))
        save(fig,f'v501_observed_scan_{year}',[curves,bands],
         f'{LABEL[year]} observed scan with GP support {support} MeV. From top to bottom: the 90% CLs full-template signal-yield upper limit A90, the corresponding minimal-visible epsilon-squared upper limit, and the nominal local asymptotic p0. The same frozen released profile fits supply all panels. The epsilon-squared conversion includes the inherited dimuon branching correction once above threshold; the visible electron-pair yield is unchanged. The dashed line marks p0=0.05. No expected bands are shown, and the minimum p0 has no look-elsewhere correction.'+(' The 2016 numerical and generating-source qualifications remain applicable.' if year=='2016' else ''),
         'The v4.9.7 Figure 85 display is rebuilt separately for each current dataset from exact saved rows; no fits, new toys, or projected p-values.')
    CHECKS['individual_scans']=checks
    t=pd.read_csv(total,float_precision='round_trip');d=pd.read_csv(dens,float_precision='round_trip');bb=pd.read_csv(babar,float_precision='round_trip')
    assert np.array_equal(t.mass_MeV,np.arange(19,251))
    density_map={(str(int(x.dataset)),int(x.mass_MeV)):x.integral_density_events_per_GeV for x in d.itertuples()}
    rows=[]
    for x in t.itertuples():
        ds=x.dataset_set.split('+'); terms={year:density_map[(year,x.mass_MeV)] for year in ds}
        cur=sum(terms.values());future=sum(v*(10 if y=='2021' else 1) for y,v in terms.items())
        scale=np.sqrt(cur/future)
        rows.append(dict(mass_MeV=x.mass_MeV,dataset_set=x.dataset_set,eps2_current=x.eps2_observed,
             density_current_events_per_GeV=cur,density_2021_full_equivalent_events_per_GeV=future,
             density_2015=terms.get('2015',0),density_2016=terms.get('2016',0),density_2021=terms.get('2021',0),
             projection_scale=scale,eps2_observed_equivalent=x.eps2_observed*scale))
    p=pd.DataFrame(rows)
    assert np.array_equal(p.eps2_current.to_numpy(),t.eps2_observed.to_numpy())
    assert np.allclose(p[p.dataset_set.eq('2021')].projection_scale,1/np.sqrt(10),rtol=1e-15)
    assert np.array_equal(p[p.mass_MeV<50].projection_scale.to_numpy(),np.ones(31))
    # Plot the original BaBar points, without smoothing or extrapolation. A separate
    # tabulation uses the original comparison's linear-mass/log-limit convention.
    bb=bb.sort_values('mass_MeV')
    valid=p.mass_MeV.between(bb.mass_MeV.min(),bb.mass_MeV.max())
    p['babar_eps2_90_logy']=np.nan
    p.loc[valid,'babar_eps2_90_logy']=np.exp(np.interp(p.loc[valid,'mass_MeV'],bb.mass_MeV,np.log(bb.eps2_90)))
    p['current_over_babar']=p.eps2_current/p.babar_eps2_90_logy
    p['proxy_over_babar']=p.eps2_observed_equivalent/p.babar_eps2_90_logy
    p.to_csv(D/'v501_current_projected_babar.csv',index=False,float_format='%.17g')
    fig,ax=plt.subplots(figsize=(9.6,5.6))
    use=bb.mass_MeV.between(19,250)
    ax.plot(bb.loc[use,'mass_MeV'],bb.loc[use,'eps2_90'],color=ORANGE,lw=1.8,label='BaBar visible dark photon (2014)')
    ax.plot(p.mass_MeV,p.eps2_current,color='.4',ls='--',lw=1.6,label='Current HPS combined: 2021 at 10%')
    ax.plot(p.mass_MeV,p.eps2_observed_equivalent,color=BLUE,lw=1.9,label='Observed-equivalent HPS proxy: 2021 at 100%')
    ax.axvline(2*105.6583755,color='.6',ls=':',lw=1)
    ax.set(xlim=(19,250),yscale='log',xlabel=r'Dark-photon mass $m_{A\prime}$ [MeV]',ylabel=r'90% upper limit on $\epsilon^2$')
    ax.set_title('Current result and conditional full-2021 density response',loc='left',pad=64,fontsize=12)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,fontsize=9)
    ax.text(2*105.6583755+.9,.95,r'$2m_\mu$',transform=ax.get_xaxis_transform(),ha='left',va='top',color='.5',fontsize=9)
    fig.tight_layout()
    pmin=p.loc[p.eps2_observed_equivalent.idxmin()];cmin=p.loc[p.eps2_current.idxmin()]
    CHECKS['babar_projection']={'points':len(p),'current_curve_exact_v500_match':True,'only_2021_scaled':10,
      'normalization_window_nsigma':1.64,'proxy_formula':'eps2_current*sqrt(sum_d density_d/sum_d scale_d*density_d)',
      'only_2021_scale_exact':float(1/np.sqrt(10)),'no_2021_scale_exact':1.,
      'current_minimum':{'mass_MeV':int(cmin.mass_MeV),'eps2':float(cmin.eps2_current)},
      'proxy_minimum':{'mass_MeV':int(pmin.mass_MeV),'eps2':float(pmin.eps2_observed_equivalent)}}
    save(fig,'v501_current_projected_babar',[total,dens,babar,original,card],
      'Current observed 90% upper limits and the conditional full-2021 density-response proxy compared with the published BaBar visible-dark-photon observed 90% contour. The current HPS curve is exactly the v5.0.0 maximal-available-dataset result over 19--250 MeV. The blue proxy holds the full 2015 and 2016 contributions fixed, multiplies only the 2021 observed normalization density in m plus or minus 1.64 sigma_m by ten, and scales epsilon-squared by sqrt(current summed density / changed summed density). This is density weighted in overlap regions and equals 1/sqrt(10) when only 2021 is active. The curve retains current fluctuations; it is neither expected sensitivity nor a future observed limit or refit. The dimuon correction is inherited once. HPS and BaBar use different statistical constructions. The 2016 qualification remains applicable.',
      'Apply the exact restricted prescription of v4.9.7 Figure 84 to current saved HPS limits and current saved normalization densities. Display native BaBar contour points; no new inference or p-values.')

def kinematic_snippet():
    src=snapshot('study_results/v4p9p7_2016_support_combined_100toy_20260902/note/source/sections/03_event_selection.tex')
    text=src.read_text();start=text.index('% ============================================================\n%  Preselection validation figures')
    end=text.index('\\subsection{2015 mass resolution}',start)
    block=text[start:end]
    # Keep the five archival figure groups and their captions exactly, changing
    # float policy only and adding an explicit prose bridge above them.
    begin=block.index('\\begin{figure}')
    block=block[begin:].replace('\\begin{figure}[H]','\\begin{figure}[p]')
    intro=r'''\subsection{Kinematic distributions after the 2021 preselection}
\label{sec:preselection-kinematics}
The five groups in Figures~\ref{fig:presel_vertex}--\ref{fig:scatter_chi2}
show the reconstructed quantities used to inspect the 2021 loose preselection.
Figure~\ref{fig:presel_vertex} displays the mass spectrum and vertex-fit quality;
Figures~\ref{fig:presel_ele_kin} and~\ref{fig:presel_pos_kin} show the electron
and positron momenta and impact parameters. The two-dimensional displays in
Figures~\ref{fig:scatter_mom} and~\ref{fig:scatter_chi2} make their mass dependence
visible and distinguish broad kinematic structure from the narrow simulated
mass hypotheses. The signal samples are scaled for display, so these figures
compare reconstruction shapes rather than absolute yields or efficiencies.
The vertex-fit $\chi^2$ remains a diagnostic: no additional vertex-fit $\chi^2$
requirement is applied to the regenerated 2021 GPR input. These archived plots
document the preselection context and do not constitute a new event selection
or a validation of the final likelihood.

'''
    # Float barriers prevent this restored series from drifting into resolution.
    (E/'v501_kinematic_integration.tex').write_text(intro+block+'\n\\FloatBarrier\n')
    used=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',block)
    asset_records=[]
    # A packaged rebuild verifies the already bundled source panels from their
    # prior ledger. Absolute archive paths are descriptive metadata only.
    ledger_path=E/'v501_figure_provenance.json'
    saved_assets={}
    if ledger_path.exists():
        previous=json.loads(ledger_path.read_text())
        group=next((r for r in previous['records'] if r['name']=='restored_v497_kinematic_figures_8_to_12'),None)
        if group is not None: saved_assets={r['path']:r for r in group['outputs']}
    for name in used:
        current=B/'source'/name
        relative=str(current.relative_to(B))
        if relative in saved_assets:
            expected=saved_assets[relative]
            assert current.exists(), f'Missing bundled kinematic panel: {relative}'
            assert sha(current)==expected['sha256'], f'Changed bundled kinematic panel: {relative}'
            asset_records.append(dict(expected))
        else:
            old=ARCH/'study_results/v4p9p7_2016_support_combined_100toy_20260902/note/source'/name
            if not current.exists(): shutil.copy2(old,current)
            assert sha(current)==sha(old)
            asset_records.append(dict(path=relative,sha256=sha(current),original=str(old)))
    RECORDS.append(dict(name='restored_v497_kinematic_figures_8_to_12',derivation='Original 20 PDF panels and five main captions retained byte-for-byte; added summary subsection and restored visibility.',inputs=[dict(path=str(src.relative_to(B)),sha256=sha(src))],outputs=asset_records))
    CHECKS['kinematics']={'groups':5,'native_pdf_panels':len(used),'all_assets_equal_archived_v497':True}

if __name__=='__main__':
    style();correlations();bounds();clean_kernel();observed_and_projection();kinematic_snippet()
    (E/'v501_figure_provenance.json').write_text(json.dumps({'version':'5.0.1','scope':'figure derivatives only','new_HPS_fits':False,'new_HPS_data':False,'new_toys':False,'records':RECORDS,'checks':CHECKS},indent=2)+'\n')
    (E/'v501_figure_checks.json').write_text(json.dumps(CHECKS,indent=2)+'\n')
    print(json.dumps({'figures':len(RECORDS),'checks':CHECKS},indent=2))
