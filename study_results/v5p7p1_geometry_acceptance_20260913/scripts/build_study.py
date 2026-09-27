#!/usr/bin/env python3
"""Single-worker, finite-sensor straight-ray acceptance from pinned HPS LCDD."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import json,csv,hashlib,time
import numpy as np
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import uproot

B=Path(__file__).resolve().parents[1]
ME=.51099895
MASS_LIMITS={"2015":250,"2016":550,"2021":950}
META=[('2015',1056.,'#2166ac','invariant_mass',15.,150.,'2015 full'),
      ('2016',2300.,'#8051a6','h_Minv_General_Final_1',30.,300.,'2016 full'),
      ('2021',3740.,'#13876e','preselection/h_invM_8000',36.,1000.,'2021 10%')]

def prepared(g):
    rows=[]
    origin=np.array(g['nominal_vertex_mm'])
    for s in g['sensors']:
        R=np.array(s['rotation_local_to_global']);p=np.array(s['center_mm'])-origin
        n=R[:,s['normal_axis_local']]
        a,b=s['planar_axes_local']
        rows.append((s['station'],s['half'],s['sensor_type'],n,np.dot(n,p),
                     R[:,a],R[:,b],np.dot(R[:,a],p),np.dot(R[:,b],p),
                     s['size_local_mm'][a]/2,s['size_local_mm'][b]/2))
    return rows

def hit_features(dirs,rows):
    """Count individual views and paired stations without double-counting alternatives."""
    groups={}
    for station,half,view,n,d,a,b,pa,pb,ha,hb in rows:
        den=dirs@n
        t=np.divide(d,den,out=np.full(len(dirs),-1.),where=np.abs(den)>1e-14)
        hit=(t>0)&(np.abs(t*(dirs@a)-pa)<=ha+1e-9)&(np.abs(t*(dirs@b)-pb)<=hb+1e-9)
        key=(station,half,view)
        if key in groups:groups[key]|=hit
        else:groups[key]=hit
    return features_from_groups(groups)

def features_from_groups(groups):
    stations=sorted({k[0] for k in groups})
    n=len(next(iter(groups.values())))
    strips=np.zeros(n,dtype=np.int8)
    paired=[]
    for st in stations:
        ok=np.zeros(n,dtype=bool)
        for view in ('axial','stereo'):
            strips+=groups[(st,'t',view)]|groups[(st,'b',view)]
        for half in ('t','b'):
            ok|=groups[(st,half,'axial')]&groups[(st,half,'stereo')]
        paired.append(ok)
    return strips,np.array(paired)

def selection_mask(year,positron,electron,reference=False):
    if reference:
        return positron[1].all(axis=0)&electron[1].all(axis=0)
    if year=='2021':
        return (positron[0]>=10)&(electron[0]>=8)
    ok=(positron[1].sum(axis=0)>=5)&(electron[1].sum(axis=0)>=5)
    if year=='2015':ok&=positron[1][0]&positron[1][1]
    return ok


def daughters(m,E,c,phi,alpha):
    beta=np.sqrt(1-(m/E)**2);gamma=E/m
    q=np.sqrt(m*m/4-ME*ME)
    qt=q*np.sqrt(1-c*c)
    px=qt*np.cos(phi);py=qt*np.sin(phi)
    dirs=[]
    for sign in (1,-1):
        pz=gamma*(beta*m/2+sign*q*c)
        # Beam-frame momentum rotated into the LCDD global frame.
        dirs.append(np.column_stack((sign*px*np.cos(alpha)+pz*np.sin(alpha),
                     sign*py,-sign*px*np.sin(alpha)+pz*np.cos(alpha))))
    return dirs

def evaluate(m,E,g,rows,year,power=13,x=1.):
    uv=qmc.Sobol(2,scramble=False).random_base2(power)
    c=2*uv[:,0]-1;phi=2*np.pi*uv[:,1]
    weights=.75*(1+c*c)
    d1,d2=daughters(m,x*E,c,phi,g['beam_angle_rad'])
    positron=hit_features(d1,rows);electron=hit_features(d2,rows)
    # Require both daughters forward along the parent direction.
    beam=np.array(g['nominal_parent_direction_global'])
    forward=(d1@beam>0)&(d2@beam>0)
    return np.array([np.sum(weights*forward*selection_mask(year,positron,electron,reference))/weights.sum()
                     for reference in (False,True)])

def vertical_edges(g,year,reference=False):
    # PROJECTED aperture only: project each convex sensor face onto beam distance/y.
    # This intentionally discards horizontal hole/slot seams and is not a 3D endpoint.
    th=np.arange(0,.1500001,.000002)
    beam=np.array(g['nominal_parent_direction_global'])
    n=max(s['station'] for s in g['sensors'])
    features=[]
    for sign in (1,-1):
        groups={}
        slope=sign*np.tan(th)
        for s in g['sensors']:
            corners=np.array(s['corners_mm'])-np.array(g['nominal_vertex_mm'])
            q=corners[:,1]/(corners@beam)
            hit=(slope>=q.min())&(slope<=q.max())
            key=(s['station'],s['half'],s['sensor_type'])
            groups[key]=groups.get(key,np.zeros(len(th),dtype=bool))|hit
        features.append(features_from_groups(groups))
    # Accept either assignment of the positron to the upper or lower ray.
    good=(selection_mask(year,features[0],features[1],reference)|
          selection_mask(year,features[1],features[0],reference))
    ix=np.flatnonzero(good)
    assert len(ix)>0,'No projected vertical hit-selection aperture'
    transitions=np.diff(np.r_[False,good,False].astype(int))
    intervals=[(float(th[a]),float(th[b-1])) for a,b in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1))]
    return float(th[ix[0]]),float(th[ix[-1]]),intervals

def symmetric_mass(E,theta):
    return np.sqrt(E*E*np.sin(theta)**2+4*ME*ME*np.cos(theta)**2)

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def draw_geometry(ax,g,E,color,limits,standalone=False):
    lo,hi=limits[:2]
    beam=np.array(g['nominal_parent_direction_global'])
    for s in g['sensors']:
        # Show axial sensor faces in projection; stereo faces are traced numerically.
        if s['sensor_type']!='axial':continue
        corners=np.array(s['corners_mm'])
        zy=np.column_stack((corners@beam,corners[:,1]))
        ax.add_patch(Polygon(zy,closed=True,facecolor=color,edgecolor=color,alpha=.32,lw=1.5))
    z=np.array([0.,945.])
    for sign in (-1,1):
        ax.plot(z,sign*np.tan(.015)*z,color='#727d88',lw=1,ls=':')
        ax.plot(z,sign*np.tan(lo)*z,color=color,lw=1.7)
        ax.plot(z,sign*np.tan(hi)*z,color=color,lw=1.7,ls='--')
    ax.plot(z,z*0,color='#606b76',lw=1)
    ax.plot(0,0,'o',color='#202c38',ms=4)
    ax.set_xlim(-15,960);ax.set_ylim(-95,95)
    ax.set_xticks([0,300,600,900]);ax.set_yticks([-80,-40,0,40,80])
    ax.set_xlabel('Distance along beam [mm]')
    ax.set_ylabel('Vertical position [mm]')
    ax.grid(alpha=.13)
    ax.text(.04,.96,f"{lo*1000:.2f}–{hi*1000:.2f} mrad",transform=ax.transAxes,va='top',
            fontsize=12 if standalone else 10.5,color=color,weight='semibold',bbox=dict(fc='white',ec='none',alpha=.9))
    ax.text(.04,.80,f'15 mrad reference ≈ {E*np.sin(.015):.2f} MeV',transform=ax.transAxes,
            va='top',fontsize=10.4 if standalone else 9.4,color='#53616d',bbox=dict(fc='white',ec='none',alpha=.9))
    ax.text(.5,-.29,
        f"Projected mass scale: {symmetric_mass(E,lo):.2f}–{symmetric_mass(E,hi):.2f} MeV",
        transform=ax.transAxes,ha='center',fontsize=11.6 if standalone else 10.4,color=color,weight='semibold')

def main():
    start=time.perf_counter()
    geo=json.loads((B/'inputs/geometry/active_sensors.json').read_text())
    out=[];summaries=[];plots=[];qa=[];specrows=[];provenance=[]
    masses=np.arange(5.,700.01,2.5)
    for year,E,color,key,crop,endpoint,sample in META:
        g=geo['geometries'][year];rows=prepared(g)
        lim=vertical_edges(g,year); ref_lim=vertical_edges(g,year,reference=True); n=max(s['station'] for s in g['sensors'])
        print(f'{year}: {n} stations, vertical aperture {lim[0]*1e3:.3f}–{lim[1]*1e3:.3f} mrad',flush=True)
        curves=np.array([evaluate(m,E,g,rows,year) for m in masses])
        c80=np.array([evaluate(m,E,g,rows,year,x=.8)[0] for m in masses])
        for m,v,w in zip(masses,curves,c80):
            out.append(dict(year=year,mass_MeV=m,selected_hits_x1=v[0],all_stations_x1=v[1],selected_hits_x0p8=w))
        # Numerical convergence at three physically distinct points; no extra campaign.
        anchors=[.025*E,.055*E,.085*E]
        for m in anchors:
            a=evaluate(m,E,g,rows,year,power=13);b=evaluate(m,E,g,rows,year,power=15)
            qa.append(dict(year=year,mass_MeV=m,standard=a.tolist(),refined=b.tolist(),max_abs_difference=float(np.max(np.abs(a-b)))))
        assert np.all(curves[:,1]<=curves[:,0]+1e-12)
        assert np.all((curves>=0)&(curves<=1))
        peak=np.argmax(curves[:,0])
        summaries.append(dict(year=year,beam_GeV=E/1000,geometry=g['name'],stations=n,sensors=len(g['sensors']),
            vertical_lower_mrad=lim[0]*1e3,vertical_upper_mrad=lim[1]*1e3,
            vertical_mlow_MeV=symmetric_mass(E,lim[0]),vertical_mhigh_MeV=symmetric_mass(E,lim[1]),
            vertical_intervals_rad=lim[2],vertical_scan_step_mrad=.002,
            all_station_reference_intervals_rad=ref_lim[2],
            hit_selection=({"positron_min_2D":10,"electron_min_2D":8,"mandatory_stations":[]} if year=="2021"
                else {"positron_min_3D":5,"electron_min_3D":5,"positron_mandatory_stations":[1,2] if year=="2015" else []}),
            sampled_peak_MeV=float(masses[peak]),sampled_peak_fraction=float(curves[peak,0]),
            positive_scan_support_MeV=[float(masses[curves[:,0]>0].min()),float(masses[curves[:,0]>0].max())]))
        path=B/'inputs'/f'{year}.root'
        with uproot.open(path) as f:raw,re=f[key].to_numpy()
        raw=np.asarray(raw,dtype=float);re=re*1000;cent=(re[1:]+re[:-1])/2
        mask=(re[:-1]>=crop-1e-9)&(re[1:]<=300+1e-9)
        edges=np.arange(301.);counts,_=np.histogram(cent[mask],bins=edges,weights=raw[mask])
        assert np.isclose(counts.sum(),raw[mask].sum(),rtol=0,atol=1e-6)
        for a,b,v in zip(edges[:-1],edges[1:],counts):
            status='displayed' if a>=crop and b<=endpoint else ('display_crop' if a<crop else 'no_source_bin')
            specrows.append(dict(year=year,low_MeV=a,high_MeV=b,selected_pairs=v if status=='displayed' else '',status=status))
        provenance.append(dict(year=year,input=str(path.relative_to(B)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            histogram=key,displayed_pairs=float(counts.sum()),sample=sample,crop_MeV=crop,source_endpoint_MeV=endpoint))
        plots.append((g,lim,curves,c80,counts))

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
        'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    # Exactly the revised top row, also provided independently for note insertion.
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.8))
    for ax,meta,pdat in zip(axes,META,plots):
        y,E,color,*_=meta;g,lim,*_=pdat
        draw_geometry(ax,g,E,color,lim,True)
        ax.set_title(f'HPS {y} | {E/1000:.3f} GeV',color=color,weight='bold',pad=13)
    fig.subplots_adjust(left=.062,right=.987,bottom=.30,top=.83,wspace=.32)
    fig.text(.5,.97,'HPS active-sensor geometry and symmetric decay rays',ha='center',va='top',fontsize=18,weight='semibold')
    fig.text(.5,.045,'2015: ≥5 paired stations each, positron L1+L2.  2016: ≥5 paired stations each.  2021: ≥10 e⁺ / ≥8 e⁻ strip hits.\n'
        'Solid/dashed: projected aperture; grey dotted: nominal 15 mrad. Either charge assignment; horizontal seams omitted; no magnetic transport.',
        ha='center',fontsize=10.7,linespacing=1.4)
    for ext in ('pdf','png','svg'):fig.savefig(B/'figures'/f'HPS_v5p7p1_geometry_illustration.{ext}',dpi=190)
    plt.close(fig)

    fig=plt.figure(figsize=(15.3,12.3))
    fig.suptitle('HPS geometry, straight-track acceptance and selected spectra | v5.7.1',y=.982,fontsize=18.5,weight='semibold')
    lefts=[.06,.383,.706]
    for i,(meta,pdat) in enumerate(zip(META,plots)):
        year,E,color,key,crop,endpoint,sample=meta;g,lim,curves,c80,counts=pdat;n=max(s['station'] for s in g['sensors'])
        ax=fig.add_axes([lefts[i],.713,.273,.18])
        draw_geometry(ax,g,E,color,lim)
        ax.set_title(f'HPS {year} | {E/1000:.3f} GeV',color=color,weight='bold',pad=12)
        ax=fig.add_axes([lefts[i],.395,.273,.205])
        ax.plot(masses,100*curves[:,0],color=color,lw=2,label='Selected hits; x = 1')
        ax.plot(masses,100*curves[:,1],color=color,lw=1.5,ls=':',label=f'All {n} stations; x = 1')
        ax.plot(masses,100*c80,color=color,lw=1.5,ls='--',label='Selected hits; x = 0.8')
        ax.set(xlim=(0,MASS_LIMITS[year]),ylim=(0,55),xlabel='True A′ mass [MeV]',ylabel='Straight-track acceptance [%]')
        ax.legend(fontsize=9,loc='upper right');ax.grid(alpha=.2)
        ax=fig.add_axes([lefts[i],.108,.273,.18])
        edges=np.arange(301.);good=(edges[:-1]>=crop)&(edges[1:]<=endpoint)
        ax.stairs(np.where(good&(counts>0),counts,np.nan),edges,color=color,lw=1.6,baseline=None)
        ax.axvspan(0,crop,color='#d9dfe6',alpha=.5,lw=0)
        if endpoint<300:
            ax.axvspan(endpoint,300,color='#d9dfe6',alpha=.5,lw=0)
            ax.text(.75,.46,'No source bins\nabove 150 MeV',transform=ax.transAxes,ha='center',color='#606b76',fontsize=10)
        ax.set(xlim=(0,300),ylim=(50,3e6),yscale='log',xlabel='Reconstructed pair mass [MeV]',ylabel='Selected pairs / MeV')
        ax.text(.95,.92,sample,transform=ax.transAxes,ha='right',color=color,weight='bold',fontsize=12)
        ax.grid(alpha=.2)
    fig.text(.06,.94,'A | Sensor layout in the vertical projection',fontsize=12.5,weight='semibold')
    fig.text(.06,.632,'B | Run-dependent hit requirements; x = E(A′)/E(beam)',fontsize=12.5,weight='semibold')
    fig.text(.06,.322,'C | Existing selected-pair spectra, without exposure scaling',fontsize=12.5,weight='semibold')
    fig.text(.06,.032,'Hit cuts: 2015 ≥5 paired stations each + e⁺ L1/L2; 2016 ≥5 each; 2021 ≥10 e⁺ / ≥8 e⁻ strip hits.\n'
        'A shows projected mass scales, not 3D endpoints. B retains finite faces and beam rotation; field, material, trigger and reconstruction are omitted.',
        fontsize=10.5,color='#455260',linespacing=1.4)
    for ax in fig.axes:ax.spines[['top','right']].set_visible(False)
    for ext in ('pdf','png'):fig.savefig(B/'figures'/f'HPS_v5p7p1_acceptance_overview.{ext}',dpi=170)
    plt.close(fig)
    # Preserve the specifically requested original 15/70 mrad illustration, vector re-rendered.
    fig,axs=plt.subplots(1,3,figsize=(13.8,4.5))
    for ax,(year,E,color,*_) in zip(axs,META):
        z=np.array([0,1.])
        ax.fill_between(z,15*z,-15*z,color='#d9dfe6',alpha=.6)
        for sign in (-1,1):
            ax.fill_between(z,sign*15*z,sign*70*z,color=color,alpha=.1)
            ax.plot(z,sign*15*z,color=color,lw=2)
            ax.plot(z,sign*70*z,color=color,lw=1.6,ls='--')
        ax.plot(z,0*z,color='#606b76');ax.plot(0,0,'ko',ms=4)
        ax.set(xlim=(-.02,1.03),ylim=(-82,82),yticks=[-70,-15,0,15,70],xticks=[])
        ax.set_title(f'HPS {year} | {E/1000:.3f} GeV',color=color,weight='bold',fontsize=14)
        ax.text(.45,64,'70 mrad (illustrative)',fontsize=10.3,color=color,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))
        ax.text(.62,22,'15 mrad',fontsize=11,color=color)
        ax.text(.5,-.12,f'15 mrad: {E*np.sin(.015):.2f} MeV\n70 mrad: {E*np.sin(.070):.2f} MeV',transform=ax.transAxes,
            ha='center',va='top',color=color,fontsize=12,weight='semibold',linespacing=1.5)
        ax.spines[['top','right','bottom']].set_visible(False)
    axs[0].set_ylabel('Vertical displacement (illustrative)')
    fig.subplots_adjust(left=.07,right=.985,top=.83,bottom=.30,wspace=.22)
    fig.text(.5,.96,'Original v5.7 angular illustration',ha='center',va='top',fontsize=18,weight='semibold')
    fig.text(.5,.035,'Equal-energy daughters; on-axis parent with E(A′) = E(beam); electron mass neglected.\n70 mrad is an assumed outer angle, not a measured detector boundary.',ha='center',fontsize=11)
    for ext in ('pdf','png','svg'):fig.savefig(B/'figures'/f'HPS_v5p7_original_angular_illustration.{ext}',dpi=190)
    plt.close(fig)

    # A two-row companion keeps labels legible when the report is printed.
    from matplotlib.lines import Line2D
    fig=plt.figure(figsize=(12.8,6.7))
    fig.suptitle('Finite-sensor straight-track acceptance and selected spectra',y=.99,fontsize=17,weight='semibold')
    legend=[Line2D([0],[0],color='#485460',lw=2,ls=s) for s in ('-',':','--')]
    fig.legend(legend,['Selected hits, x = 1','All stations, x = 1','Selected hits, x = 0.8'],
               loc='upper center',bbox_to_anchor=(.53,.946),ncol=3,frameon=False,fontsize=11)
    for i,(meta,pdat) in enumerate(zip(META,plots)):
        year,E,color,key,crop,endpoint,sample=meta;g,lim,curves,c80,counts=pdat
        left=[.070,.395,.720][i]
        fig.text(left+.12,.858,f'HPS {year} | {E/1000:.3f} GeV',ha='center',color=color,weight='bold',fontsize=13)
        ax=fig.add_axes([left,.535,.245,.28])
        for vals,ls in [(curves[:,0],'-'),(curves[:,1],':'),(c80,'--')]:ax.plot(masses,100*vals,color=color,lw=1.8,ls=ls)
        ax.set(xlim=(0,MASS_LIMITS[year]),ylim=(0,55),xticks=np.linspace(0,MASS_LIMITS[year],5 if year!='2021' else 3),yticks=[0,20,40],xlabel='True A′ mass [MeV]')
        if i==0:ax.set_ylabel('Ray acceptance [%]')
        ax.grid(alpha=.2)
        ax=fig.add_axes([left,.100,.245,.28])
        edges=np.arange(301.);good=(edges[:-1]>=crop)&(edges[1:]<=endpoint)
        ax.stairs(np.where(good&(counts>0),counts,np.nan),edges,color=color,lw=1.6,baseline=None)
        ax.axvspan(0,crop,color='#d9dfe6',alpha=.5,lw=0)
        if endpoint<300:
            ax.axvspan(endpoint,300,color='#d9dfe6',alpha=.5,lw=0)
            ax.text(.74,.48,'No source bins\nabove 150 MeV',transform=ax.transAxes,ha='center',fontsize=10,color='#606b76')
        ax.set(xlim=(0,300),ylim=(50,3e6),yscale='log',xticks=[0,100,200,300],xlabel='Reconstructed mass [MeV]')
        if i==0:ax.set_ylabel('Selected pairs / MeV')
        ax.text(.95,.9,sample,transform=ax.transAxes,ha='right',color=color,weight='bold',fontsize=11)
        ax.grid(alpha=.2)
    fig.text(.070,.425,'Recorded pair spectra; no exposure scaling',weight='semibold',fontsize=12)
    for ax in fig.axes:ax.spines[['top','right']].set_visible(False)
    for ext in ('pdf','png'):fig.savefig(B/'figures'/f'HPS_v5p7p1_response_spectra.{ext}',dpi=180)
    plt.close(fig)

    write_csv(B/'derived'/'angular_acceptance.csv',out)
    write_csv(B/'derived'/'selected_mass_spectra.csv',specrows)
    (B/'derived'/'geometry_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (B/'derived'/'input_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    maxdiff=max(q['max_abs_difference'] for q in qa)
    assert maxdiff<.006,f'Quadrature convergence worse than 0.6 percentage point: {maxdiff}'
    checks={'status':'passed','orientation_points':8192,'refinement_points':32768,
        'integration':'Unscrambled two-dimensional Sobol quadrature; transverse-vector weights normalized',
        'mass_step_MeV':2.5,'electron_mass_MeV':ME,'refinement_checks':qa,
        'max_absolute_refinement_difference':maxdiff,'count_rebinning':'passed',
        'station_requirement_ordering':'passed','fraction_bounds':'passed','single_worker':True,
        'elapsed_seconds':time.perf_counter()-start,'scope':'finite-sensor, zero-field, prompt straight-ray model'}
    (B/'qa'/'numerical_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    # Tables are generated from the same values used in the plots.
    with (B/'derived'/'geometry_table.tex').open('w') as f:
        f.write('\\begin{tabular}{lrrrrr}\\toprule\nRun & $E_{\\rm beam}$ [GeV] & Stations & $\\theta_{\\rm low}$ [mrad] & $\\theta_{\\rm high}$ [mrad] & $m_{\\rm vert}$ [MeV]\\\\\\midrule\n')
        for s in summaries:f.write(f"{s['year']} & {s['beam_GeV']:.3f} & {s['stations']} & {s['vertical_lower_mrad']:.2f} & {s['vertical_upper_mrad']:.2f} & {s['vertical_mlow_MeV']:.2f}--{s['vertical_mhigh_MeV']:.2f}\\\\\n")
        f.write('\\bottomrule\\end{tabular}\n')
    print(json.dumps({'elapsed_seconds':checks['elapsed_seconds'],'max_refinement_difference':maxdiff,'summaries':summaries},indent=2))

if __name__=='__main__':main()
