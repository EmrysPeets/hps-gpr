#!/usr/bin/env python3
"""Paper-sized conventional-fit displays from saved baseline arrays; no fits."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
C=dict(data='#222222',total='#176A9B',background='#B37816',null='#777777',signal='#AE3E30')
GROUP_SIZE={('2015',51):3,('2015',21):1,('2016',90):5,('2016',117):5,
            ('2021',78):1,('2021',65):1}
SUM_FIELDS=('counts','background_free','background_null','total_free','total_null','signal_bin_probability','signal_counts')


def display_groups(a,group_size,amplitude):
    """Sum adjacent original bins, retaining the last partial group."""
    starts=np.arange(0,len(a['counts']),group_size)
    stops=np.minimum(starts+group_size,len(a['counts']))
    grouped={'edges_MeV':np.r_[a['edges_MeV'][starts],a['edges_MeV'][-1]],
             'native_start':a['native_indices'][starts],
             'native_stop_exclusive':a['native_indices'][stops-1]+1,
             'native_bins_per_group':stops-starts}
    fields={};passed=True
    for name in SUM_FIELDS:
        values=amplitude*a['signal_bin_probability'] if name=='signal_counts' else a[name]
        grouped[name]=np.array([np.sum(values[i:j]) for i,j in zip(starts,stops)])
        independent=np.array([math.fsum(values[i:j]) for i,j in zip(starts,stops)])
        error=np.abs(grouped[name]-independent)
        native_total=math.fsum(values);display_total=math.fsum(grouped[name])
        tolerance=8*np.finfo(float).eps*np.maximum(1.,np.abs(independent))
        field_passed=bool(np.all(error<=tolerance) and
                          abs(display_total-native_total)<=8*np.finfo(float).eps*max(1.,abs(native_total)))
        if name=='counts':
            field_passed=field_passed and np.array_equal(grouped[name],independent) and display_total==native_total
        passed=passed and field_passed
        fields[name]={'passed':bool(field_passed),'native_total':native_total,'display_total':display_total,
                      'maximum_group_sum_error':float(error.max()),
                      'total_conservation_error':float(abs(display_total-native_total))}
    grouped['width_MeV']=np.diff(grouped['edges_MeV'])
    grouped['count_error']=np.sqrt(grouped['counts'])
    for name in SUM_FIELDS:
        grouped[name+'_per_MeV']=grouped[name]/grouped['width_MeV']
    grouped['count_error_per_MeV']=grouped['count_error']/grouped['width_MeV']
    complete=np.array_equal(np.concatenate([np.arange(i,j) for i,j in zip(starts,stops)]),np.arange(len(a['counts'])))
    count_error_passed=np.allclose(grouped['count_error']**2,grouped['counts'],rtol=4*np.finfo(float).eps,atol=0)
    density_passed=all(np.allclose(grouped[name+'_per_MeV']*grouped['width_MeV'],grouped[name],rtol=4*np.finfo(float).eps,atol=0) for name in SUM_FIELDS)
    passed=passed and complete and bool(count_error_passed) and bool(density_passed)
    return grouped,{'passed':bool(passed),'native_bins':len(a['counts']),'display_bins':len(starts),
                    'nominal_native_bins_per_group':group_size,'last_native_bins':int(stops[-1]-starts[-1]),
                    'nominal_width_MeV':float(group_size*np.diff(a['edges_MeV'])[0]),
                    'last_width_MeV':float(grouped['width_MeV'][-1]),
                    'complete_nonoverlapping_partition':bool(complete),'count_error_identity':bool(count_error_passed),
                    'density_conversion_identity':bool(density_passed),'sum_fields':fields}


def main():
    if not json.loads((HERE/'derived/summary.json').read_text())['passed']:
        raise RuntimeError('All fixed variants must pass numerical checks before display')
    rows=pd.read_csv(HERE/'derived/fit_summary.csv',dtype={'dataset':str},float_precision='round_trip')
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,
                         'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    paths=[];group_qa=[];group_arrays={};group_rows=[];source_hashes={}
    for year,masses in [('2015',[51,21]),('2016',[90,117]),('2021',[78,65])]:
        fig,axes=plt.subplots(2,2,figsize=(7.1,5.5),sharex='col',
                              gridspec_kw={'height_ratios':[1.05,1.]})
        fig.subplots_adjust(left=.115,right=.985,bottom=.16,top=.79,hspace=.49,wspace=.30)
        for col,mass in enumerate(masses):
            row=rows[(rows.dataset==year)&(rows.mass_MeV==mass)&(rows.variant=='baseline')].iloc[0]
            source=HERE/'derived/points'/f'{row.fit_id}__baseline.npz';a=np.load(source)
            source_hashes[str(source.relative_to(HERE))]=hashlib.sha256(source.read_bytes()).hexdigest()
            group_size=GROUP_SIZE[(year,mass)];g,qa=display_groups(a,group_size,row.amplitude_full)
            g['amplitude_full']=np.asarray(row.amplitude_full)
            if not qa['passed']:raise RuntimeError('Display grouping identity failure: '+row.fit_id)
            qa.update(fit_id=row.fit_id,dataset=year,mass_MeV=mass);group_qa.append(qa)
            group_arrays.update({row.fit_id+'__'+key:value for key,value in g.items()})
            for i,width in enumerate(g['width_MeV']):
                group_rows.append(dict(fit_id=row.fit_id,dataset=year,mass_MeV=mass,display_bin=i,
                    native_start=int(g['native_start'][i]),native_stop_exclusive=int(g['native_stop_exclusive'][i]),
                    native_bins=int(g['native_bins_per_group'][i]),low_MeV=float(g['edges_MeV'][i]),
                    high_MeV=float(g['edges_MeV'][i+1]),width_MeV=float(width),count=int(g['counts'][i])))
            x=.5*(g['edges_MeV'][:-1]+g['edges_MeV'][1:]);n=g['counts_per_MeV'];err=g['count_error_per_MeV']
            top,res=axes[:,col]
            top.errorbar(x,n,err,fmt='.',ms=2.2,lw=.55,color=C['data'],zorder=4)
            for key,color,style in [('total_free','total','-'),('background_free','background',':'),('background_null','null','--')]:
                top.stairs(g[key+'_per_MeV'],g['edges_MeV'],baseline=None,lw=1.05,ls=style,color=C[color],zorder=3)
            count_display=np.r_[n-err,n+err,g['total_free_per_MeV'],g['background_free_per_MeV'],g['background_null_per_MeV']]
            count_low,count_high=count_display.min(),count_display.max();count_span=count_high-count_low
            top.set_ylim(max(0.,count_low-.05*count_span),count_high+.05*count_span)
            width_label=f'Display {qa["nominal_width_MeV"]:g} MeV'
            if qa['last_native_bins']<group_size:width_label+=f'; last {qa["last_width_MeV"]:g} MeV'
            top.set_title(f'{mass} MeV | degree {int(row.degree)}, total {row.total_width_sigma:g}σ\n'
                          f'r = {row.root:+.2f}; D/dof = {row.deviance:.1f}/{int(row.ndof)}\n'
                          +width_label,fontsize=8.7,pad=7)
            top.set_ylabel('Events / MeV')
            residual=n-g['background_null_per_MeV'];total=g['total_free_per_MeV']-g['background_null_per_MeV'];signal=g['signal_counts_per_MeV']
            res.errorbar(x,residual,err,fmt='.',ms=2.6,lw=.7,color=C['data'],zorder=4)
            res.stairs(total,g['edges_MeV'],baseline=None,lw=1.25,color=C['total'],zorder=3)
            res.stairs(signal,g['edges_MeV'],baseline=None,lw=1.2,ls='--',color=C['signal'],zorder=3)
            displayed=np.r_[residual-err,residual+err,total,signal]
            lower,upper=displayed.min(),displayed.max();span=upper-lower
            res.set_ylim(lower-.07*span,upper+.07*span)
            res.axhline(0,color='.65',lw=.65,zorder=0)
            res.set(xlabel='Invariant mass [MeV]',ylabel='(Data - null) / MeV')
            for ax in (top,res):
                ax.axvline(mass,ls=':',color='.7',lw=.7,zorder=0)
                ax.set_xlim(a['edges_MeV'][0],a['edges_MeV'][-1])
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
                ax.grid(axis='y',alpha=.16)
        scope='10%' if year=='2021' else 'full data'
        fig.suptitle(f'{year} {scope}: conventional fits at GP-selected masses',fontsize=11.5,y=.995)
        count_handles=[Line2D([],[],marker='.',ls='',color=C['data'],label='Observed bins'),
                       Line2D([],[],color=C['total'],label='Fitted total'),
                       Line2D([],[],color=C['background'],ls=':',label='Profiled background'),
                       Line2D([],[],color=C['null'],ls='--',label='Null background')]
        fig.legend(handles=count_handles,loc='upper center',bbox_to_anchor=(.53,.955),ncol=4,
                   fontsize=8.3,frameon=False,handlelength=1.55,columnspacing=1.05)
        residual_handles=[Line2D([],[],color=C['total'],label='Fitted total - null background'),
                          Line2D([],[],color=C['signal'],ls='--',label='Gaussian signal component')]
        fig.legend(handles=residual_handles,loc='center',bbox_to_anchor=(.55,.475),ncol=2,
                   fontsize=8.4,frameon=False,handlelength=2.,columnspacing=2.)
        fig.text(.115,.035,'Fits use native bins; display groups sum whole bins. Counting errors only.',fontsize=8.3)
        for suffix in ('pdf','png'):
            path=HERE/'figures'/f'traditional_{year}_display.{suffix}'
            fig.savefig(path,dpi=200,bbox_inches='tight');paths.append(path)
        plt.close(fig)
    qa_folder=HERE/'qa'
    np.savez_compressed(qa_folder/'paper_display_groups.npz',**group_arrays)
    pd.DataFrame(group_rows).to_csv(qa_folder/'paper_display_groups.csv',index=False)
    qa_paths=[qa_folder/'paper_display_groups.npz',qa_folder/'paper_display_groups.csv']
    (qa_folder/'display_group_identity.json').write_text(json.dumps({'passed':all(q['passed'] for q in group_qa),
        'fits':group_qa,'source_sha256':source_hashes,
        'grouped_output_sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in qa_paths},
        'new_fits':0,'scope':'Disjoint whole-native-bin sums with every final partial group retained. Counts conserve exactly; prediction sums conserve to floating-point roundoff. Each sum is checked using independent compensated summation before division by its actual width.'},indent=2)+'\n')
    (HERE/'paper_figure_inventory.json').write_text(json.dumps({'figures':[str(p.relative_to(HERE)) for p in paths],
        'sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'fit_inputs_sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'derived/fit_summary.csv',HERE/'derived/summary.json']},
        'display_group_identity_sha256':hashlib.sha256((qa_folder/'display_group_identity.json').read_bytes()).hexdigest(),
        'new_fits':0,'scope':'Paper-sized baseline count and residual densities after declared whole-bin display grouping. All model variants are retained in the separate table and unchanged original supplementary figures.'},indent=2)+'\n')


if __name__=='__main__':main()
