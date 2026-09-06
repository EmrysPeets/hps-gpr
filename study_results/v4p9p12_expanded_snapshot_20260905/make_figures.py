#!/usr/bin/env python3
"""Create corrected limits and observed/auxiliary diagnostic figure data."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BANDS = REPO / "study_results/v4p9p12_combination_expected_bands_20260904"
PARENT = REPO / "study_results/v4p9p12_final_dataset_combinations_20260902"
PILOT = REPO / "study_results/v4p9p12_2021_peak_dip_diagnostic_20toys_20260905"
FIGURES = HERE / "figures"
DERIVED = HERE / "derived"
os.environ["MPLCONFIGDIR"] = str(HERE/".mplcache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import yaml

spec = importlib.util.spec_from_file_location("original_band_figures", BANDS/"make_figures.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.FIGURES = FIGURES
MUON_MASS_MEV = 105.6583745
LIMIT_COLUMNS = ("eps2_observed", "expected_q025", "expected_q16", "expected_median", "expected_q84", "expected_q975", "expected_mean", "expected_std")
BLUE, ORANGE, RED, GREEN, PURPLE = "#377EB8", "#E69F00", "#C93434", "#23966D", "#7B4EA3"
inventory, sources = [], {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, **kwargs):
    sources[str(path.relative_to(REPO))] = sha(path)
    return pd.read_csv(path, **kwargs)


def correction(m):
    m = np.asarray(m, float)
    result = np.ones_like(m)
    active = m > 2*MUON_MASS_MEV
    r = (MUON_MASS_MEV/m[active])**2
    result[active] += np.sqrt(1-4*r)*(1+2*r)
    return result


def corrected(frame):
    result = frame.copy()
    factor = correction(result.mass_MeV)
    result["dimuon_factor"] = factor
    for column in LIMIT_COLUMNS:
        result[column+"_ee_raw"] = result[column]
        result[column] = result[column]*factor
    return result


def save(fig, stem):
    fig.savefig(FIGURES/f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES/f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    inventory.append(stem)


def total_limit(total):
    fig = plt.figure(figsize=(11.4, 7.1))
    grid = fig.add_gridspec(2, 1, height_ratios=(.11, 1), hspace=.045,
                           left=.105, right=.985, bottom=.095, top=.79)
    strip = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=strip)
    for (low, high, scope, label), color in zip(old.TOTAL_WINDOW_SEGMENTS,
            ("#DDEAF3", "#D8E7DD", "#E8E0F0", "#F2E5D5", "#E5E5E5")):
        strip.add_patch(Rectangle((low-.5, 0), high-low+1, 1, facecolor=color, edgecolor="white"))
        strip.text((low+high)/2, .5, label, ha="center", va="center", fontsize=8, fontweight="semibold")
    x = total.mass_MeV.to_numpy(float)
    ax.fill_between(x, total.expected_q025, total.expected_q975, color=old.YELLOW, alpha=.4, lw=.5)
    ax.fill_between(x, total.expected_q16, total.expected_q84, color=old.GREEN, alpha=.58, lw=.5)
    median, = ax.plot(x, total.expected_median, "k--", lw=1.8)
    observed, = ax.plot(x, total.eps2_observed, "k-", lw=2.1)
    assert len(observed.get_xdata()) == len(median.get_xdata()) == 232
    for boundary in (38.5,49.5,90.5,180.5):
        ax.axvline(boundary, color=".5", ls=":", lw=.75, zorder=0)
    ax.axvline(2*MUON_MASS_MEV, color=PURPLE, ls="-.", lw=1.25, zorder=0)
    strip.set(ylim=(0,1), xlim=(18.5,250.5))
    strip.set_ylabel("active\nscope", rotation=0, ha="right", va="center", fontsize=8)
    strip.tick_params(left=False, labelleft=False, bottom=False, top=False, labelbottom=False)
    for spine in strip.spines.values():
        spine.set_visible(False)
    ax.set(yscale="log", xlabel=r"Mass hypothesis $m_{A'}$ (MeV)", ylabel=r"90% CL$_s$ upper limit on $\epsilon^2$")
    fig.suptitle("Final observed limit over the total search window", fontsize=14, fontweight="semibold", y=.99)
    handles = old.handles()+[Line2D([],[],color=PURPLE,ls="-.",label=r"Dimuon threshold $2m_\mu$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.94), ncol=3, fontsize=10)
    save(fig, "final_total_search_window_expected_bands_300toys")


def fit_legend():
    return [Line2D([],[],color="black",marker="o",ls="none",ms=4,label="Observed (Poisson errors)"),
            Line2D([],[],color=BLUE,lw=1.8,label="Frozen GP background"),
            Line2D([],[],color=ORANGE,lw=1.8,ls="--",label="Profiled background"),
            Line2D([],[],color=RED,lw=2,label="Signal + background fit"),
            Patch(facecolor=".7",alpha=.15,label=r"Fitted / excluded $\pm2.25\sigma_m$ window")]


def fit_pair(ax, residual, row, points, wide=False):
    data = points[(points.fit_id == row.fit_id) & (points.dataset == str(row.dataset))].sort_values("bin_center_MeV")
    lo, hi = (60,86) if wide else (row.mass_MeV-3.5*row.sigma_MeV, row.mass_MeV+3.5*row.sigma_MeV)
    data = data[data.bin_center_MeV.between(lo,hi)]
    fit = data[data.in_fit]
    x, width = data.bin_center_MeV.to_numpy(float), data.bin_width_MeV.to_numpy(float)
    xf, wf = fit.bin_center_MeV.to_numpy(float), fit.bin_width_MeV.to_numpy(float)
    for panel in (ax,residual):
        panel.axvspan(row.fit_low_MeV,row.fit_high_MeV,color=".7",alpha=.15,zorder=0)
        panel.set_xlim(lo,hi)
        panel.set_xlabel("Invariant mass (MeV)")
    ax.errorbar(x,data.observed/width,yerr=np.sqrt(np.maximum(data.observed,1))/width,
                fmt="o",ms=2.8,color="black",elinewidth=.55,capsize=0,zorder=4)
    ax.plot(x,data.gp_mean/width,color=BLUE,lw=1.5,zorder=2)
    ax.plot(xf,fit.total/wf,color=RED,lw=2,zorder=3)
    ax.plot(xf,fit.profiled_background/wf,color=ORANGE,ls="--",lw=1.7,zorder=5)
    ax.set_ylabel("Events / MeV")
    residual.axhline(0,color=".4",lw=.8)
    residual.errorbar(x,(data.observed-data.profiled_background)/width,
                     yerr=np.sqrt(np.maximum(data.observed,1))/width,fmt="o",ms=2.8,
                     color="black",elinewidth=.55,zorder=4)
    residual.plot(xf,fit.signal/wf,color=RED,lw=2,zorder=3)
    residual.plot(x,(data.gp_mean-data.profiled_background)/width,color=BLUE,lw=1.2,zorder=2)
    residual.set_ylabel("(Data - profiled B) / MeV")
    name = "2021 10%" if str(row.dataset) == "2021" else f"{row.dataset} full"
    suffix = " (search endpoint)" if row.search_endpoint else ""
    ax.set_title(f"{name}: {int(row.mass_MeV)} MeV{suffix}",loc="left",fontweight="semibold",fontsize=11.4)
    residual.set_title("Background-subtracted view",loc="left",fontsize=11.4)
    for panel in (ax,residual):
        panel.ticklabel_format(axis="y",style="sci",scilimits=(-3,4),useMathText=True)
        panel.yaxis.get_offset_text().set_fontsize(9)


def fit_grid(rows, points, title, stem):
    fig, axes = plt.subplots(3,2,figsize=(10.8,11.4))
    for (ax,residual), row in zip(axes,rows.itertuples()):
        fit_pair(ax,residual,row,points)
    fig.suptitle(title,fontsize=14,fontweight="semibold",y=.985)
    fig.legend(handles=fit_legend(),loc="upper center",bbox_to_anchor=(.5,.945),ncol=2,fontsize=9.8)
    fig.subplots_adjust(left=.115,right=.985,bottom=.065,top=.825,hspace=.49,wspace=.32)
    save(fig,stem)


def combined_summed(fits, points):
    """Display-only, overlap-weighted sums; never a replacement likelihood."""
    fig, axes = plt.subplots(2,2,figsize=(11.4,7.1),gridspec_kw={"height_ratios":[1.8,1]})
    output=[]
    for column,mass in enumerate((66,79)):
        rows=fits[(fits.group=="combined")&(fits.mass_MeV==mass)]
        # Limit the display to the intersection of the three fitted windows.
        coverage=[]
        for row in rows.itertuples():
            selected=points[(points.fit_id==row.fit_id)&(points.dataset==str(row.dataset))&points.in_fit]
            coverage.append((float((selected.bin_center_MeV-selected.bin_width_MeV/2).min()),
                             float((selected.bin_center_MeV+selected.bin_width_MeV/2).max())))
        lo,hi=max(pair[0] for pair in coverage),min(pair[1] for pair in coverage)
        edges=np.r_[lo,np.arange(np.ceil(lo),np.floor(hi)+1),hi]
        edges=np.unique(edges)
        centers=(edges[:-1]+edges[1:])/2; widths=np.diff(edges)
        totals={name:np.zeros(len(widths)) for name in ("observed","gp_mean","profiled_background","signal","total")}
        covariance=np.zeros((len(widths),len(widths)))
        for row in rows.itertuples():
            data=points[(points.fit_id==row.fit_id)&(points.dataset==str(row.dataset))].sort_values("bin_center_MeV")
            source_lo=data.bin_center_MeV.to_numpy(float)-data.bin_width_MeV.to_numpy(float)/2
            source_hi=data.bin_center_MeV.to_numpy(float)+data.bin_width_MeV.to_numpy(float)/2
            overlap=np.maximum(0,np.minimum(edges[1:,None],source_hi)-np.maximum(edges[:-1,None],source_lo))
            weights=overlap/(source_hi-source_lo)
            assert np.max(weights[:,~data.in_fit.to_numpy(bool)]) < 1e-10
            for name in totals: totals[name]+=weights@data[name].to_numpy(float)
            covariance+=(weights*data.observed.to_numpy(float))@weights.T
        np.savez_compressed(DERIVED/f"combined_display_poisson_covariance_m{mass:03d}.npz",
                            edges_MeV=edges,covariance_counts=covariance)
        ax,residual=axes[:,column]
        err=np.sqrt(np.diag(covariance))/widths
        ax.errorbar(centers,totals["observed"]/widths,yerr=err,fmt="o",color="black",ms=3.8,elinewidth=.8,zorder=4)
        ax.plot(centers,totals["gp_mean"]/widths,color=BLUE,lw=1.4)
        ax.plot(centers,totals["total"]/widths,color=RED,lw=1.9)
        ax.plot(centers,totals["profiled_background"]/widths,color=ORANGE,ls="--",lw=1.6)
        residual.errorbar(centers,(totals["observed"]-totals["profiled_background"])/widths,
                          yerr=err,fmt="o",color="black",ms=3.8,elinewidth=.8,zorder=4)
        residual.plot(centers,totals["signal"]/widths,color=RED,lw=1.9)
        residual.axhline(0,color=".4",lw=.7)
        ax.set_title(f"{mass} MeV combined peak",loc="left",fontweight="semibold")
        ax.set_ylabel("Summed events / MeV")
        residual.set_ylabel("(Data - profiled B) / MeV")
        residual.set_xlabel("Invariant mass (MeV)")
        for panel in (ax,residual):
            panel.set_xlim(lo,hi)
            panel.ticklabel_format(axis="y",style="sci",scilimits=(-3,4),useMathText=True)
        for i,x in enumerate(centers):
            output.append({"mass_hypothesis_MeV":mass,"bin_center_MeV":x,"bin_low_MeV":edges[i],
                "bin_high_MeV":edges[i+1],"poisson_variance":covariance[i,i],
                **{name:values[i] for name,values in totals.items()}})
    fig.suptitle("All-three signal extraction: summed display of the two leading peaks",fontsize=13,fontweight="semibold",y=.99)
    fig.legend(handles=fit_legend()[:4],loc="upper center",bbox_to_anchor=(.5,.94),ncol=2,fontsize=9.8)
    fig.subplots_adjust(left=.09,right=.985,top=.79,bottom=.085,hspace=.20,wspace=.29)
    pd.DataFrame(output).to_csv(DERIVED/"combined_summed_display.csv",index=False)
    save(fig,"combined_two_peak_summed_extraction")


def deficit_plot(summary, points, width_scan):
    row = next(summary[summary.group == "2021_deficit"].itertuples())
    fig = plt.figure(figsize=(11.3,8.2))
    grid = fig.add_gridspec(2,2,height_ratios=(1.45,1),left=.09,right=.985,
                           bottom=.085,top=.78,hspace=.50,wspace=.29)
    ax, residual = fig.add_subplot(grid[0,0]), fig.add_subplot(grid[0,1])
    fit_pair(ax,residual,row,points,wide=True)
    scan = fig.add_subplot(grid[1,:])
    nominal = width_scan[(width_scan.width_scale == 1) & width_scan.mass_MeV.between(60,88)]
    scan.plot(nominal.mass_MeV,nominal.signed_r,"o-",color=BLUE,ms=3,lw=1.5)
    scan.axhline(0,color=".4",lw=.8)
    scan.axvspan(row.fit_low_MeV,row.fit_high_MeV,color=".7",alpha=.15)
    scan.plot([71],[row.signed_r],"o",color=RED,ms=6)
    scan.set(xlim=(60,88),xlabel="Tested mass hypothesis (MeV)",ylabel=r"Signed local $r$")
    fig.suptitle("2021 deficit at 71 MeV: signed-template diagnostic",fontsize=14,fontweight="semibold",y=.985)
    handles = fit_legend()
    handles[3].set_label("B + signed template (deficit)")
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.94),ncol=2,fontsize=9.5)
    save(fig,"deficit_2021_m071")


def resolutions(config, width_scan):
    colors = {"2015":BLUE,"2016":ORANGE,"2021":GREEN}
    fig,ax = plt.subplots(figsize=(9.7,5.3))
    rows=[]
    for key in colors:
        lo,hi=np.array(config[f"range_{key}"])*1000
        masses=np.arange(round(lo),round(hi)+1)
        sigma=1000*np.polynomial.polynomial.polyval(masses/1000,config[f"sigma_coeffs_{key}"])
        ax.plot(masses,sigma,color=colors[key],lw=2,label=f"{key}"+(" 10%" if key=="2021" else " full"))
        rows.extend({"dataset":key,"mass_MeV":m,"sigma_MeV":s} for m,s in zip(masses,sigma))
    ax.set(xlabel="Mass hypothesis (MeV)",ylabel=r"Nominal mass resolution $\sigma_m$ (MeV)")
    fig.suptitle("Frozen signal-template mass resolutions",fontsize=14,fontweight="semibold",y=.98)
    fig.legend(loc="upper center",bbox_to_anchor=(.5,.91),ncol=3,fontsize=10)
    fig.subplots_adjust(left=.10,right=.985,top=.77,bottom=.12)
    save(fig,"nominal_mass_resolutions")
    pd.DataFrame(rows).to_csv(DERIVED/"nominal_mass_resolutions.csv",index=False)

    palette=plt.get_cmap("viridis")(np.linspace(.05,.95,5))
    fig,(ax,ratio)=plt.subplots(2,1,figsize=(10.4,6.9),sharex=True,gridspec_kw={"height_ratios":[2.4,1]})
    for color,scale in zip(palette,(.8,.9,1.,1.1,1.2)):
        data=width_scan[width_scan.width_scale==scale].sort_values("mass_MeV")
        ax.plot(data.mass_MeV,data.eps2_90*correction(data.mass_MeV),color="black" if scale==1 else color,
                lw=2.0 if scale==1 else 1.2,label=rf"{scale:.1f}$\,\sigma_m$")
        ratio.plot(data.mass_MeV,data.limit_ratio_to_nominal,color="black" if scale==1 else color,lw=1.2)
    ax.set(yscale="log",ylabel=r"90% CL$_s$ limit on $\epsilon^2$")
    ratio.set(xlabel="Mass hypothesis (MeV)",ylabel="Ratio to nominal",xlim=(50,250))
    ratio.axhline(1,color=".5",lw=.5)
    for panel in (ax,ratio): panel.axvline(2*MUON_MASS_MEV,color=PURPLE,ls=":",lw=.8)
    fig.suptitle("2021 signal-width variation: observed limits",fontsize=14,fontweight="semibold",y=.985)
    fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.94),ncol=5,fontsize=10)
    fig.subplots_adjust(left=.11,right=.985,top=.83,bottom=.085,hspace=.10)
    save(fig,"resolution_width_limits")
    fig,axes=plt.subplots(1,3,figsize=(12.1,4.5))
    for ax,(lo,hi),title in zip(axes,[(60,85),(50,250),(50,250)],
                               ["Peak-dip region","Full excess scan","Full deficit scan"]):
        for color,scale in zip(palette,(.8,.9,1.,1.1,1.2)):
            data=width_scan[(width_scan.width_scale==scale)&width_scan.mass_MeV.between(lo,hi)]
            y=data.signed_r if lo==60 else (np.maximum(data.signed_r,0) if title=="Full excess scan" else np.minimum(data.signed_r,0))
            ax.plot(data.mass_MeV,y,color="black" if scale==1 else color,lw=1.5 if scale==1 else 1,label=rf"{scale:.1f}$\,\sigma_m$")
        ax.axhline(0,color=".4",lw=.6)
        ax.set(xlabel="Mass hypothesis (MeV)",ylabel=r"Signed local $r$",xlim=(lo,hi))
        ax.set_title(title,loc="left",fontsize=11.5)
    fig.suptitle("2021 signal-width variation: signed local response",fontsize=14,fontweight="semibold",y=.99)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.93),ncol=5,fontsize=10)
    fig.subplots_adjust(left=.065,right=.985,top=.74,bottom=.15,wspace=.32)
    save(fig,"resolution_width_signed")


def injection_figures():
    toys=read(PILOT/"derived/twenty_toy_scans.csv")
    deterministic=read(PILOT/"derived/deterministic_mean_scans.csv")
    observed=read(PILOT/"derived/observed_scan.csv")
    fig,(ax,delta)=plt.subplots(1,2,figsize=(11.5,5.2))
    for lane,color,label in [("background",BLUE,"Background-only toy median"),("injected",ORANGE,"Injected-signal toy median")]:
        data=toys[toys.lane==lane].groupby("mass_MeV").signed_r
        quant=data.quantile([.16,.5,.84]).unstack()
        ax.fill_between(quant.index,quant[.16],quant[.84],color=color,alpha=.16)
        ax.plot(quant.index,quant[.5],color=color,lw=1.8,label=label)
    ax.plot(observed.mass_MeV,observed.signed_r,"k--",lw=1.5,label="Observed 2021")
    paired=toys.pivot(index=["pair","mass_MeV"],columns="lane",values="signed_r")
    paired["delta_r"]=paired["injected"]-paired["background"]
    quant=paired.delta_r.groupby("mass_MeV").quantile([.16,.5,.84]).unstack()
    delta.fill_between(quant.index,quant[.16],quant[.84],color=GREEN,alpha=.2)
    delta.plot(quant.index,quant[.5],color=GREEN,lw=1.8,label="Median paired injection change")
    det=deterministic.pivot(index="mass_MeV",columns="lane",values="signed_r")
    delta.plot(det.index,det["injected"]-det["background"],color=PURPLE,ls="--",lw=1.5,label="Deterministic injection change")
    for panel in (ax,delta):
        panel.axhline(0,color=".4",lw=.7)
        panel.set(xlim=(60,80),xlabel="Tested mass hypothesis (MeV)")
    ax.set_ylabel(r"Signed local $r$")
    delta.set_ylabel(r"Injection-induced $\Delta r$")
    ax.set_title("Absolute response",loc="left")
    delta.set_title("Paired difference on the same background",loc="left",fontsize=10.5)
    fig.suptitle("Positive 66 MeV injection and the adjacent deficit",fontsize=14,fontweight="semibold",y=.99)
    h,l=ax.get_legend_handles_labels();h2,l2=delta.get_legend_handles_labels()
    fig.legend(h+h2,l+l2,loc="upper center",bbox_to_anchor=(.5,.93),ncol=2,fontsize=9.5)
    fig.subplots_adjust(left=.075,right=.985,top=.69,bottom=.13,wspace=.27)
    save(fig,"injection_pilot_20spectra")
    scans=read(PILOT/"reverse_injection/derived/deterministic_scans.csv")
    changes=read(PILOT/"reverse_injection/derived/injection_induced_delta_r.csv")
    obs=read(PILOT/"reverse_injection/derived/observed_reconstruction.csv")
    fig,(ax,delta)=plt.subplots(1,2,figsize=(11.5,5.2))
    for lane,color,label in [("background",".5","Background only"),("inject_66",BLUE,"Inject at 66 MeV"),
                             ("inject_78",ORANGE,"Inject at 78 MeV"),("inject_80",GREEN,"Inject at 80 MeV")]:
        data=scans[scans.lane==lane]
        ax.plot(data.mass_MeV,data.signed_r,color=color,lw=1.6,label=label)
        if lane!="background": delta.plot(changes.mass_MeV,changes[lane],color=color,lw=1.6)
    ax.plot(obs.mass_MeV,obs.signed_r,"k--",lw=1.7,label="Observed 2021")
    for panel in (ax,delta):
        panel.axhline(0,color=".4",lw=.7)
        panel.set(xlim=(60,88),xlabel="Tested mass hypothesis (MeV)")
    ax.set_ylabel(r"Signed local $r$");delta.set_ylabel(r"Injection-induced $\Delta r$")
    ax.set_title("Absolute deterministic scans",loc="left",fontsize=11)
    delta.set_title("Change relative to the same background",loc="left",fontsize=10.5)
    fig.suptitle("Reverse-injection check on one common generating background",fontsize=13,fontweight="semibold",y=.99)
    fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.92),ncol=3,fontsize=9.5)
    fig.subplots_adjust(left=.075,right=.985,top=.74,bottom=.13,wspace=.27)
    save(fig,"reverse_injection_deterministic")


def main():
    FIGURES.mkdir(parents=True,exist_ok=True)
    DERIVED.mkdir(parents=True,exist_ok=True)
    raw=read(BANDS/"derived/expected_band_summary_300toys.csv")
    summary=corrected(raw)
    summary.to_csv(DERIVED/"expected_band_summary_dimuon_300toys.csv",index=False)
    diagnostics=read(BANDS/"derived/pvalue_diagnostics_300toys.csv")
    total=old.build_total_window(summary,diagnostics)
    total.to_csv(DERIVED/"final_total_search_window_dimuon_300toys.csv",index=False)
    old.style()
    old.single_scope(summary,"all_2015_2016_2021",300)
    old.panel_grid(summary,old.INDIVIDUAL,target_toys=300,stem="individual_expected_band_panels",
                   title="Individual final-sample expected bands",shape=(3,1))
    old.panel_grid(summary,old.COMBINATIONS,target_toys=300,stem="combination_expected_band_panels",
                   title=r"Shared-$\epsilon^2$ combination expected bands",shape=(2,2))
    old.pvalue_grid(diagnostics,300)
    old.pvalue_grid(diagnostics,300,old.INDIVIDUAL)
    inventory.extend(["all_three_expected_bands_300toys","individual_expected_band_panels_300toys",
        "combination_expected_band_panels_300toys","combination_pvalue_panels_300toys","individual_pvalue_panels_300toys"])
    total_limit(total)
    fits=read(DERIVED/"selected_fit_summary.csv",dtype={"dataset":str})
    points=read(DERIVED/"selected_fit_plot_data.csv",dtype={"dataset":str})
    for key in ("2015","2016","2021"):
        rows=fits[fits.group==key].sort_values("rank")
        fit_grid(rows,points,f"{key}"+(" 10%" if key=="2021" else " full")+": three ranked excess regions",f"fit_{key}_top3")
    for rank in (1,2):
        rows=fits[(fits.group=="combined")&(fits["rank"]==rank)].sort_values("dataset")
        mass=int(rows.mass_MeV.iloc[0])
        fit_grid(rows,points,rf"Shared-$\epsilon^2$ extraction at {mass} MeV",f"combined_peak_m{mass:03d}")
    combined_summed(fits,points)
    scan=read(PILOT/"resolution_width_scan/derived/width_scan_all_points.csv")
    deficit_plot(fits,points,scan)
    card=PARENT/"inputs/analysis_card.yaml"
    sources[str(card.relative_to(REPO))]=sha(card)
    cfg=yaml.safe_load(card.read_text())
    resolutions(cfg,scan)
    injection_figures()
    inputs=HERE/"inputs";inputs.mkdir(exist_ok=True)
    for name in ("analysis_card.yaml","reviewed_gp_states.csv","2016_PROVISIONAL_STATE_NUMERICAL_EXCEPTION.json"):
        source=PARENT/"inputs"/name
        sources[str(source.relative_to(REPO))]=sha(source)
        shutil.copy2(source,inputs/name)
    parameters={"analysis_card_sha256":sha(card),"config":cfg,"bin_width_MeV":
        {str(k):float(np.median(g.bin_width_MeV)) for k,g in points.groupby("dataset")},
        "new_toys":0,"expected_band_toys_per_mass":300}
    (DERIVED/"final_parameters.json").write_text(json.dumps(parameters,indent=2)+"\n")
    sources[str((BANDS/"make_figures.py").relative_to(REPO))]=sha(BANDS/"make_figures.py")
    (DERIVED/"figure_manifest.json").write_text(json.dumps({"figures":inventory,"sources":sources,
        "dimuon_threshold_MeV":2*MUON_MASS_MEV,"dimuon_factor_at_250MeV":float(correction([250])[0]),
        "total_observed_and_median_connected":True,"new_toys":0,"band_toys_per_mass":300,
        "plot_explanations_in_captions":True},indent=2)+"\n")
    print(json.dumps({"figures":len(inventory),"dimuon_factor_at_250MeV":float(correction([250])[0])}),flush=True)


if __name__=="__main__":
    main()
