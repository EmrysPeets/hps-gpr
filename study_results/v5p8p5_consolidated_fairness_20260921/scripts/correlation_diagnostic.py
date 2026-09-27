"""Bounded audit of frozen v5.8.2 fields; no fits and no adopted calibration.

Reuses the saved 200000 Gaussian maxima and 256 complete Poisson scans.
Generates 500000 extra 2016-only Gaussian fields to measure within-block
maximum tails before attempting an independent-block Sidak approximation.
The source, blind window and observed scan are never modified.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-v585-statistics-mpl")
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
from scipy.special import owens_t
from scipy.stats import beta, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
PARENT = BASE / "inputs/v5p8p2_nominal_gp_significance_20260917"
if not PARENT.exists():
    PARENT = BASE.parent / "v5p8p2_nominal_gp_significance_20260917"
SEED = 58520260921
N_BLOCK = 500000
START = time.monotonic()

def guard():
    if (BASE / "STOP").exists():
        raise SystemExit("STOP file present")
    if time.monotonic() - START > 240:
        raise SystemExit("Local four-minute runtime cap reached")

def ci(k, n):
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(.975, k+1, n-k)))

def tail(a, u):
    k = int(np.count_nonzero(a >= u)); n = len(a)
    lo, hi = ci(k, n)
    return dict(k=k, N=n, p=k/n, p_addone=(k+1)/(n+1), lo95=lo, hi95=hi)

def sidak(n, p):
    return -np.expm1(n * np.log1p(-np.asarray(p)))

def write_csv(name, rows):
    with (BASE / "results" / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def representatives(K, order, all_pairs):
    selected = []
    for j in order:
        compare = selected if all_pairs else selected[-1:]
        if not compare or np.all(np.abs(K[j, compare]) < .1):
            selected.append(j)
    return np.array(sorted(selected), dtype=int)

def crossing_bound(K, u):
    # Exact finite-grid Gaussian expectation: P(X_i<=u, X_{i+1}>u)
    # = 2 T(u, sqrt((1-rho_i)/(1+rho_i))).  No continuum smoothness assumption.
    rho = np.clip(np.diag(K, 1), -1+1e-15, 1-1e-15)
    n_up = float(np.sum(2*owens_t(u, np.sqrt((1-rho)/(1+rho)))))
    return n_up, min(1.0, float(norm.sf(u)+n_up))

def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(BASE / "figures" / (name+"."+ext), bbox_inches="tight", dpi=180)
    plt.close(fig)

guard()
for d in ("results", "figures"):
    (BASE/d).mkdir(exist_ok=True)
scopes = ("2015", "2016", "2021", "combined")
raw_rows = []; region_rows = []; curves = []; hashes = {}
fields = {}
for scope in scopes:
    guard()
    path = PARENT / "fields" / (scope+".npz")
    hashes[str(path.relative_to(BASE.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    f = dict(np.load(path)); fields[scope] = f
    m, a, s, r, K = (f[k] for k in ("masses", "a", "s", "observed_r", "K"))
    z = (r-a)/s; jr = int(np.argmax(r)); jz = int(np.argmax(np.where(r>0, z, -np.inf)))
    for name, j, threshold, gs, ps, localp in (
        ("raw_maximum", jr, r[jr], f["gaussian_raw_maximum"],
         np.maximum(0, f["validation"].max(axis=1)), norm.sf(z[jr])),
        ("reference_maximum", jz, z[jz], f["gaussian_maximum"],
         f["direct_maximum"], norm.sf(z[jz]))):
        row = dict(scope=scope, test=name, peak_mass_MeV=float(m[j]),
                   threshold=float(threshold), raw_r=float(r[j]), a=float(a[j]),
                   response_s=float(s[j]), reference_z=float(z[j]),
                   conditional_local_p=float(localp), raw_standard_normal_p=float(norm.sf(r[j])))
        row.update({"Gaussian_"+k:v for k,v in tail(gs, threshold).items()})
        row.update({"Poisson_"+k:v for k,v in tail(ps, threshold).items()})
        row.update({"Poisson_local_"+k:v for k,v in tail(f["validation"][:, j], r[j]).items()})
        raw_rows.append(row)
    near = representatives(K, range(len(m)), False)
    pair = representatives(K, range(len(m)), True)
    reverse = representatives(K, reversed(range(len(m))), True)
    Kr = K[np.ix_(near, near)]
    pairK = K[np.ix_(pair, pair)]
    u = float(z[jz]); p = float(norm.sf(u)); n_up, bound = crossing_bound(K,u)
    pg = tail(f["gaussian_maximum"], u)
    row = dict(scope=scope, threshold=u, local_p=p, grid_nodes=len(m),
               greedy_nearest_rho_count=len(near),
               greedy_nearest_max_pair_abs_rho=float(np.max(np.abs(Kr-np.eye(len(near))))),
               greedy_all_pair_rho_count=len(pair),
               greedy_reverse_all_pair_rho_count=len(reverse),
               greedy_all_pair_max_abs_rho=float(np.max(np.abs(pairK-np.eye(len(pair))))),
               minimum_adjacent_rho=float(np.min(np.diag(K,1))),
               nearest_point_Sidak_p=float(sidak(len(near),p)),
               all_pair_point_Sidak_p=float(sidak(len(pair),p)),
               expected_upcrossings=n_up, exact_grid_upcrossing_bound_p=bound,
               Gaussian_full_p=pg["p"], Gaussian_full_lo95=pg["lo95"], Gaussian_full_hi95=pg["hi95"])
    region_rows.append(row)
    (BASE/"results"/("stats_regions_"+scope+".json")).write_text(json.dumps(dict(
        scope=scope, nearest_selected_masses=m[near].tolist(),
        all_pair_selected_masses=m[pair].tolist(), reverse_all_pair_selected_masses=m[reverse].tolist()),indent=2)+"\n")
    assert np.max(-a/s) < 1.5, "Threshold curves must be above every positive-fit gate"
    for u in np.linspace(1.5, 4.5, 61):
        p=float(norm.sf(u)); n_up,bound=crossing_bound(K,u); t=tail(f["gaussian_maximum"],u)
        curves.append(dict(scope=scope, threshold=float(u), local_p=p, Gaussian_p=t["p"],
                           Gaussian_lo95=t["lo95"], Gaussian_hi95=t["hi95"],
                           exact_grid_upcrossing_bound_p=bound, expected_upcrossings=n_up,
                           nearest_point_Sidak_p=float(sidak(len(near),p)),
                           all_pair_point_Sidak_p=float(sidak(len(pair),p)),
                           tail_equivalent_N=float(np.log1p(-t["p"])/np.log1p(-p))))

# One intentionally explicit contiguous partition: midpoint/Voronoi blocks around
# nearest-selected representatives. There is no claim of a unique partition.
f=fields["2016"]; K=f["K"]; m=f["masses"]
near=representatives(K,range(len(m)),False)
edges=(m[near][:-1]+m[near][1:])/2
block=np.searchsorted(edges,m,side="right")
inds=[np.flatnonzero(block==j) for j in range(len(near))]
eig,U=np.linalg.eigh((K+K.T)/2); fac=U*np.sqrt(np.maximum(eig,0))
rng=np.random.default_rng(SEED)
blockmax=np.empty((N_BLOCK,len(inds)))
for start in range(0,N_BLOCK,2048):
    guard(); stop=min(start+2048,N_BLOCK)
    W=rng.standard_normal((stop-start,len(m)))@fac.T
    for j,ind in enumerate(inds):
        blockmax[start:stop,j]=W[:,ind].max(axis=1)
maximum=blockmax.max(axis=1)
u_obs=float(region_rows[1]["threshold"])
block_rows=[]
for u in sorted(set(np.linspace(1.5,4.5,61).tolist()+[u_obs])):
    pb=np.mean(blockmax>=u,axis=0)
    # The same draws estimate all marginal block probabilities; independence is
    # then an approximation to be checked against the actual joint maximum.
    product=float(-np.expm1(np.log1p(-pb).sum()))
    joint_k=int(np.sum(maximum>=u)); joint_lo,joint_hi=ci(joint_k,N_BLOCK)
    row=dict(threshold=u, blocks=len(inds), Gaussian_joint_k=joint_k,
             Gaussian_joint_N=N_BLOCK, Gaussian_joint_p=float(np.mean(maximum>=u)),
             Gaussian_joint_lo95=joint_lo,Gaussian_joint_hi95=joint_hi,
             independent_block_maxima_p=product,
             point_Sidak_p=float(sidak(len(inds),norm.sf(u))))
    block_rows.append(row)
    if u==u_obs:
        peak_block_row=row.copy()
        block_detail=[dict(block=j,mass_min=float(m[ix].min()),mass_max=float(m[ix].max()),
                           grid_nodes=len(ix),representative_mass=float(m[near[j]]),
                           marginal_exceedance_count=int(np.sum(blockmax[:,j]>=u)),
                           marginal_exceedance_p=float(pb[j])) for j,ix in enumerate(inds)]

write_csv("stats_raw_and_reference.csv",raw_rows)
write_csv("stats_correlation_regions.csv",region_rows)
write_csv("stats_same_field_tails.csv",curves)
write_csv("stats_2016_block_tails.csv",block_rows)
write_csv("stats_2016_block_details.csv",block_detail)
np.savez_compressed(BASE/"results/stats_2016_block_maxima.npz",block_maxima_first1024=blockmax[:1024],
                    representative_indices=near,block_index=block,seed=SEED)
summary=dict(source="Frozen v5.8.2 fixed nominal GP background",blind_sigma=2.25,
             new_fits=0,new_Gaussian_2016_fields=N_BLOCK,seed=SEED,
             source_hashes=hashes,block_peak=peak_block_row,regions=region_rows,
             raw_and_reference=raw_rows,
             elapsed_seconds=time.monotonic()-START,
             conclusions={
                 "rho_cutoff_is_independence":False,
                 "representatives_preserve_full_scan_maxima":False,
                 "source_uncertainty_propagated":False,
                 "production_calibration_adopted":False,
                 "finite_grid_bound_formula":"sf(u)+sum_i 2*owens_t(u,sqrt((1-rho_i)/(1+rho_i))); clipped at 1",
                 "gate_note":"All plotted thresholds exceed every positive-amplitude gate in every scope",
                 "block_note":"2016 contiguous midpoint blocks from left-to-right nearest rho<0.1 representatives; both partition and approximation are diagnostic"})
(BASE/"results/stats_summary.json").write_text(json.dumps(summary,indent=2)+"\n")

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "pdf.fonttype":42,"axes.grid":True,"grid.alpha":.15})
d=[r for r in curves if r["scope"]=="2016"];x=np.array([r["threshold"] for r in d])
get=lambda k: np.array([r[k] for r in d])
fig,ax=plt.subplots(1,2,figsize=(10.6,4.2))
ax[0].fill_between(x,get("Gaussian_lo95"),get("Gaussian_hi95"),color="#23648d",alpha=.22)
ax[0].semilogy(x,get("Gaussian_p"),color="#23648d",lw=2,label="Full correlated grid: P(max Z > u)")
ax[0].semilogy(x,get("exact_grid_upcrossing_bound_p"),color="#32845e",ls="--",label="Same-grid upcrossing upper bound")
ax[0].semilogy(x,get("local_p"),color=".35",ls=":",label="One fixed mass")
ax[0].axvline(u_obs,color=".5",lw=.7,ls=":")
ax[0].set(title="Same field: GP maxima versus upcrossings",xlabel="Reference threshold u",ylabel="Exceedance probability",ylim=(1e-5,1.1))
ax[0].legend(fontsize=8,loc="lower left")
db=block_rows;xb=np.array([r["threshold"] for r in db]);gb=lambda k: np.array([r[k] for r in db])
ax[1].semilogy(xb,gb("Gaussian_joint_p"),color="#23648d",lw=2,label="Full grid maximum (500,000 new draws)")
ax[1].fill_between(xb,gb("Gaussian_joint_lo95"),gb("Gaussian_joint_hi95"),color="#23648d",alpha=.2)
ax[1].semilogy(xb,gb("independent_block_maxima_p"),color="#7656a5",ls="--",label="24 block maxima, independence assumed")
ax[1].semilogy(xb,gb("point_Sidak_p"),color="#bb793d",ls="-.",label="24 representative points: Sidak")
ax[1].semilogy(x,get("all_pair_point_Sidak_p"),color="#ae404a",ls=":",label="6 pairwise |rho| < 0.1 points: Sidak")
ax[1].axvline(u_obs,color=".5",lw=.7,ls=":")
ax[1].set(title="A region is a search, not a single point",xlabel="Reference threshold u",ylabel="Exceedance probability",ylim=(1e-5,1.1))
ax[1].legend(fontsize=8,loc="lower left")
fig.suptitle("2016, 39–180 MeV; fixed background, 0.5-MeV grid, ±2.25 sigma blind window",fontsize=11)
fig.tight_layout();savefig(fig,"stats_same_field_lee")

fig,ax=plt.subplots(1,2,figsize=(10.5,4.0),gridspec_kw={"width_ratios":[1,1.08]})
R=K[np.ix_(near,near)]
im=ax[0].imshow(R,origin="lower",vmin=-1,vmax=1,cmap="RdBu_r",aspect="equal")
ax[0].set(xlabel="Representative index",ylabel="Representative index",title="Adjacent selected |rho| < 0.1")
fig.colorbar(im,ax=ax[0],label="Fitted-score correlation",fraction=.047,pad=.035)
ax[0].text(.03,.97,"Distant |rho| reaches 0.835",transform=ax[0].transAxes,va="top",fontsize=9,bbox=dict(facecolor="white",alpha=.9,edgecolor="none"))
j=int(np.argmax(np.where(f["observed_r"]>0,(f["observed_r"]-f["a"])/f["s"],-np.inf)))
ref=(f["observed_r"]-f["a"])/f["s"]
ax[1].plot(m,ref,color="#23648d",lw=1.1,label="Observed reference score")
ax[1].scatter(m[near],ref[near],color="#bb793d",s=17,label="24 representatives",zorder=4)
ax[1].scatter(m[j],ref[j],marker="*",s=85,color="#ae404a",zorder=5,label="Full-grid maximum")
for edge in edges:ax[1].axvline(edge,c=".6",alpha=.2,lw=.5)
ax[1].set(xlabel="Mass hypothesis [MeV]",ylabel="Reference score",title="Points discard the search inside each block",xlim=(39,180))
ax[1].legend(fontsize=8,loc="lower right")
fig.tight_layout();savefig(fig,"stats_correlation_selection")
print(json.dumps(dict(block_peak=peak_block_row,regions=region_rows,elapsed_seconds=time.monotonic()-START),indent=2))
