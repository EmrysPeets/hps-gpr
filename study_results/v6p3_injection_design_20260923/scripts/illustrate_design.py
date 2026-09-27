#!/usr/bin/env python3
"""Analytic illustrations of injection design; no HPS spectra are fitted."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import lognorm

ROOT = Path(__file__).resolve().parents[1]
for _dir in ("data", "figures", "qa"):
    (ROOT / _dir).mkdir(exist_ok=True)

# Deliberately illustrative, not a fitted HPS distribution.  Mean(s/s0)=1.
cv = 0.20
z = 3.0
tau2 = math.log1p(cv * cv)
tau = math.sqrt(tau2)
yield_dist = lognorm(s=tau, scale=z * math.exp(-tau2 / 2))
strength_dist = lognorm(s=tau, scale=z * math.exp(tau2 / 2))
x = np.linspace(1.05, 6.3, 900)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.titlesize": 11.5, "axes.labelsize": 11,
    "legend.fontsize": 9, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), layout="constrained")
for ax, dist, title, xlabel, spread_label, fixed_label, color in (
    (axes[0], yield_dist, "Expected signal yield varies under matching",
     r"Expected yield / common scale, $A/s_0$", "Matched per toy",
     "Common yield: always 3", "#2166ac"),
    (axes[1], strength_dist, "Nominal strength varies for a common yield",
     r"Expected yield / toy reference error, $A/s_t$", "Common yield",
     "Matched per toy: always 3", "#a34d16"),
):
    y = dist.pdf(x)
    ax.fill_between(x, 0, y, color=color, alpha=0.17)
    ax.plot(x, y, color=color, lw=2.1, label=spread_label)
    ax.axvline(z, color="#222222", lw=1.8, ls="--", label=fixed_label)
    ax.set(xlim=(1.05, 6.3), ylim=(0, 0.86), title=title, xlabel=xlabel,
           ylabel="Density of the varying quantity")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.15)
fig.savefig(ROOT / "figures" / "injection_design.pdf")
fig.savefig(ROOT / "figures" / "injection_design.png", dpi=200)
plt.close(fig)

# Exact two-state example, including baseline offset and response correlation.
s = np.array([10., 20.])
u = np.array([4., -4.])
g = np.array([.98, .80])
s0 = float(np.mean(s))
rows = []
for strength in (1, 3, 5):
    for scheme, A in (("matched", strength * s),
                      ("common", np.full(2, strength * s0))):
        fitted = u + g * A
        rows.append({
            "z": strength, "scheme": scheme, "A_low_error": A[0],
            "A_high_error": A[1], "mean_A": float(A.mean()),
            "mean_bias": float(np.mean(fitted - A)),
            "mean_raw_recovery": float(np.mean(fitted / A)),
            "mean_paired_response": float(np.mean((fitted-u) / A)),
            "ratio_mean_increment_to_mean_A": float(np.mean(fitted-u) / A.mean()),
        })
with (ROOT / "data" / "two_state_example.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
cov_g_s = float(np.mean((g-g.mean()) * (s-s0)))
for strength in (1, 3, 5):
    matched, common = [row for row in rows if row["z"] == strength]
    assert np.isclose(matched["mean_bias"]-common["mean_bias"], strength*cov_g_s)
    assert np.isclose(matched["mean_paired_response"], g.mean())
    assert np.isclose(common["mean_paired_response"], g.mean())

summary = {
    "scope": "Exact illustrative distributions and a two-state response model; no HPS fits or coverage calibration.",
    "reference_error_cv": cv, "nominal_z": z,
    "matched_yield_over_s0_mean": float(yield_dist.mean()),
    "matched_yield_over_s0_sd": float(yield_dist.std()),
    "matched_yield_over_s0_central90": yield_dist.ppf([.05, .95]).tolist(),
    "common_yield_over_st_mean": float(strength_dist.mean()),
    "common_yield_over_st_central90": strength_dist.ppf([.05, .95]).tolist(),
    "two_state_cov_g_s": cov_g_s,
    "two_state_inputs": {"s": s.tolist(), "u": u.tolist(), "g": g.tolist()},
    "formula_checks": "passed: bias covariance identity, paired response, lognormal means",
}
assert np.isclose(yield_dist.mean(), z)
assert np.isclose(yield_dist.std(), z*cv)
assert np.isclose(strength_dist.mean(), z*(1+cv*cv))
(ROOT / "data" / "illustration_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
(ROOT / "qa" / "analytic_checks.json").write_text(json.dumps({
    "passed": True, "checks": ["mean matched yield equals common yield", "relative matched-yield spread equals CV(s)",
    "bias difference equals z Cov(g,s)", "paired per-toy response equals g in the linear model", "inverse-error lognormal mean"]
}, indent=2)+"\n")
print(json.dumps(summary, indent=2))
