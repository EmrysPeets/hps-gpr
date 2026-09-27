"""Write the release README from the completed numerical tables."""
from pathlib import Path
import json
import pandas as pd
B=Path(__file__).resolve().parents[1]
def main():
    s=pd.read_csv(B/'results/summary.csv');p=pd.read_csv(B/'results/paired.csv')
    o=json.loads((B/'results/overview.json').read_text());nt=o['toys_per_cell']
    high=s[(s.control=='contaminated_gp')&s.injected_N.eq(30000)]
    pair=p[p.injected_N.eq(30000)]
    def span(values):return f'{100*values.min():.1f}–{100*values.max():.1f}%'
    raw={m:span(g.mean_recovery) for m,g in high.groupby('method')}
    rows=[]
    for mass,g in high.groupby('mass_MeV'):
        a=g.set_index('method');x=a.loc['pole_centered'];y=a.loc['core_shifted']
        rows.append(f'| {mass}{"*" if mass==260 else ""} | {100*x.mean_recovery:.1f}% | {100*y.mean_recovery:.1f}% | {x.pull_mean:.2f} / {x.pull_width:.2f} | {y.pull_mean:.2f} / {y.pull_width:.2f} | {int(x.profile95_count)}/{nt} | {int(y.profile95_count)}/{nt} |')
    text=f'''# HPS-GPR v6.2: MC signal injection and recovery

This release uses {nt} toys at each of four injection yields (1,000, 5,000, 10,000 and 30,000 selected MC candidates) and eleven native masses (60–260 MeV in 20 MeV steps). The 40 MeV sample is excluded. It contains {o['unique_injected_toys']:,} injected spectra and {o['primary_fits']:,} primary fits comparing windows centered on the generated mass and on the reconstructed MC core.

The first 20 toys and the original release are preserved under `history/20_toy_release/`. The additional toys use indices 20–39 with the same model and seed prescription. `qa/toy_extension.json` records the preservation check; `results/toy_extension_comparison.csv` compares the two 20-toy groups and their combined results.

## Results

The shifted window improves recovery, but some injected signal is still absorbed by the GP prediction. At 30,000 injected candidates, raw mean recovery spans {raw['pole_centered']} for pole-centered windows and {raw['core_shifted']} for shifted windows. After subtracting the fitted yield of each paired zero-signal toy, the incremental response spans {span(pair.mean_incremental_recovery_pole)} and {span(pair.mean_incremental_recovery_core)}, respectively. These ranges include the separately marked 260 MeV extension. The subtraction is a diagnostic; the primary yields, pulls and intervals retain their original values.

| Mass (MeV) | Pole recovery | Shifted recovery | Pole pull mean / width | Shifted pull mean / width | Pole nominal 95% contains truth | Shifted nominal 95% contains truth |
|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

*The 260 MeV point uses the archived 250 MeV kernel settings with the resolution evaluated at 260 MeV. It extends beyond the inherited extraction grid and is not a newly validated scan endpoint.

The deterministic controls isolate the effect of signal in the training bins: the clean-sideband incremental response is essentially one, while the injected-sideband response is lower. The report presents this comparison together with raw yields, zero-signal fits, pulls and interval containment. With {nt} toys per cell, pull widths and containment fractions still have appreciable sampling uncertainty; these are conditional checks, not a coverage calibration.

## Method

The background is sampled as a full Poisson spectrum from the pinned v5.8.2 nominal 2021 GP mean. One generating mean is used for all masses and both extraction methods. Independent background draws are made per mass and toy; each draw is shared across injection yields and extraction methods.

Each signal draw has exactly N selected candidates. A multinomial draw uses the native smeared-MC histogram rebinned by its CDF onto the analysis bins, together with below- and above-support categories. Histogram overflow remains outside the support. The fitted parameter is the full-selected candidate yield, so the signal probabilities are never renormalized within the fitted window. Counts entering the support, fit window, training bins and outside support are saved for every toy.

Both methods use the unchanged native MC shape and the same half-width, 2.25 times the nominal resolution at the generated mass. Only the fit and training window center changes: the pole mass for one method and the fixed MC-only core estimate for the other. The GP mean and correlated covariance are recomputed from each toy's exterior bins; the archived kernel hyperparameters remain fixed.

The likelihood combines Poisson counts with a correlated Gaussian constraint on the background. Signed fitted yields are retained. Pulls use the observed profile-Hessian error. Nominal 68.27% and 95% likelihood-ratio containment uses thresholds 1 and 3.841459; tables provide counts and Clopper–Pearson 95% intervals. The exact-N signal is multinomial, whereas the inherited extraction likelihood is Poisson, so unit pull width and nominal containment are reference values even in the known-background control.

The selected MC is conditional on the supplied v13 production. Signal-daughter association and complete equivalence to the v16 data selection are unvalidated. Templates and centers are fixed, and this study does not propagate finite-MC, detector-response or generating-background uncertainty.

## Files and reproduction

- `pdf/HPS_GPR_v6p2_MC_Injection_Recovery.pdf`: the LaTeX note, formatted consistently with v6.1 and v5.0.5.
- `source/report.tex`, `source/generated/`, `figures/`: editable note, numerical tables and vector figures.
- `results/toys.csv`, `summary.csv`, `paired.csv`, `asimov.csv`: complete fitted results, moments, containment and controls.
- `results/checkpoints/`: saved draws, masks, probabilities, per-mass fits and checksums.
- `qa/editorial_review.md`: independent review of the note's wording and scientific claims.
- `qa/independent_validation.json`, `toy_extension.json`, `report_visual_qa.json`: numerical, extension and rendered-page checks.
- `inputs/`, `provenance/`, `MANIFEST.sha256`: pinned dependencies and artifact identities.

Rebuild the scientific products and note with a Python environment satisfying `requirements.txt` and Tectonic installed:

```sh
python3 /path/to/v6p2_mc_injection_20260923/scripts/reproduce.py
```

To rebuild only the tables, figures and PDF:

```sh
python3 scripts/make_report.py
bash scripts/build_note.sh
```

The full launcher limits work to four local workers with one numerical thread each and a 30-minute watchdog. No S3DF jobs are used. Checkpoint dependencies and output hashes are verified before reuse. The final PDF is delivered only after text and rendered-page review.
'''
    (B/'README.md').write_text(text)
if __name__=='__main__':main()
