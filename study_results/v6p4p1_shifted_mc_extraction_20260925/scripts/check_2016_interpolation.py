"""Check 2016 neighbor interpolation and symmetric fit geometry from pinned inputs.

No fitting or random draws: reuse the v6.4 fitted centers and widths. Compare
linear-width interpolation with its previous log-width counterpart, including
leave-one-mass-out shape checks. All CDFs retain full-selected normalization.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

B = Path(__file__).resolve().parents[1]
V = B / 'inputs/v64'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    table = pd.read_csv(V / 'results/centers_and_shapes.csv').set_index('mass_MeV')
    table = table[table.primary_domain & table.valid]
    masses = table.index.to_numpy(dtype=int)
    assert masses[0] == 40 and masses[-1] == 175 and 150 not in masses
    hist = {}
    for m in masses:
        with np.load(V / 'histograms' / f'm{m:03d}.npz') as data:
            e = data['edges_MeV'].copy()
            p = data['probability'].copy()
        assert np.all(np.diff(e) > 0) and np.all(p >= 0)
        assert abs(p.sum() - 1.) < 1e-12
        hist[m] = (e, np.r_[0., np.cumsum(p)])

    def source_cdf(mass, x):
        e, f = hist[mass]
        return np.interp(x, e, f, left=0., right=1.)

    def parameters(mass, mode, omit=None):
        anchors = masses[masses != omit] if omit is not None else masses
        assert anchors[0] <= mass <= anchors[-1]
        if mass in anchors:
            row = table.loc[mass]
            return row.center_MeV, row.sigma_core_MeV, [(int(mass), 1.)]
        lo, hi = int(anchors[anchors < mass][-1]), int(anchors[anchors > mass][0])
        t = (mass - lo) / (hi - lo)
        center = (1 - t) * table.loc[lo, 'center_MeV'] + t * table.loc[hi, 'center_MeV']
        widths = table.loc[[lo, hi], 'sigma_core_MeV'].to_numpy()
        width = ((1 - t) * widths[0] + t * widths[1] if mode == 'linear'
                 else np.exp((1 - t) * np.log(widths[0]) + t * np.log(widths[1])))
        return float(center), float(width), [(lo, 1 - t), (hi, t)]

    def cdf(mass, x, mode='linear', omit=None):
        center, width, pairs = parameters(mass, mode, omit)
        if len(pairs) == 1:
            return source_cdf(pairs[0][0], x)
        u = (np.asarray(x) - center) / width
        return sum(weight * source_cdf(anchor, table.loc[anchor, 'center_MeV']
                   + table.loc[anchor, 'sigma_core_MeV'] * u)
                   for anchor, weight in pairs)

    old = pd.read_csv(V / 'results/shape_comparisons.csv').set_index('mass_MeV')
    rows = []
    old_agreement = []
    for m in masses[1:-1]:
        center, linear_width, pairs = parameters(m, 'linear', omit=m)
        _, log_width, _ = parameters(m, 'log', omit=m)
        e, truth = hist[m]
        linear = cdf(m, e, 'linear', omit=m)
        previous = cdf(m, e, 'log', omit=m)
        linear_distance = float(np.max(np.abs(linear - truth)))
        log_distance = float(np.max(np.abs(previous - truth)))
        assert np.all(np.diff(linear) >= -1e-12)
        assert np.all(np.diff(previous) >= -1e-12)
        old_agreement.append(abs(log_distance - old.loc[m, 'morph_full_cdf_distance']))
        rows.append(dict(
            mass_MeV=int(m), lower_anchor_MeV=pairs[0][0], upper_anchor_MeV=pairs[1][0],
            upper_weight=pairs[1][1], direct_center_MeV=float(table.loc[m, 'center_MeV']),
            interpolated_center_MeV=center, center_error_MeV=center-table.loc[m, 'center_MeV'],
            direct_width_MeV=float(table.loc[m, 'sigma_core_MeV']),
            linear_width_MeV=linear_width, log_width_MeV=log_width,
            width_change_MeV=linear_width-log_width,
            linear_width_ratio=linear_width/table.loc[m, 'sigma_core_MeV'],
            log_width_ratio=log_width/table.loc[m, 'sigma_core_MeV'],
            linear_full_cdf_distance=linear_distance, log_full_cdf_distance=log_distance,
            linear_minus_log_cdf_distance=linear_distance-log_distance,
            full_cdf_change_max=float(np.max(np.abs(linear-previous))),
        ))
    frame = pd.DataFrame(rows)
    assert max(old_agreement) < 1e-9

    with np.load(B / 'inputs/v6p1/inputs/spectrum_2016.npz') as data:
        x, edges = data['x'] * 1000., data['edges'] * 1000.
    geometry, production = [], []
    native_identity = 0.
    max_category_residual = 0.
    for m in range(40, 176):
        center, width, _ = parameters(m, 'linear')
        _, previous_width, _ = parameters(m, 'log')
        fit = (x >= center-3.5*width) & (x <= center+3.5*width)
        left = int(np.sum(x < center-3.5*width))
        right = int(np.sum(x > center+3.5*width))
        F, G = cdf(m, edges), cdf(m, edges, 'log')
        cats = np.r_[F[0], np.diff(F), 1-F[-1]]
        assert np.min(cats) >= -1e-12
        max_category_residual = max(max_category_residual, abs(float(cats.sum())-1.))
        geometry.append(dict(mass_MeV=m, center_MeV=center, width_MeV=width,
                             left_training_bins=left, right_training_bins=right,
                             fit_bins=int(fit.sum()),
                             usable=bool(left >= 3 and right >= 3 and fit.sum() > 3)))
        production.append(dict(mass_MeV=m, width_change_MeV=width-previous_width,
                               width_relative_change=width/previous_width-1,
                               full_cdf_change_max=float(np.max(np.abs(F-G)))))
        if m in masses:
            native_identity = max(native_identity, float(np.max(np.abs(F-source_cdf(m, edges)))))
            en, fn = hist[m]
            native_identity = max(native_identity, float(np.max(np.abs(cdf(m, en)-fn))))
    assert native_identity < 1e-12
    assert max_category_residual < 1e-12
    geom = pd.DataFrame(geometry)
    prod = pd.DataFrame(production)
    assert geom.usable.all(), geom[~geom.usable].to_dict('records')

    def maximum(frame, column):
        row = frame.loc[frame[column].abs().idxmax()]
        return dict(mass_MeV=int(row.mass_MeV), value=float(row[column]))

    (B / 'results').mkdir(exist_ok=True)
    (B / 'qa').mkdir(exist_ok=True)
    result = B / 'results/2016_interpolation_comparison.csv'
    frame.to_csv(result, index=False, float_format='%.17g')
    report = dict(
        passed=True,
        method='Linear center and linear fitted-core width; neighboring full aligned empirical CDFs.',
        previous_method='Linear center and log-linear fitted-core width; same anchors and CDF mixture.',
        cdf_metric='Maximum absolute CDF difference on the held-out histogram native edges; descriptive, not a p-value.',
        claim_boundary='This checks interpolation and geometry, not yield recovery, coverage or detector/data selection equivalence.',
        native_anchors=len(masses), held_out_interior_anchors=len(frame),
        primary_domain_MeV=[40, 175], missing_native_mass_MeV=150,
        native_identity_max_absolute_error=native_identity,
        full_category_normalization_max_absolute_error=max_category_residual,
        previous_holdout_metric_max_absolute_difference=max(old_agreement),
        holdout=dict(
            linear_full_cdf_distance_median=float(frame.linear_full_cdf_distance.median()),
            log_full_cdf_distance_median=float(frame.log_full_cdf_distance.median()),
            linear_full_cdf_distance_max=maximum(frame, 'linear_full_cdf_distance'),
            log_full_cdf_distance_max=maximum(frame, 'log_full_cdf_distance'),
            full_cdf_change_max=maximum(frame, 'full_cdf_change_max'),
            width_change_max_MeV=maximum(frame, 'width_change_MeV'),
            center_error_max_MeV=maximum(frame, 'center_error_MeV'),
            linear_width_relative_error_max=float(np.max(np.abs(frame.linear_width_ratio-1.))),
        ),
        production_grid=dict(
            mass_step_MeV=1, masses=len(prod),
            full_cdf_change_max=maximum(prod, 'full_cdf_change_max'),
            width_change_max_MeV=maximum(prod, 'width_change_MeV'),
            width_relative_change_max=maximum(prod, 'width_relative_change'),
            cdf_metric_grid='Archived 2016 observed-spectrum bin edges',
        ),
        geometry=dict(
            definition='u=(reconstructed mass-interpolated fitted-core center)/interpolated fitted-core width; fit and GP exclusion both [-3.5,3.5]; bin centers select whole bins.',
            minimum_left_training_bins=int(geom.left_training_bins.min()),
            minimum_left_at_masses_MeV=geom.loc[geom.left_training_bins == geom.left_training_bins.min(), 'mass_MeV'].astype(int).tolist(),
            minimum_right_training_bins=int(geom.right_training_bins.min()),
            minimum_right_at_masses_MeV=geom.loc[geom.right_training_bins == geom.right_training_bins.min(), 'mass_MeV'].astype(int).tolist(),
            minimum_fit_bins=int(geom.fit_bins.min()), maximum_fit_bins=int(geom.fit_bins.max()),
            unusable_points=geom[~geom.usable].to_dict('records'),
            spectrum_bins=len(x), spectrum_edge_support_MeV=[float(edges[0]), float(edges[-1])],
        ),
        hashes=dict(script_sha256=sha(__file__), result_sha256=sha(result),
                    centers_sha256=sha(V / 'results/centers_and_shapes.csv'),
                    spectrum_sha256=sha(B / 'inputs/v6p1/inputs/spectrum_2016.npz')),
    )
    (B / 'qa/2016_interpolation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
