#!/usr/bin/env python3
"""v16 TC signal templates with full-selected-yield normalization.

Mass and edge arguments use MeV. The histogram overflow remains an unresolved
upper-tail category: finite stored-bin probabilities are never renormalized.
The common and neighboring-anchor models interpolate core location and width,
then mix complete aligned empirical CDFs, including the measured broad tails.
"""
from pathlib import Path
import csv
import json
import numpy as np

B = Path(__file__).resolve().parents[1]


class TemplateBank:
    def __init__(self, base=None):
        self.base = Path(base) if base is not None else B
        self.samples = {}
        with (self.base / 'inputs/tc_core_fits.csv').open() as handle:
            rows = {int(row['mass_MeV']): row for row in csv.DictReader(handle)}
        for path in sorted((self.base / 'inputs/tc').glob('m*.npz')):
            mass = int(path.stem[1:])
            meta = json.loads(path.with_suffix('.json').read_text())
            with np.load(path) as data:
                edges = np.asarray(data['edges_GeV'], float) * 1000
                counts = np.asarray(data['sumw'], float)
            total = float(meta['sumw'])
            under = float(meta['underflow_sumw'])
            over = float(meta['overflow_sumw'])
            assert abs(counts.sum() + under + over - total) < 1e-6
            assert np.all(counts >= 0) and np.all(np.diff(edges) > 0)
            self.samples[mass] = dict(
                edges=edges, counts=counts, total=total, under=under, over=over,
                cdf=(under + np.r_[0., np.cumsum(counts)]) / total,
                center=float(rows[mass]['core_center_MeV']),
                width=float(rows[mass]['fitted_core_sigma_MeV']),
            )
        self.anchors = np.array([m for m in sorted(self.samples) if 80 <= m <= 240])

    def _retained(self, omit):
        omitted = set() if omit is None else set(np.atleast_1d(omit).astype(float))
        return np.array([m for m in self.anchors if m not in omitted])

    def neighbors(self, mass, omit=None):
        """Return interpolation anchors and their convex weights; no extrapolation."""
        mass = float(mass)
        retained = self._retained(omit)
        if mass < retained[0] or mass > retained[-1]:
            raise ValueError(f'{mass:g} MeV lies outside retained anchors {retained.tolist()}')
        if mass in retained:
            return [(int(mass), 1.)]
        right = int(np.searchsorted(retained, mass))
        low, high = int(retained[right - 1]), int(retained[right])
        weight = (mass - low) / (high - low)
        return [(low, 1. - weight), (high, weight)]

    def parameters(self, mass, kind='morph', omit=None):
        if kind == 'direct':
            sample = self.samples[int(mass)]
            if int(mass) != mass:
                raise ValueError('Direct MC exists only at generated sample masses')
            return sample['center'], sample['width']
        pairs = self.neighbors(mass, omit)
        return tuple(sum(weight * self.samples[m][key] for m, weight in pairs)
                     for key in ('center', 'width'))

    def _aligned_cdf(self, anchor, u):
        sample = self.samples[anchor]
        position = sample['center'] + sample['width'] * np.asarray(u)
        return np.interp(position, sample['edges'], sample['cdf'],
                         left=sample['cdf'][0], right=sample['cdf'][-1])

    def cdf(self, mass, edges_MeV, kind='morph', omit=None):
        edges = np.asarray(edges_MeV, dtype=float)
        if kind == 'direct':
            if int(mass) != mass or int(mass) not in self.samples:
                raise ValueError('Direct MC requires an available generated mass')
            sample = self.samples[int(mass)]
            return np.interp(edges, sample['edges'], sample['cdf'],
                             left=sample['cdf'][0], right=sample['cdf'][-1])
        center, width = self.parameters(mass, kind, omit)
        u = (edges - center) / width
        if kind == 'morph':
            pairs = self.neighbors(mass, omit)
        elif kind == 'common':
            retained = self._retained(omit)
            pairs = [(int(m), 1. / len(retained)) for m in retained]
        else:
            raise ValueError(f'Unknown template model: {kind}')
        return sum(weight * self._aligned_cdf(m, u) for m, weight in pairs)

    def probabilities(self, mass, edges_MeV, kind='morph', omit=None):
        edges = np.asarray(edges_MeV, dtype=float)
        if np.any(np.diff(edges) <= 0):
            raise ValueError('Histogram edges must increase strictly')
        probs = np.diff(self.cdf(mass, edges, kind, omit))
        if probs.min() < -1e-12:
            raise ValueError('Template CDF is not monotone')
        return np.maximum(probs, 0.)

    def categories(self, mass, edges_MeV, kind='morph', omit=None):
        cdf = self.cdf(mass, edges_MeV, kind, omit)
        probabilities = np.r_[cdf[0], np.diff(cdf), 1. - cdf[-1]]
        if probabilities.min() < -1e-12:
            raise ValueError('Template category probability is negative')
        probabilities = np.maximum(probabilities, 0.)
        assert abs(probabilities.sum() - 1) < 1e-12
        return probabilities


_DEFAULT = None


def bank():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TemplateBank()
    return _DEFAULT


def center_width(mass, method='morph', omit=None):
    return bank().parameters(mass, method, omit)


def probabilities(mass, edges_MeV, method='morph', omit=None):
    return bank().probabilities(mass, edges_MeV, method, omit)


def categories(mass, edges_MeV, method='morph', omit=None):
    return bank().categories(mass, edges_MeV, method, omit)


def cdf(mass, edges_MeV, method='morph', omit=None):
    return bank().cdf(mass, edges_MeV, method, omit)
