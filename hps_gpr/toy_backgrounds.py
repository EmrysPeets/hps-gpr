"""Shared helpers for full-range background pseudoexperiment generation."""

from __future__ import annotations

import numpy as np


def normalize_full_toy_bkg_mode(mode: str) -> str:
    """Normalize and validate the configured full-range toy background mode."""
    text = str(mode or "poisson").strip().lower()
    if text not in {"poisson", "fixed_total_multinomial"}:
        raise ValueError(
            "full_toy_bkg_mode must be 'poisson' or 'fixed_total_multinomial', "
            f"got {mode!r}"
        )
    return text


def observed_total_count(counts: np.ndarray) -> int:
    """Return the nearest-integer total event count for a histogram."""
    arr = np.asarray(counts, float).reshape(-1)
    total = float(np.sum(arr))
    if not np.isfinite(total) or total <= 0.0:
        return 0
    return int(np.rint(total))


def draw_full_background_toy(
    mean_counts: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: str = "poisson",
    total_count: int | None = None,
) -> np.ndarray:
    """Draw one full-range background pseudoexperiment.

    Parameters
    ----------
    mean_counts:
        Expected counts in each full-range bin.
    rng:
        Random-number generator.
    mode:
        ``poisson`` fluctuates each bin independently.
        ``fixed_total_multinomial`` preserves the exact total count and fluctuates
        only the per-bin allocation.
    total_count:
        Required for ``fixed_total_multinomial`` mode. This should be the total
        number of events in the original full-range histogram used to define the
        toy study.
    """
    lam = np.clip(np.asarray(mean_counts, float).reshape(-1), 0.0, None)
    mode_norm = normalize_full_toy_bkg_mode(mode)

    if mode_norm == "poisson":
        return rng.poisson(lam).astype(int)

    if total_count is None:
        raise ValueError(
            "draw_full_background_toy(..., mode='fixed_total_multinomial') "
            "requires total_count."
        )

    total = int(total_count)
    if total <= 0 or lam.size == 0:
        return np.zeros_like(lam, dtype=int)

    probs = lam.astype(float, copy=True)
    p_sum = float(np.sum(probs))
    if not np.isfinite(p_sum) or p_sum <= 0.0:
        probs = np.full(lam.shape[0], 1.0 / float(lam.shape[0]), dtype=float)
    else:
        probs /= p_sum
    return rng.multinomial(total, probs).astype(int)
