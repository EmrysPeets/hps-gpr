import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hps_gpr.plotting import plot_ul_pvalue_components
from hps_gpr.statistics import (
    _p_from_z_one_sided,
    _p_global_from_local,
    _p_local_from_global_summary,
    _z_from_p_one_sided,
)


def test_sidak_global_p_and_z_are_less_significant_than_local():
    p_local = np.asarray([0.05, 1.0e-3, 2.0e-6], float)
    p_global = _p_global_from_local(p_local, Neff=25.0, method="sidak")

    assert np.all(p_global >= p_local)
    assert np.all(_z_from_p_one_sided(p_global) <= _z_from_p_one_sided(p_local))


def test_inverse_global_threshold_round_trips_to_target_global_p():
    neff = 42.0
    for z in [1.0, 2.0, 3.0, 5.0]:
        p_global_target = _p_from_z_one_sided(z)
        p_local_threshold = _p_local_from_global_summary(
            p_global_target,
            neff=neff,
            method="sidak",
        )
        p_roundtrip = _p_global_from_local(
            np.asarray([p_local_threshold], float),
            Neff=neff,
            method="sidak",
        )[0]

        assert p_local_threshold < p_global_target
        assert np.isclose(p_roundtrip, p_global_target, rtol=1e-8, atol=1e-15)


def test_toy_limit_component_plot_draws_global_threshold_below_local_line():
    df = pd.DataFrame(
        {
            "mass_GeV": [0.08, 0.09, 0.10],
            "p_strong": [0.40, 0.12, 0.08],
            "p_weak": [0.60, 0.88, 0.92],
            "p_two": [0.80, 0.24, 0.16],
            "sigma_mass_res_GeV": [0.002, 0.002, 0.002],
        }
    )
    neff = 20.0
    local_2sigma = _p_from_z_one_sided(2.0)
    global_2sigma_threshold = _p_local_from_global_summary(
        local_2sigma,
        neff=neff,
        method="sidak",
    )

    plot_ul_pvalue_components(df, neff=neff, lee_method="sidak")
    ax = plt.gca()
    horizontal = []
    for line in ax.lines:
        y = np.asarray(line.get_ydata(), float)
        if y.size == 2 and np.allclose(y, y[0]):
            horizontal.append(float(y[0]))

    assert global_2sigma_threshold < local_2sigma
    assert any(np.isclose(y, local_2sigma, rtol=0.0, atol=1e-12) for y in horizontal)
    assert any(np.isclose(y, global_2sigma_threshold, rtol=0.0, atol=1e-12) for y in horizontal)
    assert ax.get_ylim()[0] < global_2sigma_threshold
    plt.close("all")
