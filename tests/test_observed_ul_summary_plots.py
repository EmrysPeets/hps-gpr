from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hps_gpr.plotting import plot_observed_ul_overlay, plot_ul_pvalues


def test_plot_ul_pvalues_uses_log_scale_with_zero_toy_tails():
    df = pd.DataFrame(
        {
            "mass_GeV": [0.04, 0.05, 0.06],
            "p_strong": [0.0, 1.0e-4, 0.02],
            "p_weak": [1.0, 0.2, 0.08],
            "p_two": [0.0, 2.0e-4, 0.04],
        }
    )

    plot_ul_pvalues(df, title="toy tails")
    ax = plt.gca()
    assert ax.get_yscale() == "log"
    assert ax.get_ylim()[0] > 0.0
    plt.close("all")


def test_plot_observed_ul_overlay_writes_dataset_and_combined_curves(tmp_path: Path):
    dataset = pd.DataFrame({"mass_GeV": [0.04, 0.05], "eps2_obs": [2.0e-8, 1.5e-8]})
    combined = pd.DataFrame({"mass_GeV": [0.04, 0.05], "ul_eps2_obs": [1.2e-8, 1.0e-8]})
    out = tmp_path / "overlay.png"

    plot_observed_ul_overlay(
        [("2015", dataset), ("combined", combined)],
        outpath=str(out),
    )

    assert out.exists()
    assert out.stat().st_size > 0
