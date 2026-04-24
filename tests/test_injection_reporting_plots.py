from pathlib import Path

import numpy as np
import pandas as pd

from hps_gpr.plotting import _summarize_pull_vs_mass_rows, plot_pull_histogram_by_mass


def test_summarize_pull_vs_mass_rows_regroups_fragmented_single_toy_summaries():
    df = pd.DataFrame(
        [
            {"mass_GeV": 0.05, "_inj_level": 1.0, "pull_mean": 0.5, "pull_std": np.nan, "n_toys": 1},
            {"mass_GeV": 0.05, "_inj_level": 1.0, "pull_mean": -0.3, "pull_std": np.nan, "n_toys": 1},
            {"mass_GeV": 0.05, "_inj_level": 1.0, "pull_mean": 0.1, "pull_std": np.nan, "n_toys": 1},
            {"mass_GeV": 0.06, "_inj_level": 3.0, "pull_mean": -1.0, "pull_std": np.nan, "n_toys": 1},
            {"mass_GeV": 0.06, "_inj_level": 3.0, "pull_mean": -0.4, "pull_std": np.nan, "n_toys": 1},
        ]
    )

    out = _summarize_pull_vs_mass_rows(df, has_toy_pull=False)

    row_1 = out[(out["mass_GeV"] == 0.05) & (out["inj_level"] == 1.0)].iloc[0]
    row_3 = out[(out["mass_GeV"] == 0.06) & (out["inj_level"] == 3.0)].iloc[0]

    assert np.isclose(row_1["pull_mean"], np.mean([0.5, -0.3, 0.1]))
    assert np.isclose(row_1["pull_std"], np.std([0.5, -0.3, 0.1], ddof=1))
    assert np.isclose(row_3["pull_mean"], np.mean([-1.0, -0.4]))
    assert np.isclose(row_3["pull_std"], np.std([-1.0, -0.4], ddof=1))


def test_plot_pull_histogram_by_mass_groups_sigma_scaled_toys_by_injected_level(tmp_path: Path):
    rows = []
    for strength, pulls in ((10.0, [0.2, -0.1]), (12.0, [0.1, 0.0])):
        for i, pull in enumerate(pulls):
            rows.append(
                {
                    "dataset": "2015",
                    "mass_GeV": 0.05,
                    "strength": strength,
                    "inj_nsigma": 1.0,
                    "pull_param": pull,
                    "toy": i,
                }
            )
    for i, pull in enumerate([-1.2, -0.8, -0.4]):
        rows.append(
            {
                "dataset": "2015",
                "mass_GeV": 0.05,
                "strength": 30.0,
                "inj_nsigma": 3.0,
                "pull_param": pull,
                "toy": 10 + i,
            }
        )

    paths = plot_pull_histogram_by_mass(
        pd.DataFrame(rows),
        dataset_key="2015",
        group_by_strength=True,
        outdir=str(tmp_path),
    )

    names = sorted(Path(p).name for p in paths)
    assert len(names) == 2
    assert any("_Z1" in name for name in names)
    assert any("_Z3" in name for name in names)
