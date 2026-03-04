from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from hps_gpr.cli import main


def test_inject_plot_runs_with_summary_csvs_only(tmp_path: Path):
    rows = []
    for m in (0.050, 0.060):
        for z, a in ((1.0, 10.0), (2.0, 20.0)):
            rows.append(
                {
                    "dataset": "2015",
                    "mass_GeV": m,
                    "strength": a,
                    "inj_nsigma": z,
                    "n_toys": 100,
                    "A_hat_mean": a + 0.2,
                    "sigma_A_mean": 2.0,
                    "pull_mean": 0.02,
                    "pull_std": 1.01,
                    "cov_1sigma": 0.68,
                    "cov_2sigma": 0.95,
                    "Zhat_mean": z + 0.02,
                    "Zhat_q16": z - 1.0,
                    "Zhat_q84": z + 1.0,
                    "sigmaA_ref": 2.0,
                }
            )
    df = pd.DataFrame(rows)
    inp = tmp_path / "injection_flat"
    inp.mkdir(parents=True, exist_ok=True)
    df.to_csv(inp / "inj_extract_summary_2015__jobds_2015__m_0p05__s_s1.csv", index=False)

    outdir = tmp_path / "injection_summary"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "inject-plot",
            "--input-dir",
            str(inp),
            "--output-dir",
            str(outdir),
        ],
    )

    assert res.exit_code == 0, res.output
    assert (outdir / "coverage_2015.png").exists()
    assert (outdir / "pull_vs_mass_2015.png").exists()
    assert (outdir / "z_calibration_residual_2015.png").exists()


def test_inject_plot_handles_toy_rows_mislabeled_as_summary(tmp_path: Path):
    rows = []
    for toy in range(20):
        rows.append(
            {
                "dataset": "2015",
                "mass_GeV": 0.050,
                "strength": 10.0,
                "inj_nsigma": 1.0,
                "toy": toy,
                "A_hat": 10.0 + 0.1 * toy,
                "sigma_A": 2.0,
                "pull_param": 0.01 * toy,
                "Zhat": 1.0 + 0.01 * toy,
                "sigmaA_ref": 2.0,
                "success": True,
            }
        )
    df = pd.DataFrame(rows)

    inp = tmp_path / "injection_flat"
    inp.mkdir(parents=True, exist_ok=True)
    df.to_csv(inp / "inj_extract_summary_2015__jobds_2015__m_0p05__s_s1.csv", index=False)

    outdir = tmp_path / "injection_summary"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "inject-plot",
            "--input-dir",
            str(inp),
            "--output-dir",
            str(outdir),
        ],
    )

    assert res.exit_code == 0, res.output
    assert (outdir / "inj_extract_summary_2015.csv").exists()
    assert (outdir / "z_calibration_residual_2015.png").exists()
