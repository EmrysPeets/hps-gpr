#!/usr/bin/env python3
"""Generate refit-closure matrix configs from the current GP-mean bases.

The matrix keeps the data blinding fixed at 1.64 sigma and scans the toy-refit
training exclusion together with the resolution-scaled lower length-scale
factor.  Generated configs are intentionally separate from the publication
configs so exploratory failures cannot silently tune the final procedure.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


BASE_CONFIGS = {
    "2015": Path("study_configs/config_2015_blind1p64_95CL_10k_injection_gpmean_pseudoexp.yaml"),
    "2016": Path("study_configs/config_2016_10pct_blind1p64_95CL_10k_injection_gpmean_pseudoexp.yaml"),
}

TRAIN_EXCLUDE_NSIGMA = (1.64, 1.98, 2.58, 3.0)
LS_LOWER_FACTORS = (0.5, 1.0, 1.5)


def _tag(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return payload


def _set_ls_lower_factor(cfg: dict, value: float) -> None:
    cfg["kernel_ls_res_lower_factor"] = float(value)
    by_ds = dict(cfg.get("kernel_ls_res_lower_factor_by_dataset") or {})
    if by_ds:
        for key in list(by_ds.keys()):
            by_ds[str(key)] = float(value)
    cfg["kernel_ls_res_lower_factor_by_dataset"] = by_ds


def build_matrix_configs(
    *,
    repo_root: Path,
    output_dir: Path,
    datasets: list[str],
    signal_model: str,
) -> list[Path]:
    written: list[Path] = []
    for dataset in datasets:
        base_path = repo_root / BASE_CONFIGS[dataset]
        base = _load_yaml(base_path)
        for train_nsigma in TRAIN_EXCLUDE_NSIGMA:
            for ls_lower in LS_LOWER_FACTORS:
                cfg = copy.deepcopy(base)
                cfg["blind_nsigma"] = 1.64
                cfg["inj_refit_gp_on_toy"] = True
                cfg["inj_refit_gp_optimize"] = True
                cfg["inj_refit_fail_on_error"] = False
                cfg["inj_train_exclude_nsigma"] = float(train_nsigma)
                cfg["signal_model"] = str(signal_model)
                _set_ls_lower_factor(cfg, float(ls_lower))

                train_tag = _tag(train_nsigma)
                ls_tag = _tag(ls_lower)
                model_tag = "" if str(signal_model) == "default" else f"_{signal_model}"
                cfg["output_dir"] = (
                    f"outputs/refit_closure_matrix/{dataset}{model_tag}/"
                    f"train{train_tag}_lslb{ls_tag}"
                )
                out_path = (
                    output_dir
                    / f"config_{dataset}_refit_train{train_tag}_lslb{ls_tag}{model_tag}.yaml"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(cfg, handle, sort_keys=False)
                written.append(out_path)
    return written


def write_command_file(path: Path, configs: list[Path], repo_root: Path, n_toys: int) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# One run per generated matrix config.",
    ]
    for cfg_path in configs:
        rel = cfg_path.relative_to(repo_root)
        name = cfg_path.stem
        dataset = "2016" if "config_2016_" in name else "2015"
        lines.extend(
            [
                "",
                f'echo "[refit-matrix] {name}"',
                (
                    "python -m hps_gpr.cli inject "
                    f"--config {rel} "
                    f"--dataset {dataset} "
                    "--masses 0.03,0.06,0.09 "
                    "--strengths s0,s1,s2,s3,s4,s5 "
                    f"--n-toys {int(n_toys)} "
                    "--stream-aggregate "
                    "--no-write-toy-csv"
                ),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(BASE_CONFIGS.keys()),
        default=sorted(BASE_CONFIGS.keys()),
        help="Datasets to generate.",
    )
    parser.add_argument(
        "--output-dir",
        default="study_configs/refit_closure_matrix",
        help="Directory for generated YAML configs.",
    )
    parser.add_argument(
        "--commands",
        default="study_configs/refit_closure_matrix/run_refit_closure_matrix.sh",
        help="Optional command helper to write.",
    )
    parser.add_argument(
        "--n-toys",
        type=int,
        default=1000,
        help="Toy count to place in the generated command helper.",
    )
    parser.add_argument(
        "--signal-model",
        choices=["default", "kernel"],
        default="default",
        help="Signal model to write into generated configs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    written = build_matrix_configs(
        repo_root=repo_root,
        output_dir=output_dir,
        datasets=[str(x) for x in args.datasets],
        signal_model=str(args.signal_model),
    )
    if args.commands:
        write_command_file(repo_root / args.commands, written, repo_root, args.n_toys)

    print(f"Wrote {len(written)} matrix configs to {output_dir}")
    if args.commands:
        print(f"Wrote command helper to {repo_root / args.commands}")


if __name__ == "__main__":
    main()
