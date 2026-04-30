#!/usr/bin/env python3
"""Write 2015 functional-form refit lock-in length-scale comparator configs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml


MASSES = [0.030, 0.045, 0.060, 0.075, 0.090, 0.105, 0.120]
STRENGTHS = ["s0", "s1", "s2", "s3", "s5"]
BASE_CONFIG = Path("study_configs/config_2015_blind1p64_95CL_funcform100_refit_lslb1p5_train1p96.yaml")
HELPER_DIR = Path("study_configs/funcform_refit_lockin_lslb")
COMPARISON_OUTDIR = "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_comparison"


@dataclass(frozen=True)
class Study:
    value: float
    yaml_tag: str
    output_tag: str

    @property
    def config_name(self) -> str:
        return f"config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb{self.yaml_tag}.yaml"

    @property
    def output_dir(self) -> str:
        return f"outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_{self.output_tag}"

    @property
    def slurm_tag(self) -> str:
        return f"lslb{self.yaml_tag}"

    @property
    def job_name(self) -> str:
        return f"hps2015_ffinj_{self.slurm_tag}_tr196"


STUDIES = (
    Study(0.5, "0p5", "0pt5"),
    Study(1.0, "1p0", "1pt0"),
    Study(1.5, "1p5", "1pt5"),
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return payload


def _dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _set_lockin_common(cfg: dict) -> None:
    cfg.update(
        {
            "blind_nsigma": 1.64,
            "gp_train_exclude_nsigma": 1.96,
            "ul_bands_train_exclude_nsigma": 1.96,
            "inj_train_exclude_nsigma": 1.96,
            "inj_shape_mode": "full",
            "signal_model": "default",
            "inject_signal": True,
            "inj_dataset_key": "2015",
            "inj_masses_gev": MASSES,
            "inj_mode": "poisson",
            "inj_strength_mode": "sigmaA",
            "inj_sigma_a_source": "asimov",
            "inj_refit_gp_on_toy": True,
            "inj_refit_gp_restarts": 0,
            "inj_refit_gp_optimize": True,
            "inj_sigma_multipliers": [0.0, 1.0, 2.0, 3.0, 5.0],
            "inj_write_toy_csv": True,
            "inj_write_qmu": True,
            "inj_stream_aggregate": True,
            "inj_aggregate_every": 100,
            "inj_n_workers": 1,
            "inj_threads_per_worker": 1,
            "enable_2015": True,
            "enable_2016": False,
            "enable_2021": False,
            "do_combined": False,
            "run_limit_bands_on": "2015",
            "make_eps2_bands": True,
        }
    )


def _set_ls_lower_factor(cfg: dict, value: float) -> None:
    cfg["kernel_ls_res_lower_factor"] = float(value)
    by_dataset = dict(cfg.get("kernel_ls_res_lower_factor_by_dataset") or {})
    by_dataset["2015"] = float(value)
    by_dataset["2016"] = float(by_dataset.get("2016", 0.5))
    by_dataset["2021"] = float(by_dataset.get("2021", 0.5))
    cfg["kernel_ls_res_lower_factor_by_dataset"] = by_dataset


def _build_config(base: dict, study: Study) -> dict:
    cfg = copy.deepcopy(base)
    _set_lockin_common(cfg)
    _set_ls_lower_factor(cfg, study.value)
    cfg["output_dir"] = study.output_dir
    return cfg


def _normalized_for_comparison(cfg: dict) -> dict:
    normalized = copy.deepcopy(cfg)
    normalized["output_dir"] = "<OUTPUT_DIR>"
    normalized["kernel_ls_res_lower_factor"] = "<LS_LOWER>"
    by_dataset = dict(normalized.get("kernel_ls_res_lower_factor_by_dataset") or {})
    by_dataset["2015"] = "<LS_LOWER>"
    normalized["kernel_ls_res_lower_factor_by_dataset"] = by_dataset
    return normalized


def write_configs(repo_root: Path) -> dict[Study, Path]:
    base_path = repo_root / BASE_CONFIG
    base = _load_yaml(base_path)
    written: dict[Study, Path] = {}
    normalized_ref: dict | None = None

    for study in STUDIES:
        cfg = _build_config(base, study)
        out_path = repo_root / "study_configs" / study.config_name
        _dump_yaml(out_path, cfg)
        written[study] = out_path
        print(f"Wrote {out_path.relative_to(repo_root)}")

        normalized = _normalized_for_comparison(cfg)
        if normalized_ref is None:
            normalized_ref = normalized
        elif normalized != normalized_ref:
            raise RuntimeError("Generated configs differ by more than output_dir and length-scale lower bound")

    print("Validated generated YAMLs: only output_dir and 2015 length-scale lower-bound fields differ.")
    return written


def _write_executable(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    print(f"Wrote {path}")


def write_helpers(repo_root: Path, configs: dict[Study, Path]) -> None:
    helper_dir = repo_root / HELPER_DIR
    masses_arg = ",".join(f"{mass:.3f}" for mass in MASSES)
    strengths_arg = ",".join(STRENGTHS)

    smoke_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"',
        'cd "${REPO_ROOT}"',
        "",
        "# Smoke-test the new comparator settings on one functional-form toy.",
    ]
    for study in STUDIES[:2]:
        smoke_lines.extend(
            [
                "",
                f'echo "[smoke] {study.slurm_tag}"',
                "hps-gpr funcform-inject \\",
                f"  --config study_configs/{study.config_name} \\",
                "  --dataset 2015 \\",
                "  --max-toys 1 \\",
                "  --masses 0.060 \\",
                "  --strengths s1 \\",
                "  --n-injection-toys 1 \\",
                "  --write-toy-csv \\",
                "  --write-qmu \\",
                f"  --output-dir outputs/smoke_2015_funcform_refit_lockin_{study.slurm_tag}",
            ]
        )
    _write_executable(helper_dir / "run_smoke.sh", smoke_lines)

    generate_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"',
        'cd "${REPO_ROOT}"',
        "",
        f'MASSES="{masses_arg}"',
        f'STRENGTHS="{strengths_arg}"',
        "",
        "generate_one() {",
        '  local tag="$1"',
        '  local config="$2"',
        '  local job_name="$3"',
        '  local slurm_dir="study_configs/funcform_refit_lockin_lslb/slurm_${tag}"',
        '  mkdir -p "${slurm_dir}"',
        "  hps-gpr slurm-gen-funcform-inject \\",
        '    --config "${config}" \\',
        "    --dataset 2015 \\",
        '    --masses "${MASSES}" \\',
        '    --strengths "${STRENGTHS}" \\',
        "    --n-injection-toys 1 \\",
        "    --write-qmu \\",
        "    --cpus-per-task 1 \\",
        '    --job-name "${job_name}" \\',
        "    --partition roma \\",
        "    --account hps:hps-prod \\",
        "    --time 1:00:00 \\",
        "    --memory 8G \\",
        '    --output "${slurm_dir}/submit_funcform_injection_${tag}.slurm"',
        "}",
    ]
    for study, cfg_path in configs.items():
        generate_lines.extend(
            [
                "",
                f'generate_one "{study.slurm_tag}" "{cfg_path.relative_to(repo_root)}" "{study.job_name}"',
            ]
        )
    _write_executable(helper_dir / "generate_slurm_all.sh", generate_lines)

    submit_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"',
        'cd "${REPO_ROOT}"',
        "",
        "SCRIPTS=(",
    ]
    for study in STUDIES:
        submit_lines.append(
            f"  study_configs/funcform_refit_lockin_lslb/slurm_{study.slurm_tag}/submit_funcform_injection_all.sh"
        )
    submit_lines.extend(
        [
            ")",
            "",
            'for script in "${SCRIPTS[@]}"; do',
            '  if [ ! -x "${script}" ]; then',
            '    echo "Missing ${script}; run study_configs/funcform_refit_lockin_lslb/generate_slurm_all.sh first." >&2',
            "    exit 1",
            "  fi",
            '  echo "[submit] ${script}"',
            '  bash "${script}" --account=hps:hps-prod --partition=roma',
            "done",
        ]
    )
    _write_executable(helper_dir / "submit_all.sh", submit_lines)

    compile_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"',
        'cd "${REPO_ROOT}"',
        "",
        "OUTDIRS=(",
    ]
    for study in STUDIES:
        compile_lines.append(f"  {study.output_dir}")
    compile_lines.extend(
        [
            ")",
            "",
            'for outdir in "${OUTDIRS[@]}"; do',
            '  echo "[inject-plot] ${outdir}"',
            "  hps-gpr inject-plot \\",
            '    -i "${outdir}" \\',
            '    -o "${outdir}/injection_summary" \\',
            "    --dataset 2015 \\",
            "    --write-merged-toys",
            "done",
            "",
            "python3 study_configs/funcform_refit_lockin_lslb/compare_lslb_studies.py \\",
            f"  --output-dir {COMPARISON_OUTDIR}",
        ]
    )
    _write_executable(helper_dir / "compile_all.sh", compile_lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    configs = write_configs(repo_root)
    write_helpers(repo_root, configs)


if __name__ == "__main__":
    main()
