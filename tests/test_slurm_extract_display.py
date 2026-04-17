from pathlib import Path

from hps_gpr.slurm import (
    generate_extraction_display_slurm_scripts,
    generate_injection_slurm_scripts,
    generate_slurm_script,
)


def test_generate_extraction_display_slurm_scripts_writes_expected_commands(tmp_path):
    job = tmp_path / "submit_extract_display.slurm"

    job_script, submit_script, n_jobs = generate_extraction_display_slurm_scripts(
        config_path="study_configs/config_2015_extraction_display_v15p8.yaml",
        output_path=str(job),
        dataset="combined",
        dataset_keys=["2015", "2016", "2021"],
        masses=[0.040, 0.080],
        strengths=[3.0, 5.0],
        output_root="outputs/extraction_display_batch",
        mass_range=(0.03, 0.25),
    )

    assert n_jobs == 4
    job_text = Path(job_script).read_text()
    submit_text = Path(submit_script).read_text()
    assert "hps-gpr extract-display" in job_text
    assert '--masses "${EXTRACT_MASS}"' in job_text
    assert '--strengths "${EXTRACT_STRENGTH}"' in job_text
    assert 'EXTRACT_DATASET_KEYS_CSV="${EXTRACT_DATASET_KEYS//:/,}"' in job_text
    assert '--datasets "${EXTRACT_DATASET_KEYS_CSV}"' in job_text
    assert 'EXTRACT_DATASET="combined"' in submit_text
    assert 'EXTRACT_DATASET_KEYS="2015:2016:2021"' in submit_text
    assert "EXTRACT_MASS=0.08" in submit_text


def test_generate_scan_slurm_script_writes_cpus_per_task(tmp_path):
    job = tmp_path / "submit_scan.slurm"
    job_script, _ = generate_slurm_script(
        config_path="config_example.yaml",
        n_jobs=3,
        output_path=str(job),
        cpus_per_task=6,
    )
    assert "#SBATCH --cpus-per-task=6" in Path(job_script).read_text()


def test_generate_injection_slurm_script_writes_cpus_per_task(tmp_path):
    job = tmp_path / "submit_inject.slurm"
    job_script, _, n_jobs = generate_injection_slurm_scripts(
        config_path="config_example.yaml",
        output_path=str(job),
        datasets=["2015"],
        masses=[0.04],
        strengths=[1.0, 2.0],
        n_toys=10,
        output_root="outputs/injection",
        cpus_per_task=4,
    )
    assert n_jobs == 2
    assert "#SBATCH --cpus-per-task=4" in Path(job_script).read_text()
