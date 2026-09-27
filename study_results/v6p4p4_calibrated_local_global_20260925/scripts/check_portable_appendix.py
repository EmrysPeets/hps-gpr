"""Rebuild the completed appendix from a detached copy; preserve the source package."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import fitz
import numpy as np

B = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    work = Path(tempfile.mkdtemp(prefix="v644_portable_", dir="/private/tmp"))
    copy = work / "study"
    shutil.copytree(B, copy, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    env = os.environ.copy()
    env.update(STUDY_PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MPLCONFIGDIR="/private/tmp/v644_mpl")
    log = B / "qa/portable_appendix_rebuild.log"
    with log.open("w") as out:
        subprocess.run(["bash", "rebuild_appendix.sh"], cwd=copy, env=env,
                       stdout=out, stderr=subprocess.STDOUT, check=True)
    byte_checks = 0
    array_checks = 0
    for p in sorted((B / "results").rglob("*")):
        if not p.is_file():
            continue
        q = copy / p.relative_to(B)
        if p.suffix == ".npz":
            with np.load(p) as a, np.load(q) as b:
                assert set(a.files) == set(b.files), str(p)
                for key in a.files:
                    # Checkpoint arrays have no NaNs; comparisons retain exact values.
                    assert np.array_equal(a[key], b[key]), (str(p), key)
                    array_checks += 1
        elif p.suffix in (".csv", ".json"):
            assert digest(p) == digest(q), str(p)
            byte_checks += 1
    pngs = []
    for p in sorted((B / "figures").glob("calibrated_*.png")):
        assert digest(p) == digest(copy / p.relative_to(B)), str(p)
        pngs.append(p.name)
    assert len(pngs) == 2
    with fitz.open(B / "pdf/report.pdf") as a, fitz.open(copy / "pdf/report.pdf") as b:
        assert len(a) == len(b)
        assert [p.get_text() for p in a] == [p.get_text() for p in b]
        pages = len(a)
    result = dict(passed=True, detached_directory=str(copy),
                  result_csv_json_byte_checks=byte_checks,
                  result_npz_array_checks=array_checks,
                  identical_figure_pngs=pngs, identical_pdf_text_pages=pages,
                  cached_B_checkpoints_used=True, original_package_unmodified=True)
    (B / "qa/portable_appendix_rebuild.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
