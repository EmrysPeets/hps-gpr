from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import audit_scan_optimization as audit


def _row(
    *,
    seed: int,
    lml: float,
    const_opt: float,
    ls_opt: float,
    attempt: int,
) -> dict:
    return {
        "truth_model": "gengamma",
        "study_scenario": "2021_1pct_x100",
        "background_toy_index": 0,
        "mass_GeV": 0.05,
        "ls_upper_factor_requested": 20,
        "task_id": "scan__f20__gengamma__2021_1pct_x100__t0000",
        "attempt_number": attempt,
        "attempt_path": f"/tmp/attempt_{attempt:03d}",
        "result_path": f"/tmp/attempt_{attempt:03d}/result.csv",
        "lml": lml,
        "ls_lo": 0.01,
        "ls_hi": 0.20,
        "ls_init": 0.05,
        "ls_opt": ls_opt,
        "const_opt": const_opt,
        "optimizer_seed": seed,
    }


class AttemptAwareAuditTests(unittest.TestCase):
    def test_independent_initialization_reproduction_still_fails(self) -> None:
        rows = pd.DataFrame(
            [
                _row(seed=11, lml=100.0, const_opt=1.0, ls_opt=0.05, attempt=1),
                _row(seed=22, lml=100.0, const_opt=1.0, ls_opt=0.05, attempt=2),
            ]
        )
        selected = audit._select_reviewed_rows(rows).iloc[0]
        self.assertTrue(selected["initialization_lock"])
        self.assertTrue(
            selected["initialization_state_independently_reproduced"]
        )
        self.assertTrue(selected["initialization_state_unresolved"])
        self.assertEqual(
            selected["initialization_state_review_status"],
            "reproduced_but_not_validated_stationary_state",
        )

    def test_better_noninitial_branch_supersedes_initialization(self) -> None:
        rows = pd.DataFrame(
            [
                _row(seed=11, lml=100.0, const_opt=1.0, ls_opt=0.05, attempt=1),
                _row(seed=22, lml=110.0, const_opt=4.0, ls_opt=0.08, attempt=2),
            ]
        )
        selected = audit._select_reviewed_rows(rows).iloc[0]
        self.assertFalse(selected["initialization_lock"])
        self.assertTrue(selected["initialization_state_superseded"])
        self.assertFalse(selected["initialization_state_unresolved"])
        self.assertEqual(
            selected["initialization_state_review_status"],
            "superseded_by_better_actual_branch",
        )

    def test_nondominating_noninitial_branch_remains_unresolved(self) -> None:
        rows = pd.DataFrame(
            [
                _row(seed=11, lml=100.0, const_opt=1.0, ls_opt=0.05, attempt=1),
                _row(
                    seed=22,
                    lml=100.00005,
                    const_opt=4.0,
                    ls_opt=0.08,
                    attempt=2,
                ),
            ]
        )
        selected = audit._select_reviewed_rows(rows).iloc[0]
        self.assertFalse(selected["initialization_lock"])
        self.assertFalse(selected["initialization_state_superseded"])
        self.assertTrue(selected["initialization_state_unresolved"])

    def test_better_exact_warm_state_supersedes_original_lock(self) -> None:
        nominal = _row(
            seed=11,
            lml=100.0,
            const_opt=1.0,
            ls_opt=0.05,
            attempt=1,
        )
        warm = _row(
            seed=22,
            lml=110.0,
            const_opt=4.0,
            ls_opt=0.08,
            attempt=2,
        )
        warm["ls_init"] = 0.08
        warm["repair_kernel_constant_init"] = 4.0
        warm["repair_warm_start"] = True
        rows = pd.DataFrame([nominal, warm])
        selected = audit._select_reviewed_rows(rows).iloc[0]
        self.assertTrue(selected["selected_at_config_initialization_state"])
        self.assertFalse(selected["initialization_lock"])
        self.assertTrue(selected["initialization_state_superseded"])
        self.assertFalse(selected["initialization_state_unresolved"])

    def test_reviewed_collection_pair_is_exact_and_complete(self) -> None:
        frame = pd.DataFrame([_row(
            seed=11,
            lml=100.0,
            const_opt=2.0,
            ls_opt=0.08,
            attempt=1,
        )])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "scan_optimizer_reviewed_actual_rows.csv"
            reviewed = root / "scan_reviewed_rows_complete.csv"
            report_path = root / "scan_reviewed_collection_complete.json"
            summary_path = root / "scan_optimizer_audit_summary.json"
            task_status_path = root / "scan_task_status_complete.csv"
            frame.to_csv(source, index=False)
            report = audit._write_reviewed_collection_report(
                reviewed_rows=frame,
                reviewed_csv=reviewed,
                source_csv=source,
                report_path=report_path,
                summary_path=summary_path,
                task_status_path=task_status_path,
                study_id="unit_test",
                completed_tasks=1,
                incomplete_tasks=0,
            )
            actual_sha = hashlib.sha256(reviewed.read_bytes()).hexdigest()
            on_disk = json.loads(report_path.read_text())
            self.assertEqual(report, on_disk)
            self.assertEqual(on_disk["kind"], "scan")
            self.assertFalse(on_disk["partial"])
            self.assertEqual(on_disk["incomplete_tasks"], 0)
            self.assertEqual(on_disk["output"], str(reviewed.resolve()))
            self.assertEqual(on_disk["output_sha256"], actual_sha)
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                actual_sha,
            )


if __name__ == "__main__":
    unittest.main()
