from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml


STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import repair_scan_optimization as repair


class RepairPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = repair.runner.load_spec()
        cls.tasks = repair._task_map()
        cls.manifest = pd.read_csv(repair.REPAIR_MANIFEST_PATH)
        if cls.manifest.empty:
            reviewed = pd.read_csv(
                STUDY_DIR
                / "derived"
                / "scan_optimizer_reviewed_actual_rows.csv"
            )
            row = reviewed.iloc[0]
            cls.manifest = pd.DataFrame(
                [
                    {
                        "truth_model": row["truth_model"],
                        "study_scenario": row["study_scenario"],
                        "background_toy_index": row[
                            "background_toy_index"
                        ],
                        "mass_GeV": row["mass_GeV"],
                        "repair_factor": row[
                            "ls_upper_factor_requested"
                        ],
                        "target_task_id": row["task_id"],
                        "target_attempt_path": row["attempt_path"],
                        "reason": "synthetic_unit_test_target",
                        "warm_start_source_factor": row[
                            "ls_upper_factor_requested"
                        ],
                        "warm_start_source_attempt_path": row[
                            "attempt_path"
                        ],
                        "warm_start_ls_opt": row["ls_opt"],
                        "warm_start_const_opt": row["const_opt"],
                        "warm_start_is_feasible": True,
                        "current_target_lml": row["lml"],
                        "source_lml": row["lml"],
                    }
                ]
            )

    def test_feasible_row_gets_three_salts_plus_warm(self) -> None:
        feasible = self.manifest[
            self.manifest["warm_start_is_feasible"].astype(bool)
        ].head(1)
        plan = repair.build_plan_rows(self.spec, feasible, self.tasks)
        self.assertEqual(len(plan), 4)
        self.assertEqual(
            [row["variant"] for row in plan],
            ["salt_01", "salt_02", "salt_03", "warm"],
        )
        self.assertEqual(
            len({row["planned_optimizer_seed"] for row in plan}), 4
        )
        self.assertTrue(all(row["optimizer_restarts"] == 12 for row in plan))
        self.assertTrue(
            all(row["expected_limit_bands"] is False for row in plan)
        )
        self.assertTrue(all(row["interpolation_used"] is False for row in plan))

    def test_no_feasible_warm_start_gets_exactly_three_salts(self) -> None:
        unavailable = self.manifest.head(1).copy()
        unavailable.loc[:, "warm_start_is_feasible"] = False
        unavailable.loc[:, "warm_start_ls_opt"] = float("nan")
        unavailable.loc[:, "warm_start_const_opt"] = float("nan")
        plan = repair.build_plan_rows(self.spec, unavailable, self.tasks)
        self.assertEqual(
            [row["variant"] for row in plan],
            ["salt_01", "salt_02", "salt_03"],
        )
        self.assertEqual(
            len({row["planned_optimizer_seed"] for row in plan}), 3
        )

    def test_prepare_dry_run_never_calls_fit_executor(self) -> None:
        before = (
            repair.runner._sha256_file(repair.PLAN_PATH)
            if repair.PLAN_PATH.is_file()
            else None
        )
        with mock.patch.object(repair, "_execute_one") as execute:
            report = repair.prepare_plan(write=False)
        execute.assert_not_called()
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["fit_launches"], 0)
        after = (
            repair.runner._sha256_file(repair.PLAN_PATH)
            if repair.PLAN_PATH.is_file()
            else None
        )
        self.assertEqual(before, after)

    def test_warm_config_preserves_restarts_and_disables_bands(self) -> None:
        feasible = self.manifest[
            self.manifest["warm_start_is_feasible"].astype(bool)
        ].head(1)
        plan = repair.build_plan_rows(self.spec, feasible, self.tasks)
        warm = next(row for row in plan if row["warm_start"])
        warm_ls = float(warm["warm_start_ls_opt"])
        bounds = (warm_ls * 0.5, warm_ls * 1.5)
        with tempfile.TemporaryDirectory() as temporary:
            path, constant = repair._write_warm_config(
                warm, Path(temporary), bounds
            )
            with path.open() as stream:
                config = yaml.safe_load(stream)
        self.assertEqual(config["n_restarts"], 12)
        self.assertFalse(config["make_ul_bands"])
        self.assertFalse(config["do_combined_bands"])
        self.assertFalse(config["make_eps2_bands"])
        self.assertEqual(
            config["kernel_ls_init_by_dataset"]["2021"],
            warm["warm_start_ls_opt"],
        )
        self.assertEqual(
            config["kernel_ls_bounds_by_dataset"]["2021"],
            list(bounds),
        )
        self.assertEqual(constant, warm["warm_start_const_opt"])

    def test_non_integer_mev_mass_fails_closed(self) -> None:
        with self.assertRaises(repair.RepairError):
            repair._mass_mev(0.050123)

    def test_round_two_has_distinct_ids_seeds_and_run_root(self) -> None:
        target = self.manifest.head(1)
        round_one = repair.build_plan_rows(
            self.spec, target, self.tasks, repair_round=1
        )
        round_two = repair.build_plan_rows(
            self.spec, target, self.tasks, repair_round=2
        )
        self.assertTrue(
            all(row["repair_round"] == 2 for row in round_two)
        )
        self.assertTrue(
            all("repair_r02__" in row["repair_attempt_id"] for row in round_two)
        )
        self.assertTrue(
            all(
                "round_002" in repair._variant_root(row).parts
                for row in round_two
            )
        )
        self.assertTrue(
            {
                row["planned_optimizer_seed"] for row in round_one
            }.isdisjoint(
                {row["planned_optimizer_seed"] for row in round_two}
            )
        )

    def test_round_two_collection_paths_do_not_overwrite_round_one(self) -> None:
        round_one = repair._collection_paths({"repair_round": 1})
        round_two = repair._collection_paths({"repair_round": 2})
        self.assertTrue(set(round_one).isdisjoint(set(round_two)))
        self.assertTrue(all("round_002" in path.parts for path in round_two))


if __name__ == "__main__":
    unittest.main()
