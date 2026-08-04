import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


STUDY_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = STUDY_DIR / "run_ensemble.py"
SPEC = importlib.util.spec_from_file_location("v4p1_run_ensemble", MODULE_PATH)
driver = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(driver)


class EnsembleDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = driver.load_spec()

    def test_default_grid_is_predeclared_11_points(self):
        masses = driver.mass_grid(self.spec, None, None, None)
        self.assertEqual(len(masses), 11)
        self.assertAlmostEqual(masses[0], 0.050)
        self.assertAlmostEqual(masses[-1], 0.250)
        self.assertTrue(
            np.allclose(np.diff(masses), np.full(10, 0.020), rtol=0, atol=1e-12)
        )

    def test_base_config_is_exact_reviewed_2021_k15_card(self):
        entry = self.spec["base_config"]
        self.assertEqual(
            entry["path_from_repo"],
            "study_configs/finalist_k15_2021_10pct_combined100toy_20260803/"
            "config_obsUL90_2021_10pct_fit040_300_k15_observed_only.yaml",
        )
        self.assertEqual(
            entry["sha256"],
            "3a4120ab520cca3352d281e06d4d0c5e4c05c83cec97c319b1aabe19e9c0b3f2",
        )

    def test_task_counts_cover_both_truth_models(self):
        tasks = driver.build_tasks(self.spec)
        scans = [task for task in tasks if task["kind"] == "scan"]
        injections = [task for task in tasks if task["kind"] == "injection"]
        expected = 6 * 2 * 5 * 10
        self.assertEqual(len(scans), expected)
        self.assertEqual(len(injections), expected)
        self.assertEqual(
            {task["truth_model"] for task in scans},
            {"gengamma", "sigpowexpq"},
        )

    def test_factor15_absolute_injection_protocol_is_frozen(self):
        closure = driver._injection_protocol_spec(self.spec)
        self.assertEqual(
            closure["protocol"], "factor15_prefit_asimov_absolute_v1"
        )
        self.assertEqual(closure["anchor_factor"], 15)
        self.assertEqual(closure["anchor_sigma_a_source"], "asimov")
        self.assertEqual(
            closure["anchor_sigma_a_ref_mode"], "prefit_asimov"
        )
        self.assertTrue(closure["fixed_absolute_amplitudes_across_factors"])
        identities = driver.build_anchor_identities(self.spec)
        self.assertEqual(len(identities), 2 * 5 * 10)
        self.assertEqual(len({row["anchor_id"] for row in identities}), 100)
        frozen = self.spec["frozen_generated_config_sha256_by_factor"]
        self.assertEqual(
            set(frozen), {"6", "9", "12", "15", "20", "25"}
        )
        self.assertEqual(
            frozen["15"],
            "71a6f45574eae96fb89c188606a27beb855b7a0dd48aa2210d9ceb7324fdce16",
        )

    def test_nested_poisson_draw_is_deterministic_and_nonnegative(self):
        mean = np.array([0.2, 1.5, 4.0, 10.0])
        arrays_a, meta_a = driver.draw_nested_family(
            self.spec, mean, "gengamma", "one_pct", 3
        )
        arrays_b, meta_b = driver.draw_nested_family(
            self.spec, mean, "gengamma", "one_pct", 3
        )
        for scenario in arrays_a:
            np.testing.assert_array_equal(arrays_a[scenario], arrays_b[scenario])
            self.assertEqual(meta_a[scenario], meta_b[scenario])
        self.assertTrue(
            np.all(arrays_a["2021_1pct_x10"] >= arrays_a["2021_1pct"])
        )
        self.assertTrue(
            np.all(
                arrays_a["2021_1pct_x100"] >= arrays_a["2021_1pct_x10"]
            )
        )

    def test_realized_histogram_is_never_multiplied(self):
        mean = np.array([3.0, 7.0, 11.0, 19.0])
        arrays, _ = driver.draw_nested_family(
            self.spec, mean, "gengamma", "ten_pct", 4
        )
        # Exact equality would be possible in principle, but is vanishingly
        # unlikely for this deterministic test vector and seed. This protects
        # against accidentally replacing independent increments with 10*N.
        self.assertFalse(
            np.array_equal(
                arrays["2021_10pct_x10"], 10 * arrays["2021_10pct"]
            )
        )

    def test_optimizer_pairing_is_factor_independent(self):
        seed_a = driver._mass_seed(
            self.spec, "gengamma", "2021_1pct_x10", 2, 0.120
        )
        seed_b = driver._mass_seed(
            self.spec, "gengamma", "2021_1pct_x10", 2, 0.120
        )
        self.assertEqual(seed_a, seed_b)

    def test_config_switches_every_limit_band_off(self):
        base = {
            "kernel_ls_res_upper_factor_by_dataset": {"2021": 15.0},
            "make_ul_bands": True,
            "ul_bands_toys": 100,
            "do_combined_bands": True,
            "combined_bands_n_toys": 100,
            "make_eps2_bands": True,
        }
        cfg = driver._config_overrides(self.spec, base, 20)
        self.assertEqual(
            cfg["kernel_ls_res_upper_factor_by_dataset"]["2021"], 20.0
        )
        self.assertFalse(cfg["make_ul_bands"])
        self.assertEqual(cfg["ul_bands_toys"], 0)
        self.assertFalse(cfg["do_combined_bands"])
        self.assertEqual(cfg["combined_bands_n_toys"], 0)
        self.assertFalse(cfg["make_eps2_bands"])
        self.assertEqual(cfg["inj_background_mode"], "fixed_hist")
        self.assertEqual(cfg["inj_mode"], "poisson")

    def test_fixed_anchor_annotation_preserves_absolute_amplitude_and_hash(self):
        draw_hash = "a" * 64
        anchor = {
            "injection_anchor_factor": 15,
            "injection_anchor_sigmaA_ref": 2.5,
            "strength_points": [
                {
                    "injection_anchor_nsigma": 3.0,
                    "injection_anchor_strength": 7.5,
                    "signal_Nsig_full": 9,
                    "signal_Nsig_win": 6,
                    "signal_Nsig_train": 2,
                    "signal_draw_sha256": draw_hash,
                }
            ],
        }
        rows = [
            {
                "strength": "7.5",
                "Nsig_win": "6",
                "Nsig_train": "2",
                "sigmaA_ref": "3.0",
                "sigmaA_ref_mode": "prefit_asimov",
            }
        ]
        captured = [
            {
                "strength": 7.5,
                "signal_Nsig_full": 9,
                "signal_draw_sha256": draw_hash,
            }
        ]
        factor6 = driver._annotate_fixed_anchor_rows(
            rows=rows,
            captured=captured,
            anchor_entry=anchor,
            anchor_ledger_sha256="b" * 64,
            candidate_factor=6,
        )[0]
        factor25 = driver._annotate_fixed_anchor_rows(
            rows=rows,
            captured=captured,
            anchor_entry=anchor,
            anchor_ledger_sha256="b" * 64,
            candidate_factor=25,
        )[0]
        for required in (
            "injection_anchor_factor",
            "injection_anchor_nsigma",
            "injection_anchor_strength",
            "injection_anchor_sigmaA_ref",
            "injection_anchor_ledger_sha256",
            "injection_protocol",
        ):
            self.assertIn(required, factor6)
        self.assertEqual(
            factor6["injection_protocol"],
            "factor15_prefit_asimov_absolute_v1",
        )
        self.assertEqual(
            factor6["injection_anchor_strength"],
            factor25["injection_anchor_strength"],
        )
        self.assertEqual(
            factor6["signal_draw_sha256"],
            factor25["signal_draw_sha256"],
        )
        self.assertTrue(factor6["signal_draw_hash_verified"])

    def test_fixed_anchor_annotation_fails_on_signal_draw_drift(self):
        anchor = {
            "injection_anchor_factor": 15,
            "injection_anchor_sigmaA_ref": 1.0,
            "strength_points": [
                {
                    "injection_anchor_nsigma": 1.0,
                    "injection_anchor_strength": 1.0,
                    "signal_Nsig_full": 1,
                    "signal_Nsig_win": 1,
                    "signal_Nsig_train": 0,
                    "signal_draw_sha256": "c" * 64,
                }
            ],
        }
        with self.assertRaisesRegex(
            driver.StudyError, "Signal realization differs"
        ):
            driver._annotate_fixed_anchor_rows(
                rows=[
                    {
                        "strength": "1",
                        "Nsig_win": "1",
                        "Nsig_train": "0",
                        "sigmaA_ref": "1.5",
                        "sigmaA_ref_mode": "prefit_asimov",
                    }
                ],
                captured=[
                    {
                        "strength": 1.0,
                        "signal_Nsig_full": 1,
                        "signal_draw_sha256": "d" * 64,
                    }
                ],
                anchor_entry=anchor,
                anchor_ledger_sha256="e" * 64,
                candidate_factor=6,
            )

    def test_anchor_preparation_is_dry_run_by_default(self):
        with mock.patch.object(
            driver, "_anchor_part_is_valid", return_value=False
        ), mock.patch.object(driver.subprocess, "run") as run:
            report = driver.prepare_injection_anchors(
                self.spec,
                execute=False,
                max_parts=2,
                workers=4,
                force=False,
            )
        run.assert_not_called()
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["parts_selected"], 2)
        self.assertFalse(report["would_write_shared_configs"])
        self.assertFalse(report["would_write_task_manifest"])
        self.assertTrue(
            all("run-anchor-part" in command for command in report["commands"])
        )

    def test_signal_capture_hashes_exact_realized_array(self):
        class FakeInjectionModule:
            @staticmethod
            def _inject_counts_from_template(template, strength, rng, mode):
                signal = np.asarray([0, 2, 1], dtype=int)
                return signal, 3, 1.0

        module = FakeInjectionModule()
        original = module._inject_counts_from_template

        def callback():
            module._inject_counts_from_template(
                np.ones(3), 4.0, np.random.default_rng(8), "poisson"
            )

        captured = driver._run_with_signal_draw_capture(module, callback)
        self.assertIs(module._inject_counts_from_template, original)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["signal_Nsig_full"], 3)
        self.assertEqual(
            captured[0]["signal_draw_sha256"],
            driver._sha256_array_int64(np.asarray([0, 2, 1])),
        )

    def test_collection_verifies_cross_factor_signal_identity(self):
        base = {
            "truth_model": "gengamma",
            "study_scenario": "2021_1pct_x10",
            "background_toy_index": "2",
            "mass_GeV": "0.12",
            "toy": "0",
            "injection_protocol": "factor15_prefit_asimov_absolute_v1",
            "injection_anchor_factor": "15",
            "injection_anchor_nsigma": "3",
            "injection_anchor_strength": "12.5",
            "injection_anchor_sigmaA_ref": "4.166666666666667",
            "injection_anchor_ledger_sha256": "f" * 64,
            "signal_draw_sha256": "a" * 64,
            "signal_draw_hash_verified": "True",
            "Nsig_win": "8",
            "Nsig_train": "3",
            "signal_Nsig_full": "13",
        }
        rows = [
            {**base, "ls_upper_factor_requested": "6"},
            {**base, "ls_upper_factor_requested": "25"},
        ]
        report = driver.validate_collected_injection_pairing(rows)
        self.assertEqual(report["groups_validated"], 1)
        self.assertEqual(report["minimum_factors_per_group"], 2)
        self.assertTrue(
            report["signal_draw_hash_and_Nsig_identical_within_group"]
        )
        rows[1]["Nsig_win"] = "9"
        with self.assertRaisesRegex(
            driver.StudyError, "Cross-factor injection mismatch"
        ):
            driver.validate_collected_injection_pairing(rows)

    def test_run_pending_subprocess_has_one_run_task_token(self):
        task = {
            "task_id": "scan__f06__gengamma__unit_scenario__t9999",
            "kind": "scan",
            "factor": 6,
            "truth_model": "gengamma",
            "function_tag": "fGenGammaThresh",
            "scenario": "unit_scenario",
            "source_family": "one_pct",
            "exposure_multiplier": 1,
            "toy_index": 9999,
            "toy_root": "unused.root",
            "toy_container": "unused",
            "toy_name": "toy_9999",
            "config": "unused.yaml",
        }
        completed = mock.Mock(returncode=0)
        with mock.patch.object(driver.subprocess, "run", return_value=completed) as run:
            result = driver.run_pending(
                self.spec,
                [task],
                [0.05],
                "scan",
                max_tasks=1,
                workers=1,
                execute=True,
                factors=[],
                truths=[],
                scenarios=[],
            )
        self.assertEqual(result[0]["returncode"], 0)
        command = run.call_args.args[0]
        self.assertEqual(command.count("run-task"), 1)
        self.assertEqual(command[2], "run-task")
        self.assertEqual(command[3], task["task_id"])
        self.assertNotIn("--mass-step-mev", command)

    def test_run_pending_fails_closed_on_parallel_child_failure(self):
        tasks = []
        for index in (9997, 9998):
            tasks.append(
                {
                    "task_id": f"scan__f06__gengamma__failure_{index}__t{index}",
                    "kind": "scan",
                    "factor": 6,
                    "truth_model": "gengamma",
                    "function_tag": "fGenGammaThresh",
                    "scenario": f"failure_{index}",
                    "source_family": "one_pct",
                    "exposure_multiplier": 1,
                    "toy_index": index,
                    "toy_root": "unused.root",
                    "toy_container": "unused",
                    "toy_name": f"toy_{index}",
                    "config": "unused.yaml",
                }
            )
        completed = [mock.Mock(returncode=0), mock.Mock(returncode=7)]
        with mock.patch.object(driver.subprocess, "run", side_effect=completed):
            with self.assertRaisesRegex(
                driver.StudyError, "run-pending child task failure"
            ):
                driver.run_pending(
                    self.spec,
                    tasks,
                    [0.05, 0.07],
                    "scan",
                    max_tasks=2,
                    workers=2,
                    execute=True,
                    factors=[],
                    truths=[],
                    scenarios=[],
                )


if __name__ == "__main__":
    unittest.main()
