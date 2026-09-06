import importlib.util
import json
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parents[1]
SCRIPT = STUDY_DIR / "make_final_length_scale_bound_interpretation.py"
SPEC = importlib.util.spec_from_file_location("final_ls_interpretation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_final_interpretation_is_deterministic_and_scope_safe():
    first, first_markdown = MODULE.build()
    second, second_markdown = MODULE.build()

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first_markdown == second_markdown
    assert first["definitions"]["bound_fraction"] == 0.999
    assert first["decision"]["provisional_projected_factor"] == 20
    assert first["decision"]["universal_common_factor"] == 25
    assert first["decision"]["factor20_projected_scan_at_bound_rows"] == 0
    assert first["decision"]["factor20_projected_injection_refit_at_bound_rows"] == 0
    assert first["decision"]["factor20_all_scenarios_scan_at_bound_rows"] == 42
    assert first["decision"]["factor25_all_scenarios_scan_at_bound_rows"] == 0
    assert first["decision"]["factor25_all_scenarios_scan_near_bound_rows"] == 0
    assert len(first["projected_100pct_rows"]) == 12
    aggregate = first["projected_100pct_aggregate"]
    assert 2.03 < aggregate["paired_response_deficit_percent_min"] < 2.05
    assert 2.80 < aggregate["paired_response_deficit_percent_max"] < 2.82
    assert first["unpaired_factor20_comparison"]["source_families_paired"] is False
    assert first["qmu_exclusion"]["included_in_bound_or_response_interpretation"] is False
    assert first["caveats"]["expected_limit_bands"] is False
    assert first["caveats"]["coverage_qualified"] is False

    source = SCRIPT.read_text(encoding="utf-8")
    assert "hps_gpr" not in source
    assert "subprocess" not in source
