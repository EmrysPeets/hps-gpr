#!/usr/bin/env python3
"""Assemble the exact 415-state ledger for the v4.9.7 combination.

2015 and 2021 coordinates are accepted only from their SHA-pinned archived
ledgers. The 2016 input must be a future reviewed, no-interpolation ledger
bound to the same frozen support decision supplied on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_2015 = HERE / "inputs" / "archived_2015_source_ledger.csv"
DEFAULT_2021 = HERE / "inputs" / "archived_2021_source_ledger.csv"
EXPECTED_SOURCE_SHA256 = {
    "2015": "a962c01aa030429c04e2cc102253b6b8750eacc3c9e294a7a99f851a9870aea9",
    "2021": "e1b568a841c2dded8048e467f67528a90fd5a26b8542803c0cdb3e60109de447",
}
EXPECTED_GRIDS_MEV = {
    "2015": tuple(range(19, 91)),
    "2016": tuple(range(39, 181)),
    "2021": tuple(range(50, 251)),
}
CORE_COLUMNS = ("dataset", "mass_GeV", "const_opt", "ls_opt", "lml")
REQUIRED_2016_REVIEW_COLUMNS = (
    "interpolated",
    "branch_multiplicity",
    "selected_source",
    "row_source",
    "review_status",
    "selected_support_low_MeV",
    "support_high_MeV",
)


class LedgerError(RuntimeError):
    """Raised when a source cannot enter the combined reviewed ledger."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_has_true(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series.dtype):
        return bool(series.fillna(False).any())
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return bool(normalized.isin({"true", "1", "yes"}).any())


def bool_all_false(series: pd.Series) -> bool:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return bool(normalized.isin({"false", "0", "no"}).all())


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_freeze(
    path: Path,
    low_mev: int,
    high_mev: int,
) -> Dict[str, Any]:
    decision = json.loads(path.read_text(encoding="utf-8"))
    study_spec = HERE / "study_spec.json"
    protocol = HERE / "STUDY_PROTOCOL.md"
    if decision.get("study_id") != HERE.name:
        raise LedgerError("2016 support decision names another study.")
    if decision.get("study_spec_sha256") != sha256(study_spec):
        raise LedgerError("2016 support decision does not match study_spec.json.")
    if decision.get("frozen_protocol_sha256") != sha256(protocol):
        raise LedgerError("2016 support decision does not match STUDY_PROTOCOL.md.")
    if decision.get("status") != "support_edge_frozen":
        raise LedgerError("2016 support has not been frozen.")
    if decision.get("observed_scan_authorized") is not True:
        raise LedgerError("2016 support decision does not authorize observed use.")
    found = (
        int(decision["selected_support_low_MeV"]),
        int(decision["support_high_MeV"]),
    )
    requested = (int(low_mev), int(high_mev))
    if found != requested:
        raise LedgerError(f"CLI support {requested} does not match freeze {found}.")
    if found[0] not in range(28, 34) or found[1] != 210:
        raise LedgerError(f"Frozen support {found} is outside the protocol.")
    if decision.get("data_range_2016") != [found[0] / 1000.0, 0.210]:
        raise LedgerError("2016 support decision has inconsistent data_range_2016.")
    for key in (
        "absolute_upper_limit_used_for_selection",
        "retuning_after_confirmation",
        "holdout_65MeV_used_for_selection",
    ):
        if decision.get(key) is not False:
            raise LedgerError(f"Support decision violates frozen control {key}.")
    return decision


def normalized_dataset(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def validate_grid(frame: pd.DataFrame, dataset: str) -> None:
    if frame.empty:
        raise LedgerError(f"No {dataset} rows found.")
    masses = frame["mass_GeV"].to_numpy(float)
    grid = []
    for value in masses:
        mass_mev = int(round(value * 1000.0))
        if not np.isclose(value, mass_mev / 1000.0, rtol=0.0, atol=1.0e-12):
            raise LedgerError(f"Off-grid {dataset} mass: {value!r}")
        grid.append(mass_mev)
    if tuple(grid) != EXPECTED_GRIDS_MEV[dataset]:
        raise LedgerError(f"{dataset} rows do not match the exact expected grid.")
    if frame.duplicated(["dataset", "mass_GeV"]).any():
        raise LedgerError(f"{dataset} source contains duplicate coordinates.")
    values = frame.loc[:, ["const_opt", "ls_opt", "lml"]].to_numpy(float)
    if not bool(np.isfinite(values).all()):
        raise LedgerError(f"{dataset} source contains non-finite GP coordinates.")


def read_source(path: Path, dataset: str) -> Tuple[pd.DataFrame, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LedgerError(f"Missing {dataset} source: {resolved}")
    digest = sha256(resolved)
    if dataset in EXPECTED_SOURCE_SHA256:
        expected = EXPECTED_SOURCE_SHA256[dataset]
        if digest != expected:
            raise LedgerError(
                f"Unexpected {dataset} source SHA-256: {digest} != {expected}"
            )
    frame = pd.read_csv(resolved)
    missing = sorted(set(CORE_COLUMNS).difference(frame.columns))
    if missing:
        raise LedgerError(f"{dataset} source is missing columns: {missing}")
    frame = frame.copy()
    frame["dataset"] = normalized_dataset(frame["dataset"])
    frame = frame.loc[frame["dataset"] == dataset].copy()
    frame = frame.sort_values("mass_GeV").reset_index(drop=False)
    frame = frame.rename(columns={"index": "source_row_index"})
    if (
        "interpolated" in frame.columns
        and not bool_all_false(frame["interpolated"])
    ):
        raise LedgerError(
            f"{dataset} source interpolation flags are not explicitly all false."
        )
    validate_grid(frame, dataset)
    return frame, digest


def canonical_rows(
    frame: pd.DataFrame,
    dataset: str,
    source_path: Path,
    source_sha256: str,
    support: Tuple[int, int],
) -> pd.DataFrame:
    output = frame.loc[:, list(CORE_COLUMNS)].copy()
    output["interpolated"] = False
    if dataset == "2016":
        output["branch_multiplicity"] = frame["branch_multiplicity"].astype(int)
        output["selected_source"] = frame["selected_source"].astype(str)
        output["row_source"] = frame["row_source"].astype(str)
        output["review_status"] = frame["review_status"].astype(str)
    elif dataset == "2015":
        for column in (
            "branch_multiplicity",
            "selected_source",
            "row_source",
            "review_status",
        ):
            output[column] = frame[column].values
    else:
        output["branch_multiplicity"] = 1
        output["selected_source"] = str(source_path.resolve())
        output["row_source"] = "archived_v4p9p5_final_repaired"
        output["review_status"] = "archived_v4p9p5_final_repaired"
    output["source_path"] = str(source_path.resolve())
    output["source_sha256"] = source_sha256
    output["source_row_index"] = frame["source_row_index"].astype(int)
    output["source_role"] = {
        "2015": "archived_2015",
        "2016": "optimized_2016",
        "2021": "archived_2021",
    }[dataset]
    output["selected_support_low_MeV"] = int(support[0])
    output["support_high_MeV"] = int(support[1])
    return output


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-2016-csv", type=Path, required=True)
    parser.add_argument("--support-freeze-json", type=Path, required=True)
    parser.add_argument("--support-2016-low-mev", type=int, required=True)
    parser.add_argument("--support-2016-high-mev", type=int, required=True)
    parser.add_argument("--source-2015-csv", type=Path, default=DEFAULT_2015)
    parser.add_argument("--source-2021-csv", type=Path, default=DEFAULT_2021)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_csv = args.output_csv.expanduser().resolve()
    provenance_out = args.provenance_out.expanduser().resolve()
    if output_csv.exists() or provenance_out.exists():
        raise SystemExit("Refusing to overwrite an assembled ledger or provenance.")

    freeze_path = args.support_freeze_json.expanduser().resolve()
    if not freeze_path.is_file():
        raise SystemExit(f"Missing support freeze: {freeze_path}")
    decision = load_freeze(
        freeze_path,
        args.support_2016_low_mev,
        args.support_2016_high_mev,
    )
    support = (
        int(args.support_2016_low_mev),
        int(args.support_2016_high_mev),
    )

    sources = {
        "2015": args.source_2015_csv.expanduser().resolve(),
        "2016": args.reviewed_2016_csv.expanduser().resolve(),
        "2021": args.source_2021_csv.expanduser().resolve(),
    }
    frames = {}
    digests = {}
    for dataset, path in sources.items():
        frames[dataset], digests[dataset] = read_source(path, dataset)

    missing_review = sorted(
        set(REQUIRED_2016_REVIEW_COLUMNS).difference(frames["2016"].columns)
    )
    if missing_review:
        raise LedgerError(
            "2016 reviewed source is missing review/support provenance columns: "
            f"{missing_review}"
        )
    if not bool_all_false(frames["2016"]["interpolated"]):
        raise LedgerError(
            "2016 reviewed interpolation flags are not explicitly all false."
        )
    support_pairs = set(
        zip(
            frames["2016"]["selected_support_low_MeV"].astype(int),
            frames["2016"]["support_high_MeV"].astype(int),
        )
    )
    if support_pairs != {support}:
        raise LedgerError(
            f"2016 reviewed source support {sorted(support_pairs)} != {[support]}"
        )
    if bool((frames["2016"]["branch_multiplicity"].astype(int) < 1).any()):
        raise LedgerError("2016 review contains a non-positive branch multiplicity.")
    for column in ("selected_source", "row_source", "review_status"):
        if bool(frames["2016"][column].fillna("").astype(str).str.strip().eq("").any()):
            raise LedgerError(f"2016 review has blank {column} provenance.")
    statuses = frames["2016"]["review_status"].astype(str).str.lower()
    if statuses.str.contains("pending|unresolved|fail", regex=True).any():
        raise LedgerError("2016 review contains a pending or failed state.")
    if (
        "repair_reproduction_pending" in frames["2016"].columns
        and bool_has_true(frames["2016"]["repair_reproduction_pending"])
    ):
        raise LedgerError("2016 review still has pending repair reproduction.")

    pieces = [
        canonical_rows(
            frames[dataset],
            dataset,
            sources[dataset],
            digests[dataset],
            support,
        )
        for dataset in ("2015", "2016", "2021")
    ]
    combined = pd.concat(pieces, ignore_index=True)
    ordering = {"2015": 0, "2016": 1, "2021": 2}
    combined["_dataset_order"] = combined["dataset"].map(ordering)
    combined = combined.sort_values(
        ["mass_GeV", "_dataset_order"]
    ).drop(columns="_dataset_order").reset_index(drop=True)
    if len(combined) != 415:
        raise LedgerError(f"Expected 415 reviewed states, found {len(combined)}.")
    if combined.duplicated(["dataset", "mass_GeV"]).any():
        raise LedgerError("Assembled ledger contains duplicate states.")

    atomic_text(output_csv, combined.to_csv(index=False))
    source_provenance = {
        dataset: {
            "path": str(sources[dataset]),
            "sha256": digests[dataset],
            "rows_selected": int(len(frames[dataset])),
            "mass_low_MeV": EXPECTED_GRIDS_MEV[dataset][0],
            "mass_high_MeV": EXPECTED_GRIDS_MEV[dataset][-1],
        }
        for dataset in ("2015", "2016", "2021")
    }
    provenance = {
        "schema_version": 1,
        "support_freeze": str(freeze_path),
        "support_freeze_sha256": sha256(freeze_path),
        "support_freeze_status": decision["status"],
        "support_2016_low_MeV": support[0],
        "support_2016_high_MeV": support[1],
        "sources": source_provenance,
        "row_counts": {"2015": 72, "2016": 142, "2021": 201, "total": 415},
        "interpolation_permitted": False,
        "output_csv": str(output_csv),
        "output_csv_sha256": sha256(output_csv),
        "assembler": str(Path(__file__).resolve()),
        "assembler_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_text(
        provenance_out,
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    )
    print(f"Wrote {len(combined)} reviewed states to {output_csv}")
    print(f"Wrote {provenance_out}")


if __name__ == "__main__":
    main()
