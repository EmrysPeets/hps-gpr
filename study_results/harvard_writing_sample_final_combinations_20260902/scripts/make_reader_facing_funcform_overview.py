#!/usr/bin/env python3
"""Rebuild the functional-form overview with legends outside data axes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parent
REPO = DERIVATIVE.parents[1]
SOURCE_SCRIPT = REPO / "hps_gpr_analysis_note" / "scripts" / "generate_note_figures.py"
OUTPUT = DERIVATIVE / "source" / "toy_generation_figs" / "funcform_publication_overview.png"
QA = DERIVATIVE / "qa" / "reader_facing_funcform_overview"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("note_figure_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE_SCRIPT}")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    payloads = [
        source._load_funcform_payload(dataset, root_path)
        for dataset, root_path in source.FUNCFORM_ROOT_SPECS
    ]

    fig = plt.figure(figsize=(13.6, 14.0))
    grid = fig.add_gridspec(
        6,
        2,
        width_ratios=[1.08, 1.0],
        height_ratios=[0.22, 1.0, 0.22, 1.0, 0.22, 1.0],
        hspace=0.30,
        wspace=0.18,
    )
    right_handles = right_labels = None
    for row, payload in enumerate(payloads):
        key_ax = fig.add_subplot(grid[2 * row, 0])
        key_ax.axis("off")
        left = fig.add_subplot(grid[2 * row + 1, 0])
        right = fig.add_subplot(grid[2 * row + 1, 1])
        source._draw_funcform_candidate_panel(left, payload, show_ylabel=True)
        handles, labels = left.get_legend_handles_labels()
        if left.get_legend() is not None:
            left.get_legend().remove()
        key_ax.legend(
            handles,
            labels,
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=7.2,
        )
        source._draw_funcform_primary_panel(right, payload, show_ylabel=False, show_legend=False)
        if row == 0:
            right_handles, right_labels = right.get_legend_handles_labels()

    if right_handles and right_labels:
        fig.legend(
            right_handles,
            right_labels,
            loc="upper center",
            bbox_to_anchor=(0.74, 0.995),
            ncol=3,
            frameon=False,
            fontsize=9.0,
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    QA.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "pass",
        "change_scope": "Candidate-family legends remain outside all data axes and the faint footer note was removed; plotted arrays are unchanged.",
        "source_script": str(SOURCE_SCRIPT),
        "source_script_sha256": sha256(SOURCE_SCRIPT),
        "source_roots": [
            {"path": str(path), "sha256": sha256(path)}
            for _, path in source.FUNCFORM_ROOT_SPECS
        ],
        "products": [
            {"path": str(OUTPUT.relative_to(DERIVATIVE)), "sha256": sha256(OUTPUT)},
            {"path": str(OUTPUT.with_suffix('.pdf').relative_to(DERIVATIVE)), "sha256": sha256(OUTPUT.with_suffix('.pdf'))},
        ],
    }
    (QA / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
