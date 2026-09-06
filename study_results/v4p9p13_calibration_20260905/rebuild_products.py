#!/usr/bin/env python3
"""Rebuild the released collection, figures and note, without numerical fits."""
from pathlib import Path
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    inputs = json.loads((HERE/'collection_inputs.json').read_text())
    if not inputs or len(inputs) != len(set(inputs)) or inputs[0] != 'derived':
        raise RuntimeError('Invalid ordered whole-coordinate replacement list')
    command = [sys.executable, '-B', str(HERE/'collect_results.py')]
    for name in inputs:
        path = (HERE/name).resolve()
        if not path.is_relative_to(HERE) or not (path/'contract.json').is_file():
            raise RuntimeError('Missing/out-of-study frozen input: '+name)
        command.extend(('--input-dir', str(path)))
    subprocess.run(command, cwd=ROOT, check=True)
    for script in ('audit_chunked_results.py', 'make_figures.py',
                   'make_validation_figures.py', 'make_truth_figure.py'):
        subprocess.run([sys.executable, '-B', str(HERE/script)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-B', str(HERE/'build_note.py'), '--reverse-truth-dir',
                    str(HERE/'reverse_truth_71/checkpoint_b593e9414310')], cwd=ROOT, check=True)


if __name__ == '__main__':main()
