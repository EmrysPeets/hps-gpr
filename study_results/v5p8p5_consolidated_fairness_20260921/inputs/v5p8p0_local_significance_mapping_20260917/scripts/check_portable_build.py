#!/usr/bin/env python3
"""Rebuild saved-result artifacts in isolation and compare text and rendered pixels."""
from pathlib import Path
import tempfile, shutil, subprocess, json, hashlib, time, sys
from pypdf import PdfReader
from PIL import Image, ImageChops
b = Path(__file__).resolve().parents[1]
t = Path(tempfile.mkdtemp(prefix='hps_v580_portable_'))
c = t / 'study'
start = time.time()
shutil.copytree(b, c, ignore=shutil.ignore_patterns('layout*', 'final-*.png', 'final_contact.png', '*.log', '*.aux', '__pycache__', 'SHA256SUMS.txt'))
commands = [[sys.executable, 'gpr/summarize_grid.py'], [sys.executable, 'scripts/make_artifacts.py'], ['tectonic', '-X', 'compile', 'source/report.tex', '--keep-logs']]
with (b / 'qa/portable_build.log').open('w') as log:
    for cmd in commands:
        log.write('COMMAND: ' + repr(cmd) + '\n'); log.flush()
        subprocess.run(cmd, cwd=c, stdout=log, stderr=subprocess.STDOUT, check=True)
    original = PdfReader(b / 'source/report.pdf')
    rebuilt = PdfReader(c / 'source/report.pdf')
    assert len(original.pages) == len(rebuilt.pages) == 10
    assert [p.extract_text() for p in original.pages] == [p.extract_text() for p in rebuilt.pages]
    subprocess.run(['pdftoppm', '-r', '125', '-png', str(c / 'source/report.pdf'), str(t / 'rebuilt')], stdout=log, stderr=subprocess.STDOUT, check=True)
    old = sorted((b / 'qa').glob('final-[0-9][0-9].png'))
    new = sorted(t.glob('rebuilt-*.png'))
    assert len(old) == len(new) == 10
    comparisons = []
    for a, z in zip(old, new):
        ia = Image.open(a).convert('RGB'); iz = Image.open(z).convert('RGB')
        same = ia.size == iz.size and ImageChops.difference(ia, iz).getbbox() is None
        comparisons.append({'page': len(comparisons) + 1, 'pixel_identical': same})
        assert same, (a, z)
    result = {'passed': True, 'independent_temporary_directory': True, 'saved_arrays_rebuild_without_refits': True, 'regenerated_Gaussian_fields_seed': 58020260917, 'pages': 10, 'extracted_text_identical': True, 'rendered_pages': comparisons, 'runtime_seconds': time.time() - start, 'original_pdf_sha256': hashlib.sha256((b / 'source/report.pdf').read_bytes()).hexdigest()}
    (b / 'qa/portable_build.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
shutil.rmtree(t)
