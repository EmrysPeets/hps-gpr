"""Compare stable native Slides properties and confirm the instructed order."""
from pathlib import Path
import json

B = Path(__file__).resolve().parents[1]
before = json.loads((B / 'presentation-before.json').read_text())
raw = json.loads((B / 'raw-output.json').read_text())
after = raw.get('structuredContent', raw)
scope = {9, 11, 13, 14, 21, 22, 26, 27}

def stable(x):
    if isinstance(x, dict):
        if x.get('type') == 'SLIDE_NUMBER':
            x = {**x, 'content': '<automatic slide number>'}
        return {k: stable(v) for k, v in x.items()
                if k not in {'contentUrl', 'thumbnailUrl', 'revisionId'}}
    if isinstance(x, list):
        return [stable(v) for v in x]
    return x

old_ids = [s['objectId'] for s in before['slides']]
expected = old_ids[:14] + ['rc23_validation_overview'] + old_ids[14:23] + ['rc23_validation_summary'] + old_ids[23:]
actual = [s['objectId'] for s in after['slides']]
by_id = {s['objectId']: s for s in after['slides']}
changed = [i for i, s in enumerate(before['slides'], 1)
           if stable(s) != stable(by_id[s['objectId']])]
unexpected = sorted(set(changed) - scope)
report = {
    'original_slide_count': len(old_ids), 'final_slide_count': len(actual),
    'expected_order_matches': expected == actual,
    'original_slides_changed': changed,
    'unexpected_original_slides_changed': unexpected,
    'untouched_original_slides': len(old_ids) - len(changed),
    'masters_unchanged': stable(before['masters']) == stable(after['masters']),
    'layouts_unchanged': stable(before['layouts']) == stable(after['layouts']),
    'normalization': 'Ignore expiring content/thumbnail URLs, revision IDs and automatic slide-number values only.',
    'mapping': [{'original': i, 'final': actual.index(s) + 1, 'slide_id': s}
                for i, s in enumerate(old_ids, 1)]
}
(B / 'qa' / 'scope-verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'mapping'}, indent=2))
assert report['expected_order_matches'] and not unexpected
assert report['masters_unchanged'] and report['layouts_unchanged']
