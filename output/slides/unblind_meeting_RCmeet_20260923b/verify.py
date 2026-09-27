"""Validate the scoped native edit and preserved scientific inputs."""
from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parent
before=json.loads((B/'raw-prewrite.json').read_text())['structuredContent']
after=json.loads((B/('raw-final.json' if (B/'raw-final.json').exists() else 'raw-output.json')).read_text())['structuredContent']
def stable(x):
 if isinstance(x,dict):return {k:stable(v) for k,v in x.items() if k not in {'contentUrl','thumbnailUrl','revisionId'}}
 if isinstance(x,list):return [stable(v) for v in x]
 return x
old_ids=[s['objectId'] for s in before['slides']]
new_ids=[s['objectId'] for s in after['slides']]
assert old_ids==new_ids
changed=[i for i,(a,b) in enumerate(zip(before['slides'],after['slides']),1) if stable(a)!=stable(b)]
external=[15] if (B/'raw-final.json').exists() else []
assert changed==sorted([13,14,29]+external),changed
# Verify every request targets only the three instructed slides. Slide15's
# concurrent prose/footer edits were inspected and are absent from our writes.
targets={}
for i,s in enumerate(before['slides'],1):
 targets[s['objectId']]=i
 for e in s.get('pageElements',[]):targets[e['objectId']]=i
 np=s['slideProperties']['notesPage']
 for e in np.get('pageElements',[]):targets[e['objectId']]=i
 targets[np['notesProperties']['speakerNotesObjectId']]=i
bundle=json.loads((B/'requests.json').read_text())
requests=bundle['requests']+[r for im in bundle['images'] for r in im['requests']]
if (B/'repair_requests.json').exists():requests+=json.loads((B/'repair_requests.json').read_text())
written=set()
for r in requests:
 op=next(iter(r.values()))
 if 'elementProperties' in op:
  targets[op['objectId']]=targets[op['elementProperties']['pageObjectId']]
 oid=op.get('objectId',op.get('imageObjectId'))
 written.add(targets[oid])
assert written=={13,14,29},written
assert stable(before['masters'])==stable(after['masters'])
assert stable(before['layouts'])==stable(after['layouts'])
oid='g409df70e3d8_1_115'
a=next(x for x in before['slides'][28]['pageElements'] if x['objectId']==oid)
b=next(x for x in after['slides'][28]['pageElements'] if x['objectId']==oid)
assert stable(a)==stable(b),'Original limit plot changed'
root=B.parents[2]
protocol=json.loads((B/'science/provenance/protocol.json').read_text())
for file,sha in protocol['input_hashes'].items():assert hashlib.sha256((root/file).read_bytes()).hexdigest()==sha,file
report={'slide_count':len(old_ids),'order_unchanged':True,'slides_changed_by_our_requests':sorted(written),'all_native_differences_since_prewrite':changed,'additional_concurrent_user_changes':external,'masters_and_layouts_unchanged':True,'slide29_plot_and_transform_unchanged':True,'parent_hashes_valid':True,'preserved_concurrent_changes':[6,7]+external,'final_revision':after['revisionId']}
(B/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
