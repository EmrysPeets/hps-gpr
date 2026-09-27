from pathlib import Path
import json,re,socket
import uproot,numpy as np
root=Path('/sdf/data/hps/physics2021/preselection/v16');out={'host':socket.gethostname(),'root':str(root),'families':{}}
for family,d in [('tc','ap_signal_prompt_smeared'),('uc','ap_signal_prompt_dis_smeared')]:
 p=root/d;rows=[]
 for x in sorted(p.iterdir()):
  if x.is_dir():
   files=sorted(x.glob('*.root'));rows.append({'path':str(x),'files':[str(f) for f in files],'bytes':sum(f.stat().st_size for f in files)})
 files=[Path(f) for row in rows if 'ap60MeV' in row['path'] for f in row['files']]
 sample=files[0] if files else Path(rows[0]['files'][0]);f=uproot.open(sample);t=f['preselection']
 keys=t.keys();wanted=[k for k in ['vertex./vertex.invM_','vertex./vertex.type_','weight','true_ap./true_ap.mass_','ele_p_smear_ratio','pos_p_smear_ratio'] if k in keys]
 previews={k:t[k].array(entry_stop=6,library='np').tolist() for k in wanted}
 cf={}
 for k in ['vertex_cutflow_h','event_cutflow_h']:
  if k in f:cf[k]={'labels':f[k].axis().labels(),'counts':f[k].values().tolist()}
 out['families'][family]={'directory':str(p),'samples':rows,'schema_file':str(sample),'keys':keys,'tree_key':str(t.object_path),'entries':t.num_entries,'preview':previews,'cutflows':cf}
print(json.dumps(out,indent=2))
