import uproot,json,sys,numpy as np,re
path=sys.argv[1];f=uproot.open(path);out=dict(path=path,keys=f.classnames())
t=f['preselection'];out['entries']=t.num_entries;out['branches']=t.keys()
chosen=[k for k in t.keys() if (re.search(r'(invM_|type_|p_|px_|py_|pz_|energy_|pdg_|mass_)$',k) or k in ['weight','ele_p_smear_ratio','pos_p_smear_ratio']) and t[k].num_baskets]
out['preview']={}
for k in chosen:
    try:
        a=t[k].array(entry_stop=8,library='np')
        out['preview'][k]=a.tolist() if a.dtype.kind in 'biufUS' else [v if isinstance(v,str) else type(v).__name__ for v in a]
    except Exception as e:out['preview'][k]=str(e)
out['histograms']={}
for k,cl in f.classnames(cycle=False).items():
    if 'TH1' in cl and ('cutflow' in k or 'invM' in k):
        h=f[k];out['histograms'][k]=dict(values=h.values().tolist(),edges=h.axis().edges().tolist(),labels=h.axis().labels())
print(json.dumps(out,indent=2,default=str))
