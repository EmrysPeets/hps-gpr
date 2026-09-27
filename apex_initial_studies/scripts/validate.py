"""Numerical, provenance and PDF checks for the standalone comparison."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.stats import norm
import fitz

B=Path(__file__).resolve().parents[1]; checks=[]
def check(name,value,detail=None):
    checks.append(dict(check=name,passed=bool(value),detail=detail))
def read(name):return pd.read_csv(B/name)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

for entry in json.loads((B/'provenance/hps_inputs.json').read_text()):
    check('snapshot '+entry['snapshot'],digest(B/entry['snapshot'])==entry['sha256'])
    p=Path(entry['source'])
    if p.exists():check('parent preserved '+p.name,digest(p)==entry['sha256'])
for e in json.loads((B/'provenance/user_inputs.json').read_text()):
    check('screenshot '+Path(e['copy']).name,digest(B/'inputs'/Path(e['copy']).name)==e['sha256'])

a=read('derived/apex_limit_digitized.csv');p=read('derived/apex_pvalue_digitized.csv')
r=read('derived/apex_total_resolution.csv');c=read('derived/common_mass_scan.csv')
for name,d in [('APEX limits',a),('APEX p',p)]:
    check(name+' unique increasing mass',len(d.mass_MeV.unique())==len(d) and np.all(np.diff(d.mass_MeV)>0))
    check(name+' finite positive values',np.isfinite(d.value).all() and (d.value>0).all())
    check(name+' centre inside stroke envelope',((d.pixel_low<=d.value)&(d.value<=d.pixel_high)).all())
check('620 p columns / 740 limit columns',len(a)==740 and len(p)==620)
check('p values within (0,1]',(p.value<=1).all())
check('all 12 total squares',r.mass_MeV.tolist()==list(range(140,251,10)))
check('total trace visual landmarks',abs(r.iloc[0].total_axis_value-1.02)<.01 and abs(r.iloc[-1].total_axis_value-.72)<.01)
for mass,lo,hi in [(158.5,1.5e-6,2.1e-6),(212,1.2e-7,2.0e-7),(269,3.2e-6,4.7e-6)]:
    v=np.exp(np.interp(mass,a.mass_MeV,np.log(a.value)));check('manual contour landmark '+str(mass),lo<v<hi,float(v))
for mass,lo,hi in [(229.25,.025,.05),(214.37,.035,.075),(160,.28,.55)]:
    v=np.exp(np.interp(mass,p.mass_MeV,np.log(p.value)));check('manual p landmark '+str(mass),lo<v<hi,float(v))
check('no unsupported common masses',c.mass_MeV.tolist()==list(range(155,251)))
check('2016+2021 through 180',set(c[c.mass_MeV<=180].datasets)=={'2016+2021'})
check('2021 only above 180',set(c[c.mass_MeV>180].datasets)=={'2021'})
check('HPS p and Z conventions',np.allclose(norm.sf(c.hps_Z),c.hps_p,rtol=1e-10,atol=1e-12))
check('native tenfold exposure scaling',np.allclose(c.hps_2021_full_observed_equivalent*np.sqrt(10),c.hps_2021_limit,rtol=1e-13))
check('current contour above APEX including pixel envelope',(c.hps_limit>c.apex_local_pixel_envelope_high).all())
check('no shared p below 0.1',not ((c.hps_p<.1)&(c.apex_p<.1)).any())
check('no shared p below 0.1 with low pixel bound',not ((c.hps_p<.1)&(c.apex_p_pixel_low<.1)).any())
check('finite-grid bound uninformative',(c.p_concordance_bonferroni==1).all())
check('projected competitive nodes',c.loc[c.hps_2021_full_observed_equivalent<c.apex_limit,'mass_MeV'].tolist()==[155,158,159,166,169,170,174])
check('pixel-stable projected nodes',c.loc[c.hps_2021_full_observed_equivalent<c.apex_local_pixel_envelope_low,'mass_MeV'].tolist()==[158,166,170])
q=read('derived/projected_signal_compatibility.csv')
check('all four saved injection comparisons',set(q.scenario)=={'ten_160','ten_210','one_210','one_extra244'})
for _,row in q.iterrows():
    catalog=read('inputs/hps_projection_catalogue.csv');cr=catalog[catalog.scenario==row.scenario].iloc[0]
    bg=read('inputs/'+row.scenario+'_background_asimov.csv').set_index('mass_MeV').loc[row.mass_MeV]
    epsilon=cr.injected_yield/bg.K_counts_per_epsilon2*bg.branching_factor
    check('injected rate conversion '+row.scenario,np.isclose(epsilon,row.injected_epsilon2,rtol=1e-12))
check('native injection ratios material',2.3<q.iloc[0].injection_over_apex_limit<2.6 and 18<q.iloc[1].injection_over_apex_limit<21)
check('historical source kept distinct',set(q.lane)=={'ten','one'})

pdf=B/'output/pdf/APEX_Initial_Studies.pdf'
if not pdf.exists():pdf=B/'output/pdf/main.pdf'
doc=fitz.open(pdf);content='\n'.join(pg.get_text() for pg in doc)
import unicodedata
search_text=' '.join(unicodedata.normalize('NFKC',content).split()).lower()
check('no orphan spill pages',len(doc)==8,len(doc))
for phrase in ['confidence level','have not been verified','not a statistical uncertainty','no toys','155','229','same-mass','Bonferroni']:
    check('document qualification '+phrase,phrase.lower() in search_text)
check('no unresolved references','[?]' not in content and '??' not in content)
for i,pg in enumerate(doc):
    outside=[]
    for block in pg.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<12 or y0<5 or x1>pg.rect.width-12 or y1>pg.rect.height-5:outside.append(span['text'])
    check('page geometry '+str(i+1),not outside,outside)
    pg.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(B/'qa/rendered'/f'page-{i+1:02d}.png')
log=(B/'output/pdf/main.log').read_text()
check('no overfull TeX boxes','Overfull' not in log)
for path in (B/'source').glob('*.tex'):
    check('no control characters '+path.name,b'\r' not in path.read_bytes())
(B/'qa/extracted_text.txt').write_text(content)
out=dict(status='passed' if all(x['passed'] for x in checks) else 'failed',checks=len(checks),
         passed=sum(x['passed'] for x in checks),pdf_sha256=digest(pdf),pages=len(doc),results=checks)
(B/'qa/validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='results'},indent=2))
for row in checks:
    if not row['passed']:print('FAIL',row)
sys.exit(0 if out['status']=='passed' else 1)
