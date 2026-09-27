"""Check echo-to-scan integrity and render the complete report."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,fitz
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[1];D=B/'derived';Q=B/'qa';E=pd.read_csv(D/'echo_catalogue.csv');C=pd.read_csv(B/'inputs/catalogue.csv');T=pd.read_csv(D/'toy_echo_locations.csv');checks=[]
def check(name,v):
 checks.append({'check':name,'passed':bool(v)});assert v,name
check('exact30flanks',len(E)==30 and E[['scenario','side']].duplicated().sum()==0 and set(E.scenario)==set(C.scenario))
u=E[E.status=='outside_scan'];check('244rightunavailable',len(u)==1 and u.iloc[0].scenario=='one_extra244' and u.iloc[0].side=='right')
for _,e in E.iterrows():
 if e.status=='outside_scan':continue
 p=D/'scans'/e.scenario;bg=pd.read_csv(p/'background_asimov.csv');a=pd.read_csv(p/'matched_asimov.csv');x=a.mass_MeV.to_numpy();R=a.A90.to_numpy()/bg.A90.to_numpy();i=np.flatnonzero(x==e.echo_mass_MeV).item();sid=e.scenario+':'+e.side
 check(sid+':ratio',abs(e.echo_ratio-R[i])<1e-12 and abs(e.echo_depth-(1-R[i]))<1e-12)
 check(sid+':flank',2<=abs(e.offset_sigma)<=8 and e.search_lo_MeV<=e.echo_mass_MeV<=e.search_hi_MeV)
 if e.local_minimum:check(sid+':localminimum',0<i<len(x)-1 and R[i]<=R[i-1] and R[i]<=R[i+1])
 if np.isfinite(e.ratio90_lo_MeV):
  region=(x>=e.ratio90_lo_MeV)&(x<=e.ratio90_hi_MeV);check(sid+':extent',np.all(R[region]<=.9) and e.ratio90_lo_MeV<=e.echo_mass_MeV<=e.ratio90_hi_MeV)
 t=T[(T.scenario==e.scenario)&(T.side==e.side)].sort_values('toy');check(sid+':20locations',list(t.toy)==list(range(20)));vals=[]
 for k in range(20):
  toy=pd.read_csv(p/f'toy_{k:02}.csv');r=toy.A90.to_numpy()/bg.A90.to_numpy();vals.append(r[i]);j=np.flatnonzero(x==t.iloc[k].mass_min_MeV).item();check(sid+f':toy{k}',abs(t.iloc[k].ratio_min-r[j])<1e-12 and abs(t.iloc[k].ratio_at_Asimov_echo-r[i])<1e-12)
 check(sid+':fixedquantiles',np.allclose(np.quantile(vals,[.16,.5,.84]),[e.toy_ratio_fixed_q16,e.toy_ratio_fixed_median,e.toy_ratio_fixed_q84],atol=1e-12))
(Q/'echo_validation.json').write_text(json.dumps(dict(passed=True,checks_total=len(checks),checks=checks),indent=2)+'\n')
pdf=B/'source/main.pdf';doc=fitz.open(pdf);render=Q/'rendered';render.mkdir(exist_ok=True);bad=[]
for i,p in enumerate(doc):
 p.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(render/f'page_{i+1:02}.png')
 for b in p.get_text('dict')['blocks']:
  for line in b.get('lines',[]):
   for span in line['spans']:
    x0,y0,x1,y1=span['bbox']
    if x0<15 or x1>p.rect.width-15 or y0<12 or y1>p.rect.height-12:bad.append(dict(page=i+1,text=span['text'],bbox=span['bbox']))
for start in range(0,len(doc),6):
 sheet=Image.new('RGB',(1890,1650),'white')
 for j,i in enumerate(range(start,min(start+6,len(doc)))):
  im=Image.open(render/f'page_{i+1:02}.png');im.thumbnail((610,790));tile=Image.new('RGB',(630,825),'#dddddd');tile.paste(im,((630-im.width)//2,25));ImageDraw.Draw(tile).text((10,5),f'Page {i+1}',fill='black');sheet.paste(tile,((j%3)*630,(j//3)*825))
 sheet.save(render/f'contact_{start+1:02}.png')
result=dict(passed=not bad,pages=len(doc),offpage_text=bad,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest());(Q/'pdf_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(echo_checks=len(checks),pages=len(doc),offpage_text=len(bad))))
