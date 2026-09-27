"""Analytic centered-Gaussian controls; N is illustrative, not an HPS trials fit."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
B=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
t=np.linspace(0,5,1000)
fig,ax=plt.subplots(figsize=(8.8,3.1))
for n,color in [(1,'#235b7e'),(10,'#a94135'),(30,'#29745c'),(100,'#70548f')]:
    ax.plot(t,n*norm.pdf(t)*norm.cdf(t)**(n-1),color=color,label=f'N = {n}')
ax.set(xlabel='Maximum positive root T',ylabel='Continuous density',xlim=(0,5))
ax.legend(frameon=False,ncol=4);ax.grid(alpha=.15)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(B/'figures'/f'ideal_maximum.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)
rows=[]
for n in [1,10,30,100]:
    mean=quad(lambda t:1-norm.cdf(t)**n,0,12,epsabs=1e-12)[0]
    rows.append({'independent_coordinates':n,'atom_at_zero':2.**(-n),'mean_max_positive':mean,'median':max(0,float(norm.ppf(.5**(1/n))))})
(B/'results/ideal_null.json').write_text(json.dumps({'description':'Independent standard-normal illustration; not an HPS effective-trials measurement. Positive-part distribution has atom2^-N at0 plus displayed continuous density.','rows':rows},indent=2)+'\n')
print(json.dumps(rows))
