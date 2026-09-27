"""Reproducible screenshot digitization; pixel envelopes are not statistical errors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parents[1]
CAL = {
    'limit': dict(image='apex_limits_pvalues.png', x_pixels=[183,312,440,568,696,824,952],
                  x_values=[160,180,200,220,240,260,280], y_pixels=[74,315,557],
                  y_values=[-5,-6,-7], roi=[150,889,115,607], logarithmic=True),
    'pvalue': dict(image='apex_limits_pvalues.png', x_pixels=[1159,1266,1374,1481,1589,1696,1804],
                   x_values=[160,180,200,220,240,260,280], y_pixels=[121,432],
                   y_values=[0,-1], roi=[1131,1750,123,601], logarithmic=True,
                   excluded_axis_tick_boxes=[[x-2,x+2,596,613] for x in [1159,1266,1374,1481,1589,1696,1804]]),
    'resolution': dict(image='apex_resolution.png', x_pixels=[82,214,346,478,610,742,876,1008],
                       x_values=[120,140,160,180,200,220,240,260],
                       y_pixels=[690,574,456,339,222,104], y_values=[0,.2,.4,.6,.8,1],
                       logarithmic=False),
}

def coeff(c, axis):
    return np.polyfit(c[axis+'_pixels'], c[axis+'_values'], 1)

def trace(name, threshold=150):
    c=CAL[name]; gray=np.asarray(Image.open(B/'inputs'/c['image']).convert('L'))
    xa,xb,ya,yb=c['roi']; rows=[]
    for x in range(xa,xb+1):
        ys=np.flatnonzero(gray[ya:yb+1,x]<threshold)+ya
        for bx0,bx1,by0,by1 in c.get('excluded_axis_tick_boxes',[]):
            if bx0<=x<=bx1:ys=ys[(ys<by0)|(ys>by1)]
        if not len(ys): continue
        center=float(np.median(ys)); xc=coeff(c,'x'); yc=coeff(c,'y')
        conv=lambda y:float(10**np.polyval(yc,y))
        rows.append(dict(x_pixel=x,y_pixel=center,y_top_pixel=int(ys.min()),y_bottom_pixel=int(ys.max()),
                         mass_MeV=float(np.polyval(xc,x)),value=conv(center),
                         pixel_low=conv(ys.max()+2),pixel_high=conv(ys.min()-2),
                         mass_pixel_error_MeV=abs(float(xc[0]))*2,
                         stroke_span_pixels=int(np.ptp(ys)),n_dark_pixels=len(ys)))
    return pd.DataFrame(rows)

def main():
    (B/'derived').mkdir(exist_ok=True); (B/'figures').mkdir(exist_ok=True)
    for name in ['limit','pvalue']:
        d=trace(name); d.to_csv(B/'derived'/f'apex_{name}_digitized.csv',index=False)
        # Deliberately preserve all per-column information; no mass smoothing.
        for t in [120,180]: trace(name,t).to_csv(B/'derived'/f'apex_{name}_threshold_{t}.csv',index=False)
    c=CAL['resolution']; im=np.asarray(Image.open(B/'inputs'/c['image']).convert('RGB'))
    # Black square centres, excluding the red, blue and grey component curves.
    guesses=[(213,93),(279,119),(346,115),(412,119),(478,141),(543,174),
             (610,205),(676,227),(742,242),(809,260),(876,268),(942,270)]
    rows=[]
    for mass,(x,y) in zip(range(140,251,10),guesses):
        sub=im[y-6:y+7,x-6:x+7]
        yy,xx=np.where(np.max(sub,axis=2)<100)
        px=float(np.median(xx)+x-6);py=float(np.median(yy)+y-6)
        v=float(np.polyval(coeff(c,'y'),py))
        rows.append(dict(mass_MeV=mass,x_pixel=px,y_pixel=py,total_axis_value=v,
                         pixel_error=abs(float(coeff(c,'y')[0]))*2,
                         sigma_if_MeV=v,sigma_if_percent_MeV=mass*v/100))
    pd.DataFrame(rows).to_csv(B/'derived/apex_total_resolution.csv',index=False)
    for c in CAL.values():
        for axis in ['x','y']:
            cc=coeff(c,axis); c[axis+'_linear_coefficients']=cc.tolist()
            c[axis+'_max_tick_residual']=float(np.max(abs(np.polyval(cc,c[axis+'_pixels'])-c[axis+'_values'])))
    (B/'provenance/pixel_calibration.json').write_text(json.dumps(CAL,indent=2))
    # Overlay on the exact screenshots is the direct visual QA of digitization.
    fig,axs=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[1.1,1]})
    axs[0].imshow(Image.open(B/'inputs/apex_limits_pvalues.png'))
    for name,col in [('limit','#e05b36'),('pvalue','#009e99')]:
        d=pd.read_csv(B/'derived'/f'apex_{name}_digitized.csv')
        axs[0].plot(d.x_pixel,d.y_pixel,color=col,lw=.65)
    axs[0].set_title('Recovered centre traces over the supplied screenshot',fontsize=12)
    axs[1].imshow(Image.open(B/'inputs/apex_resolution.png'))
    r=pd.DataFrame(rows);axs[1].plot(r.x_pixel,r.y_pixel,'o-',color='#d65e00',lw=1,ms=3)
    axs[1].set_title('Total-resolution square centres; vertical-axis units are cropped',fontsize=12)
    for ax in axs: ax.set_axis_off()
    fig.tight_layout();fig.savefig(B/'figures/digitization_overlay.png',dpi=180);plt.close(fig)
    print('Digitized:',{n:len(pd.read_csv(B/'derived'/f'apex_{n}_digitized.csv')) for n in ['limit','pvalue']})

if __name__=='__main__':main()
