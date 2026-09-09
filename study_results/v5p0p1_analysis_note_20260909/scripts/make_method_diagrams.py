#!/usr/bin/env python3
"""Symbolic method diagrams: no data, fits, or simulated statistical results."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

B = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,
                     'axes.unicode_minus':False})
INK='#24364B'; BLUE='#256FA1'; TEAL='#267D76'; PALE='#EDF4F8'; GOLD='#A66A21'
records=[]

def box(ax,x,y,w,h,title,body,color=BLUE,fs=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.013',
                 lw=1.25,edgecolor=color,facecolor=PALE,transform=ax.transAxes))
    ax.text(x+w/2,y+h*.76,title,ha='center',va='center',weight='bold',color=color,
            fontsize=fs,transform=ax.transAxes)
    ax.text(x+w/2,y+h*.34,body,ha='center',va='center',color=INK,fontsize=fs,
            linespacing=1.55,transform=ax.transAxes)

def arrow(ax,a,b,color=INK,style='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.25,
                               color=color,linestyle=style,transform=ax.transAxes))

def canvas(w=11.7,h=3.3):
    fig,ax=plt.subplots(figsize=(w,h));fig.subplots_adjust(left=.015,right=.985,top=.97,bottom=.03)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');return fig,ax

def save(fig,name,meaning):
    outputs=[]
    for ext in ['pdf','png']:
        p=B/'figures'/f'{name}.{ext}';fig.savefig(p,dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.08)
        outputs.append({'path':str(p.relative_to(B)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(fig);records.append({'name':name,'type':'symbolic explanatory diagram','meaning':meaning,'outputs':outputs})

fig,ax=canvas(h=3.5)
items=[('Train on sidebands',r'Predict $b$ and $C_{\rm eff}$'+'\n'+r'Factor $C_{\rm eff}=LL^T$'),
       ('Test one yield',r'$\lambda=b+L\theta+Aw$'+'\n'+r'Hold $b,L,w$ fixed'),
       ('Profile the background',r'Minimize Poisson NLL'+'\n'+r'plus $\frac{1}{2}\theta^T\theta$'),
       ('Build the profile',r'Record $\ell_p(A)$'+'\n'+r'Fit $\widehat A$ or test $A_{90}$')]
xs=[.014,.268,.522,.776]
for i,(title,body) in enumerate(items):
    box(ax,xs[i],.29,.207,.55,title,body,fs=11.5)
    if i<3:arrow(ax,(xs[i]+.218,.565),(xs[i+1]-.012,.565))
ax.text(.5,.12,r'$L\theta$ moves bins coherently; the Gaussian penalty controls the size of that deformation.',
        ha='center',va='center',color=INK,fontsize=12,transform=ax.transAxes)
save(fig,'v501_profile_background_flow','Conditional nuisance optimization with fixed GP prediction; no fitted results shown.')

fig,ax=canvas(h=3.15)
items=[('Selected mass spectrum','Counts and bin edges\nfor one campaign'),
       ('Narrow-signal fit','Gaussian signal template\n+ profiled GP background'),
       ('Signal yield',r'Extract $\widehat A$ and uncertainty'+'\n'+r'or solve $CL_s(A_{90})=0.10$'),
       ('Coupling conversion',r'Use campaign factor $K_d(m)$'+'\n'+r'$\widehat\epsilon^2=\widehat A/K_d$; $\epsilon^2_{90}=A_{90}/K_d$')]
for i,(title,body) in enumerate(items):
    box(ax,xs[i],.21,.207,.66,title,body,fs=11)
    if i<3:arrow(ax,(xs[i]+.218,.54),(xs[i+1]-.012,.54))
save(fig,'v501_yield_to_coupling_flow','Yield extraction followed by campaign-specific radiative normalization and declared branching convention.')

fig,ax=canvas(h=4.7)
box(ax,.025,.39,.185,.25,'One common coupling',r'$\psi=\epsilon^2$',fs=12)
for y,d in zip([.71,.395,.08],['2015','2016','2021']):
    box(ax,.315,y,.295,.235,f'{d}: its own response',rf'$A_d=K_d^{{\rm eff}}\psi$,  $s_d=A_dw_d$'+'\n'+r'Counts $n_d$; background $b_d+L_d\theta_d$',fs=11.5)
    arrow(ax,(.22,.515),(.303,y+.118));arrow(ax,(.622,y+.118),(.752,.515))
box(ax,.764,.345,.209,.34,'One joint likelihood',r'$\mathcal{L}_{\rm comb}=\prod_d\mathcal{L}_d$'+'\n'+r'Profile each $\theta_d$'+'\n'+'Fit one common $\psi$',color=TEAL,fs=12)
save(fig,'v501_common_coupling_flow','Independent campaign responses and nuisance blocks share one coupling coordinate; only active datasets contribute at a mass.')

fig,ax=canvas(h=4.45)
items=[('Coherent background',r'$B$ and $B+\sqrt{B_i}\,e_i$'+'\n'+'One controlled bin change\nfor each support bin'),
       ('Run the full analysis','Baseline + response scans\n'+r'Obtain offset $a$ and response $D$'+'\n'+'Keep the complete mass grid'),
       ('Build the field covariance',r'$\Gamma=D^TD$'+'\n'+r'$s_m^2=\Gamma_{mm}$'+'\n'+r'$R_{mn}=\Gamma_{mn}/(s_ms_n)$'),
       ('Sample entire fields',r'$z^*\sim N(0,R)$'+'\n'+r'$r^*=a+s\odot z^*$'+'\n'+'Evaluate one maximum per field')]
for i,(title,body) in enumerate(items):
    box(ax,xs[i],.405,.207,.51,title,body,fs=10.6)
    if i<3:arrow(ax,(xs[i]+.218,.66),(xs[i+1]-.012,.66))
box(ax,.17,.035,.66,.24,'Independent validation',
    'Complete Poisson spectra run through the original analysis.\nCheck distributions, correlations and accessible tails.',
    color=TEAL,fs=10.8)
arrow(ax,(.501,.388),(.501,.285),color=TEAL,style='--')
save(fig,'v501_global_gp_flow','Expensive complete scans estimate a covariance once; subsequent correlated-field draws avoid repeating the fits. Direct Poisson scans test the approximation.')

(B/'editorial/method_diagram_provenance.json').write_text(json.dumps({'new_fits':0,'new_toys':0,'figures':records,
 'basis':['source/sections/04_methodology.tex','source/sections/v5_global_significance.tex','editorial/MATHEMATICAL_REVIEW.md'],
 'literature':'Ananiev and Read, JINST18(2023)P05041; Cowan et al., EPJC71(2011)1554; mathematical details are also checked against the archived implementation.'},indent=2)+'\n')
print('Wrote',len(records),'symbolic PDF/PNG diagram pairs.')
