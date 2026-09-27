from pathlib import Path
import os, json
os.environ['MPLCONFIGDIR']='/tmp/hps-rc-slides-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).parent/'assets'
def make(name,lines,size):
 fig=plt.figure(figsize=size,facecolor='white')
 for text,y,fs in lines:fig.text(.01,y,text,ha='left',va='center',fontsize=fs,color='#172b4d')
 for ext in ['png','pdf','svg']:fig.savefig(O/(name+'.'+ext),dpi=320,bbox_inches='tight',pad_inches=.04,facecolor='white')
 plt.close(fig)
make('equation18_pull',[(r'$p_t(z)=\frac{\widehat A_t(z)-A_{{\rm inj},t}(z)}{\sigma_{A,t}(z)}$',.5,28)],(7,1))
make('equation22_injection',[(r'$A_{\rm inj}(z)=z\,\sigma_{A,\rm ref},\quad z\in\{0,1,3,5\}$',.5,26)],(8,.72))
make('equation49_response',[(r'$r^*\simeq a+D^T\xi,\quad\xi\sim N(0,I)$',.72,24),(r'$\Sigma_r=D^TD,\quad R_{ij}=\frac{\Sigma_{r,ij}}{s_i s_j}$',.18,24)],(8,1.6))
make('equation50_mapping',[(r'$Z_{\rm local}=\max(r_{\rm obs},0)$',.81,24),(r'$T=\max_m\max(r(m),0)$',.43,24),(r'$p_{\rm global}=P_B(T^*\geq T_{\rm obs})$',.04,24)],(8,1.85))
print('4 equation assets created')
