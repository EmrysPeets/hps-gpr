from common import *
import fitz,re
from scipy.integrate import quad
from scipy.special import roots_legendre
q=pd.read_csv(B/'derived/projected_contours.csv',float_precision='round_trip');checks={}
def check(k,v):
 checks[k]=bool(v)
 if not v:raise RuntimeError(k)
check('complete_19_250_grid',q.mass_MeV.tolist()==list(range(19,251)))
check('finite_positive_contours',np.isfinite(q[['projected_three_minimal','projected_four_minimal']]).all().all() and (q[['projected_three_minimal','projected_four_minimal']]>0).all().all())
check('identical_below_2019_support',np.array_equal(q[q.mass_MeV<75].projected_three_minimal,q[q.mass_MeV<75].projected_four_minimal))
check('2015_active_through_100',q.loc[q.mass_MeV<=100,'datasets_three'].str.contains('2015').all() and not q.loc[q.mass_MeV>100,'datasets_three'].str.contains('2015').any())
check('2019_active_75_250',q.loc[q.mass_MeV>=75,'datasets_four'].str.contains('2019').all() and not q.loc[q.mass_MeV<75,'datasets_four'].str.contains('2019').any())
check('current_three_cls_roots',abs(q.three_cls-.1).max()<2e-6)
check('current_four_cls_roots',abs(q[q.mass_MeV>=75].cls-.1).max()<2e-6)
check('profile_scores',max(q.three_max_score.max(),q.max_score.max())<2e-7)
a=q.density_2015;b=q.density_2016;c=q.density_2021;d=q.density_2019
check('three_density_scale',np.allclose(q.scale_three,np.sqrt((a+b+c)/(a+b+10*c)),rtol=2e-15))
check('four_density_scale',np.allclose(q.scale_four,np.sqrt((a+b+c+d)/(a+b+10*c+100*d)),rtol=2e-15))
check('three_projection_formula',np.allclose(q.projected_three_minimal,q.observed_three_minimal*q.scale_three,rtol=2e-15))
check('four_projection_formula',np.allclose(q.projected_four_minimal,q.observed_four_minimal*q.scale_four,rtol=2e-15))
check('full2021_only_scale',np.allclose(q.loc[q.mass_MeV>180,'scale_three'],np.sqrt(.1),rtol=2e-15))
check('single_branching_correction_three',np.allclose(q.observed_three_minimal,q.observed_three_ee*q.dimuon_factor,rtol=2e-15))
check('single_branching_correction_four',np.allclose(q.observed_four_minimal,q.observed_four_ee*q.dimuon_factor,rtol=2e-15))
check('2019_root_identity',sha(B/'inputs/hps_2019_1pct_invariant_mass.root')==json.loads((B/'inputs/2019_resolved_card.json').read_text())['input_sha256'])
sc=pd.read_csv(B/'inputs/2019_kernel_scan.csv',float_precision='round_trip')
check('2019_kernel_ceiling_disclosed',np.allclose(sc.ls_opt_over_ls_hi,1.,atol=1e-8))
# The full-covariance comparison checks that the truncated representation is harmless.
y='2019';DATA[y]=dict(np.load(B/'inputs/spectrum_2019.npz'));DATA[y]['idx']={int(m):i for i,m in enumerate(DATA[y]['masses'])};LIMITS[y]=(75.,250.)
full_cov=[]
for m in [92,170,230]:
 row=q[q.mass_MeV==m].iloc[0];ps=[moving_context(y,m) for y in row.datasets_four.split('+')]
 for p in ps:p['L'],_=factor_cov(p['C'],p['b'],full=True)
 result=OneSignalProfile(np.concatenate([p['b'] for p in ps]),block_diag(*[p['L'] for p in ps]),np.concatenate([p['S'][:,0] for p in ps])).limit(np.concatenate([p['n'] for p in ps]))
 error=result['A90']*1e-8/row.observed_four_ee-1
 check(f'full_covariance_{m}',abs(error)<2e-5);full_cov.append(dict(mass_MeV=m,relative_error=float(error)))
# Independently integrate the loop and invert each displayed locus.
g=pd.read_csv(B/'derived/g2_central_curves.csv');alpha=1/137.035999206
for name,ml,delta in [('muon_central_eps2',105.6583745,3.8e-10),('electron_Rb_central_eps2',.510998950,3.4e-13)]:
 errs=[]
 for row in g.iloc[::113].itertuples(index=False):
  r=row.mass_MeV/ml
  val=quad(lambda z:2*z*(1-z)**2/((1-z)**2+r*r*z),0,1,epsabs=1e-20,epsrel=2e-11,limit=300)[0]
  errs.append(abs(alpha*getattr(row,name)*val/(2*np.pi)/delta-1))
 check(name+'_loop_closure',max(errs)<1e-9)
# Rounded Fan23 and Rb20 alpha values reproduce the stated electron reference.
ae=1/137.035999166;ar=1/137.035999206
da=.5*(ae-ar)/np.pi-.328478965579193*(ae-ar)*(ae+ar)/np.pi**2
check('electron_reference_rounding',abs(da/3.4e-13-1)<.01)
check('no_unqualified_APEX_2019_exclusion','apex_2019_physics_run.csv' in json.loads((B/'derived/contour_display_protocol.json').read_text())['excluded_inputs'])
log=(B/'qa/build/main.log').read_text()
check('no_overfull_boxes','Overfull' not in log)
doc=fitz.open(B/'qa/build/main.pdf');texts=[p.get_text() for p in doc]
check('no_unresolved_references',not any('??' in t for t in texts))
check('version_retained','Analysis Note v5.0.4' in texts[0])
figure_pages=[i for i,t in enumerate(texts) if 'Figure 2:' in t]
check('one_figure2',len(figure_pages)==1)
check('both_subfigures_in_note','(a)2015+2016+2021' in re.sub(r'\s+','',texts[figure_pages[0]]) and '(b)2015+2016+2019+2021' in re.sub(r'\s+','',texts[figure_pages[0]]))
check('methods_appendix_present',any('L\nConstruction of the Figure 2 projections' in t for t in texts))
figure_qa={}
for name in ['figure2_clean_overview','figure2_overview_and_projections','figure2_projection_panels']:
 d=fitz.open(B/'figures'/f'{name}.pdf');check(name+'_one_page',len(d)==1);check(name+'_vector',not d[0].get_images())
 check(name+'_text',len(d[0].get_text())>200)
 d[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(B/'qa'/f'{name}_render.png')
 fonts=d[0].get_fonts(full=True);check(name+'_no_type3',all(f[2]!='Type3' for f in fonts))
 figure_qa[name]=dict(pages=1,vector=True,fonts=[f[3] for f in fonts],sha256=sha(B/'figures'/f'{name}.pdf'))
rendered=[]
for i in sorted(set([0,2,3,4,5,6,7,127,128,129,len(doc)-4,len(doc)-3,len(doc)-2,len(doc)-1])):
 doc[i].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(B/'qa'/f'review_page_{i+1:03d}.png');rendered.append(i+1)
proof=fitz.open();proof.insert_pdf(doc,from_page=figure_pages[0],to_page=figure_pages[0]);proof.save(B/'figures/figure2_captioned_proof.pdf')
write(B/'qa/validation.json',dict(passed=True,checks=checks,page_count=len(doc),figure2_page=figure_pages[0]+1,pdf_sha256=sha(B/'qa/build/main.pdf'),full_covariance_checks=full_cov,figures=figure_qa,review_pages=rendered,maximum_parent_three_limit_difference=float(abs(q.parent_replay_relative_error).max()),electron_residual_from_rounded_alpha=float(da),reference_limit_semantics='Current-sample refits for this figure only; frozen Section 6 results unchanged.'))
print(json.dumps(dict(passed=True,checks=len(checks),pages=len(doc),figure2_page=figure_pages[0]+1,full_covariance_checks=full_cov)))
