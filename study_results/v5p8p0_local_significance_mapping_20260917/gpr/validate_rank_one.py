from run_grid import *
from rank_one import RankOneAsimov
results=[]
for mass in [100.,150.]:
 old=np.load(OLD/'derived/field_2016/field.npz');truth=old['truth'];ctx=setup('2016',mass)
 bank=np.broadcast_to(truth,(len(truth)+1,len(truth))).copy();ii=np.arange(len(truth));bank[ii+1,ii]+=np.sqrt(truth)
 t=time.monotonic();fast=RankOneAsimov(ctx,truth,cfg);r=evaluate(ctx,bank,moment_provider=fast)
 j=int(np.flatnonzero(old['masses']==mass)[0]);expected=np.r_[old['a'][j],old['D'][:,j]+old['a'][j]]
 err=float(np.max(abs(r-expected)));assert err<2e-5,err
 sd=np.linalg.norm(r[1:]-r[0]);sd_err=float(abs(sd-old['s'][j]))
 q=dict(mass_MeV=mass,seconds=time.monotonic()-t,max_abs_root_difference=err,response_sd_abs_difference=sd_err,moment_checks=fast.qa);results.append(q);print(q,flush=True)
write(B/'rank_one_validation.json',dict(passed=True,results=results))
