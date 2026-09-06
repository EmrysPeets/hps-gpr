#!/usr/bin/env python3
from pathlib import Path
import json,time
import run_global as run
core,np,pd,c=run.core,run.np,run.pd,run.c
HERE=Path(__file__).resolve().parent
cfg=c.production.load_config(c.production.DEFAULT_CARD)
c.production.validate_card(cfg)
datasets=c.production.make_datasets(cfg)
states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
records=[]
for year,mass in [('2016',39),('2016',88),('2016',120),('2016',180),('2021',50),('2021',150),('2021',250)]:
 scope=next(s for s in core.SCOPES if s[2]==(year,))
 ctx=core.Context(scope,mass,cfg,datasets,states)
 start=time.monotonic();passed=core.enable_lowrank(ctx);gate_seconds=time.monotonic()-start
 source=HERE/'global'/year/'pilot10'
 counts=np.load(source/'spectra.npz')['counts']
 exact=np.load(source/f'm{mass:03d}.npz')
 start=time.monotonic();values,checks=run.evaluate(ctx,counts);fast_seconds=time.monotonic()-start
 errors={method:float(np.max(abs(values[method]-exact[method]))) for method in values}
 exact_seconds=json.loads((source/f'm{mass:03d}_qa.json').read_text())['seconds']
 record=dict(dataset=year,mass_MeV=mass,gate_passed=passed,backend=ctx.gp_backend,gate_seconds=gate_seconds,ten_fast_seconds=fast_seconds,ten_exact_seconds=exact_seconds,max_root_errors=errors,gate_records=ctx.numerical_checks,fallback_reason=ctx.gp_fallback_reason)
 records.append(record)
 run.write_json(HERE/'provenance/acceleration_benchmark.json',records)
 print(json.dumps({k:v for k,v in record.items() if k!='gate_records'}),flush=True)
 if max(errors.values())>=1e-3:raise RuntimeError('Pilot overlap numerical gate failed')
