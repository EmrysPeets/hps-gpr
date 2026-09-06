#!/usr/bin/env python3
"""Finish the already-running 2016 derivative, then process 2021 sequentially."""
from pathlib import Path
import json,subprocess,sys,time
HERE=Path(__file__).resolve().parent

def main():
    start=time.monotonic();folder=HERE/'global_fast/2016'
    while not (folder/'execution_summary.json').exists():
        if list(folder.glob('*FAILURE*')):
            raise RuntimeError('2016 numerical failure retained')
        if time.monotonic()-start>3600:
            raise RuntimeError('Timed out waiting for active 2016 calculation')
        time.sleep(5)
    records=[]
    for year in ('2016','2021'):
        if year=='2021' and not (HERE/'global_fast/2021/execution_summary.json').exists():
            cmd=['nice','-n','10',sys.executable,'-B',str(HERE/'run_global_accelerated.py'),'--dataset',year]
            print('Starting complete 2021 numerical derivative',flush=True)
            with (HERE/'logs/2021_fast.log').open('a') as log:
                run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            records.append(dict(command=cmd,exit_code=run.returncode))
            if run.returncode:
                raise RuntimeError('2021 numerical calculation failed')
        cmd=[sys.executable,'-B',str(HERE/'analyze_global.py'),'--dataset',year]
        print('Analyzing '+year,flush=True)
        with (HERE/'logs'/f'{year}_analysis.log').open('w') as log:
            run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(command=cmd,exit_code=run.returncode))
        (HERE/'provenance/accepted_pipeline_commands.json').write_text(json.dumps(records,indent=2)+'\n')
        print('Analysis exit '+year+': '+str(run.returncode),flush=True)
    if any(x['exit_code'] for x in records):
        raise RuntimeError('A saved analysis requires attention')

if __name__=='__main__':main()
