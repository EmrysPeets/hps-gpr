#!/usr/bin/env python3
"""Sequential continuation of the declared ensembles, with command records."""
from pathlib import Path
import argparse, datetime, json, subprocess, sys, time

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-running-2016', action='store_true')
    args = parser.parse_args()
    if args.wait_for_running_2016:
        start = time.monotonic()
        folder = HERE/'global/2016/validation1000'
        while not (folder/'summary.json').exists():
            if list(folder.glob('*FAILURE*')):
                raise RuntimeError('The running 2016 validation saved a failure')
            if time.monotonic()-start>7200:
                raise RuntimeError('Timed out waiting for the already running phase')
            time.sleep(5)
    records = []
    for year in ('2016','2021'):
        for ensemble in ('validation1000','asimov'):
            folder = HERE/'global'/year/ensemble
            saved = folder/'summary.json'
            if saved.exists():
                info = json.loads(saved.read_text())
                if info['complete'] and info['passed']:
                    continue
            cmd = ['nice','-n','10',sys.executable,'-B',str(HERE/'run_global.py'),
                   '--dataset',year,'--ensemble',ensemble]
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            print('Starting '+year+' '+ensemble, flush=True)
            with (HERE/'logs'/f'{year}_{ensemble}.log').open('a') as log:
                run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            records.append(dict(command=cmd,started_utc=stamp,exit_code=run.returncode))
            (HERE/'provenance/continuation_commands.json').write_text(json.dumps(records,indent=2)+'\n')
            if run.returncode:
                raise RuntimeError('Phase failed: '+year+' '+ensemble)
        cmd = [sys.executable,'-B',str(HERE/'analyze_global.py'),'--dataset',year]
        print('Analyzing '+year, flush=True)
        with (HERE/'logs'/f'{year}_analysis.log').open('w') as log:
            run = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(command=cmd,exit_code=run.returncode))
        (HERE/'provenance/continuation_commands.json').write_text(json.dumps(records,indent=2)+'\n')
        if run.returncode:
            raise RuntimeError('Analysis failed: '+year)
        print('Completed '+year,flush=True)

if __name__ == '__main__':
    main()
