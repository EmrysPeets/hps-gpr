"""Run a reviewed local Python helper through the authenticated iana hop."""
import argparse,subprocess,shlex
from pathlib import Path
def run(script,args=(),timeout=180):
    inner='ssh -o BatchMode=yes -o ConnectTimeout=15 iana '+shlex.quote('/sdf/home/e/epeets/src/hps-gpr-main/venv/bin/python - '+' '.join(shlex.quote(str(a)) for a in args))
    return subprocess.run(['ssh','-S','/tmp/hps-v61-s3df.sock','-o','BatchMode=yes','epeets@s3dfdtn.slac.stanford.edu',inner],input=Path(script).read_bytes(),capture_output=True,timeout=timeout)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('script');p.add_argument('output');p.add_argument('args',nargs='*');p.add_argument('--timeout',type=int,default=180);a=p.parse_args()
    r=run(a.script,a.args,a.timeout);Path(a.output+'.stderr').write_bytes(r.stderr)
    if r.returncode:raise SystemExit(r.stderr.decode(errors='replace'))
    Path(a.output).write_bytes(r.stdout);print('Saved',len(r.stdout),'bytes to',a.output)
