"""Use a user-authenticated control connection, then hop to iana.

Credentials are entered by the user in their own terminal. This helper only
reuses that connection and performs the read-only inventory.
"""
import argparse,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='epeets@s3dfdtn.slac.stanford.edu')
    parser.add_argument('--socket',default='/tmp/hps-v61-s3df.sock');args=parser.parse_args()
    base=json.loads((B/'protocol.json').read_text())['requested_mc_root']
    import shlex
    inner='ssh -o BatchMode=yes -o ConnectTimeout=15 iana '+shlex.quote('python3 - '+shlex.quote(base)+' --depth 2 --max-entries 10000')
    command=['ssh','-S',args.socket,'-o','BatchMode=yes','-o','ConnectTimeout=15',args.host,inner]
    result=subprocess.run(command,input=(B/'scripts/inventory_remote.py').read_text(),capture_output=True,text=True,timeout=90)
    (B/'qa/inventory_ssh_stderr.txt').write_text(result.stderr)
    if result.returncode:
        raise SystemExit('SSH inventory did not complete; no sample inventory was written. See qa/inventory_ssh_stderr.txt.')
    inventory=json.loads(result.stdout)
    if not inventory['root_exists']:raise SystemExit('Requested MC sample directory is not accessible on iana.')
    (B/'inputs/remote_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(host=inventory['host'],entries=len(inventory['entries']),truncated=inventory['truncated'],errors=inventory['errors']),indent=2))

if __name__=='__main__':main()
