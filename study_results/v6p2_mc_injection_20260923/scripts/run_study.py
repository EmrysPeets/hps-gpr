"""Bounded, resumable mass workers. Set STOP to stop at the next toy boundary."""
import argparse,time,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
import injection_core as I

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--masses',type=int,nargs='+',default=list(I.MASSES))
    ap.add_argument('--toys',type=int,default=I.TOYS)
    ap.add_argument('--pilot',action='store_true')
    args=ap.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Maximum four single-thread workers')
    if set(args.masses)-set(I.MASSES):raise ValueError('Only native60--260 masses')
    if not args.pilot and args.toys!=I.TOYS:raise ValueError(f'Production requires {I.TOYS} toys')
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(I.run_mass,m,args.toys,args.pilot) for m in args.masses]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    print(json.dumps({'complete':True,'seconds':time.monotonic()-start}),flush=True)

if __name__=='__main__':main()
