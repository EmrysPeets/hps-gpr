"""Read-only bounded sample inventory; run on iana with Python 3.

Lists metadata only. Does not read full event payloads, write remote files,
change samples, or silently infer generated mass from ambiguous filenames.
"""
import argparse,json,os,socket,time
from pathlib import Path

def inventory(root,depth=2,max_entries=10000):
    root=Path(root);start=time.time();rows=[];errors=[];queue=[(root,0)];truncated=False
    while queue and len(rows)<max_entries:
        directory,level=queue.pop(0)
        try:
            entries=sorted(os.scandir(directory),key=lambda x:x.name)
        except OSError as error:
            errors.append({'path':str(directory),'error':str(error)});continue
        for entry in entries:
            if len(rows)>=max_entries:truncated=True;break
            try:
                stat=entry.stat(follow_symlinks=False);isdir=entry.is_dir(follow_symlinks=False)
                rows.append(dict(path=entry.path,relative_path=str(Path(entry.path).relative_to(root)),
                                 kind='directory' if isdir else 'symlink' if entry.is_symlink() else 'file',
                                 bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
                if isdir and level<depth:queue.append((Path(entry.path),level+1))
            except OSError as error:errors.append(dict(path=entry.path,error=str(error)))
    return dict(host=socket.gethostname(),root=str(root),root_exists=root.is_dir(),max_depth=depth,
                truncated=truncated or bool(queue),seconds=time.time()-start,entries=rows,errors=errors)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--max-entries',type=int,default=10000);args=parser.parse_args()
    print(json.dumps(inventory(args.root,args.depth,args.max_entries),indent=2))
