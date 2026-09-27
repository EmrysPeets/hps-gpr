"""Package the validated continuation and unchanged parent report."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
from pypdf import PdfReader,PdfWriter
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def included(p):
 parts=p.relative_to(B).parts
 return p.is_file() and '__pycache__' not in parts and 'mpl' not in parts and not p.name.startswith('.')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 extension=out/'HPS_GPR_v5p8p5_Common_Mass_and_Coherence.pdf';shutil.copy2(B/'source/report.pdf',extension)
 parent=PdfReader(B/'inputs/v5p8p4_report.pdf');new=PdfReader(extension);writer=PdfWriter();writer.append(parent,import_outline=False);writer.append(new,import_outline=False)
 writer.add_outline_item('v5.8.4: original report',0);writer.add_outline_item('v5.8.5: common mass and coherence',len(parent.pages));writer.add_metadata({'/Title':'HPS-GPR v5.8.5: extended report','/Author':'Emrys Peets'})
 full=out/'HPS_GPR_v5p8p5_Extended_Report.pdf'
 with full.open('wb') as f:writer.write(f)
 joined=PdfReader(full);assert [p.extract_text() for p in joined.pages]==[p.extract_text() for p in parent.pages]+[p.extract_text() for p in new.pages]
 (B/'qa/package_validation.json').write_text(json.dumps(dict(complete=True,parent_pages=len(parent.pages),extension_pages=len(new.pages),extended_pages=len(joined.pages),concatenated_text_identical=True,extension_sha256=sha(extension)),indent=2)+'\n')
 files=[p for p in sorted(B.rglob('*')) if included(p) and p.name!='SHA256SUMS.txt']
 (B/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(B).as_posix()}\n' for p in files))
 files.append(B/'SHA256SUMS.txt')
 sourcezip=out/'HPS_GPR_v5p8p5_Source_and_Data.zip'
 with zipfile.ZipFile(sourcezip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,Path(B.name)/p.relative_to(B))
 plotzip=out/'HPS_GPR_v5p8p5_Plots.zip'
 with zipfile.ZipFile(plotzip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted((B/'figures').glob('*')):z.write(p,Path('figures')/p.name)
  z.write(B/'README.md','README.md')
 shutil.copy2(B/'README.md',out/'README.md')
 for p in [sourcezip,plotzip]:
  with zipfile.ZipFile(p) as z:assert z.testzip() is None
 with zipfile.ZipFile(sourcezip) as z:
  for p in files:assert hashlib.sha256(z.read(str(Path(B.name)/p.relative_to(B)))).hexdigest()==sha(p)
 manifest={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['SHA256SUMS.txt','manifest.json']}
 (out/'manifest.json').write_text(json.dumps(dict(version='5.8.5',study_directory=str(B),files=manifest),indent=2)+'\n')
 (out/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'))
 print(json.dumps(dict(output=str(out),source_files=len(files),parent_pages=len(parent.pages),extension_pages=len(new.pages),extended_pages=len(joined.pages),files=[p.name for p in out.iterdir()]),indent=2))
if __name__=='__main__':main()
