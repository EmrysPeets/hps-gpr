from pathlib import Path
import argparse,subprocess,os,shutil,json,hashlib
from build_combined_appendix import build_combined_appendix
B=Path(__file__).resolve().parents[1]
def table(headers,rows,fmt=None):
 fmt=fmt or 'l'+'r'*(len(headers)-1)
 return '\n'.join([r'\begin{center}\small\begin{tabular}{'+fmt+r'}\toprule',' & '.join(headers)+r'\\\midrule']+[' & '.join(map(str,row))+r'\\' for row in rows]+[r'\bottomrule\end{tabular}\end{center}'])
def pic(name,caption,height=''):
 pic.number+=1;opts=r'width=\linewidth'+(',height='+height+',keepaspectratio' if height else '')
 return r'\begin{center}\includegraphics['+opts+']{../figures/'+name+r'.pdf}\end{center}'+'\n'+r'{\small\textbf{Figure '+str(pic.number)+'.} '+caption+'}\\par\\medskip\n'
def main(build):
 pic.number=0
 pre=r'''\documentclass[11pt]{article}
\usepackage[margin=.73in]{geometry}
\usepackage{lmodern,amsmath,amssymb,booktabs,graphicx,microtype,fancyhdr,xurl,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black}
\pagestyle{fancy}\fancyhf{}\lhead{HPS GPR: combined morphed-template scan}\rhead{v6.3.9}\cfoot{\thepage}
\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\emergencystretch}{2em}
\begin{document}
'''
 doc=pre+build_combined_appendix(B,pic,table).replace('\\clearpage','',1)+'\n'+r'\end{document}'+'\n';(B/'source/report.tex').write_text(doc)
 if build:
  subprocess.run([os.environ.get('TECTONIC') or shutil.which('tectonic') or '/opt/homebrew/bin/tectonic','--only-cached','--keep-logs','--outdir',str(B/'pdf'),str(B/'source/report.tex')],check=True,cwd=B/'source')
  (B/'pdf/report.pdf').replace(B/'pdf/HPS_GPR_v6p3p9_Combined_Morph_Scan.pdf')
 (B/'source/report_manifest.json').write_text(json.dumps({'figures':pic.number,'source_sha256':hashlib.sha256(doc.encode()).hexdigest()},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');main(p.parse_args().build)
