"""Append the window study without changing the main-body numerical material."""
import build_report as parent
from pathlib import Path
import shutil,subprocess
B=Path(__file__).resolve().parents[1]
def main():
    shapes=(B/'source/shape_sections.tex').read_text();marker=r'\clearpage\section*{Native signal-MC catalogue: 30--75 MeV}'
    body,gallery=shapes.split(marker,1)
    pre=parent.PREAMBLE.replace('v6.4.1','v6.4.2').replace('Version 6.4.1','Version 6.4.2')
    pre=pre.replace('The final catalogue shows every supplied 2016 histogram.',r'The catalogue shows every supplied 2016 histogram. Appendix A (page~\pageref{sec:windowappendix}) compares a narrower $\pm2u$ window for 2016; the main-body analysis remains at $\pm3.5u$.')
    text=pre+body+parent.METHOD+(B/'source/extraction_sections.tex').read_text()+parent.LOCAL+parent.RANK.replace('@@RANK_TABLE@@',parent.rank_table())+parent.END+marker+gallery
    text+=(B/'source/window_appendix.tex').read_text()+r'\end{document}'+'\n'
    (B/'source/report.tex').write_text(text)
    subprocess.run([shutil.which('tectonic') or '/opt/homebrew/bin/tectonic','--only-cached','--keep-logs','--outdir',str(B/'pdf'),str(B/'source/report.tex')],cwd=B/'source',check=True)
if __name__=='__main__':main()
