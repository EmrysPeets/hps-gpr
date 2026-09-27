#!/usr/bin/env python3
"""Rebuild the reading guides and vector-preserving plotbook from saved artifacts."""
import csv
import hashlib
import html
import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote

from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'output/pdf/study_logbook_20260927'
GH = 'https://github.com/EmrysPeets/hps-gpr/blob/studies-2026-09-27/'
W, H = 1008, 720
NAVY, TEAL, INK, MUTED = '#15324a', '#147a81', '#20313d', '#566675'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return p.relative_to(ROOT).as_posix()


def mdlink(p):
    import os
    return quote(os.path.relpath(p,HERE).replace('\\','/'), safe='/._-')


def url(p):
    return GH + quote(rel(p),safe='/._-')


def readcsv(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def claim_checks(cat):
    checks=[]
    def check(name,passed,values,source):
        if not passed:
            raise AssertionError((name,values))
        checks.append(dict(name=name,passed=True,values=values,source=rel(source),sha256=digest(source)))
    p=ROOT/cat['v6p4p4_calibrated_local_global_20260925']['path']/'results/calibrated_summary.csv'
    rows=readcsv(p)
    for r,k in zip(rows,[89,231,75]):
        check(r['scope']+' B global count',int(r['minp_B_k'])==k and int(r['minp_B_N'])==1024 and abs(float(r['minp_B_p'])-(k+1)/1025)<1e-14,
              {x:r[x] for x in ['scope','localfirst_peak_mass_MeV','minp_B_k','minp_B_N','minp_B_p','minp_B_p95_low','minp_B_p95_high','local_map_floor']},p)
    check('Combined rank floor',rows[2]['local_map_floor']=='True' and int(rows[2]['local_A_exceedances'])==0,rows[2]['local_A_p'],p)
    p=ROOT/cat['v6p3p1_fixed_yield_2021_100toy_20260924']['path']/'results/evaluation_summary.csv'
    rows=readcsv(p)
    check('Fixed-yield complete 80 cells',len(rows)==80 and all(int(r['attempted'])==int(r['fit_valid'])==int(r['profile_valid'])==100 for r in rows),len(rows),p)
    for shape,expected,contain in [('gaussian',[.961,.978],[88,98]),('mc',[.716,.894],[88,97])]:
        selected=[r for r in rows if r['shape']==shape and float(r['z'])==5]
        values=[float(r['paired_response_mean']) for r in selected]
        counts=[int(r['contain95_k']) for r in selected]
        check(shape+' z5 response',len(selected)==10 and [round(min(values),3),round(max(values),3)]==expected,[min(values),max(values)],p)
        check(shape+' z5 containment',[min(counts),max(counts)]==contain,[min(counts),max(counts)],p)
    p=ROOT/cat['v6p2_mc_injection_20260923']['path']/'results/paired.csv'
    rows=[r for r in readcsv(p) if float(r['injected_N'])==30000]
    for method,expected in [('pole',[71.5,89.5]),('core',[78.0,93.8])]:
        vals=[100*float(r['mean_incremental_recovery_'+method]) for r in rows]
        check('Final v6.2 '+method+' response',[round(min(vals),1),round(max(vals),1)]==expected,[min(vals),max(vals)],p)
    p=ROOT/cat['v6p2_mc_injection_20260923']['path']/'results/summary.csv'
    rows=readcsv(p)
    check('v6.2 final cohort',all(int(r['n'])==40 for r in rows if r['control']=='contaminated_gp'),40,p)
    p=ROOT/cat['unblind_meeting_RCmeet_20260923b']['path']/'science/data/sideband_center_summary.csv'
    rows=readcsv(p)
    check('42 pointwise sideband checks',len(rows)==42 and all(float(r['toy_q05'])<=float(r['observed_D_per_bin'])<=float(r['toy_q95']) for r in rows),len(rows),p)
    p=ROOT/cat['v6p3p9_combined_morph_20260925']['path']/'results/summary.json'
    d=json.loads(p.read_text())['minima']['morph_starter']
    check('2021-only MC combined local peak',d['mass_MeV']==68 and round(d['Z_local'],3)==2.910 and round(d['p0'],6)==.001806,d,p)
    p=ROOT/cat['v6p4_2016_mc_shapes_20260925']['path']/'results/summary.json'
    d=json.loads(p.read_text())
    check('2016 catalogue qualification',d['histograms']==29 and d['missing_grid_masses_MeV']==[150] and d['invalid_nominal']==[30],{k:d[k] for k in ['histograms','primary_shift_range_MeV','primary_fraction_2_range','missing_grid_masses_MeV']},p)
    p=ROOT/cat['v6p4p2_2016_window_comparison_20260925']['path']/'results/window_comparison/signal_response.csv'
    rows=readcsv(p)
    for width,expected in [(2,[.877,.922]),(3.5,[.998,1.001])]:
        vals=[float(r['response']) for r in rows if float(r['half_width_u'])==width]
        check('2016 response width '+str(width),len(vals)==5 and [round(min(vals),3),round(max(vals),3)]==expected,[min(vals),max(vals)],p)
    (HERE/'qa/source_claim_checks.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
    return checks


def report_for(d):
    if d['series']=='Presentation':
        p=ROOT/d['path']/'renders/delivered/presentation.pdf'
        if p.exists():return p
    p=ROOT/d['path']
    candidates=[p/'report.pdf',p/'source/report.pdf']+list((p/'pdf').glob('*.pdf'))
    candidates=[q for q in candidates if q.exists() and q.name!='appendix.pdf']
    if candidates:return max(candidates,key=lambda q:q.stat().st_size)
    if d['folder']=='apex_initial_studies':return p/'output/pdf/APEX_Initial_Studies.pdf'
    return None


def paragraph(c,text,x,top,width,size=11,leading=None,color=INK,font='Helvetica',maxheight=None):
    style=ParagraphStyle('body',fontName=font,fontSize=size,leading=leading or size*1.35,textColor=colors.HexColor(color))
    p=Paragraph(text,style)
    _,height=p.wrap(width,1000)
    if maxheight is not None and height>maxheight:
        raise ValueError('Text exceeds planned frame: '+text[:80]+f' ({height}>{maxheight})')
    p.drawOn(c,x,top-height)
    return top-height


def header(c,label,title,subtitle,page):
    c.setFillColor(colors.HexColor(TEAL));c.rect(0,H-10,W,10,fill=1,stroke=0)
    paragraph(c,label.upper(),44,686,920,9,color=TEAL,font='Helvetica-Bold')
    paragraph(c,html.escape(title),44,663,920,25,29,color=NAVY,font='Times-Bold',maxheight=58)
    paragraph(c,html.escape(subtitle),44,620,920,11,14,color=INK,maxheight=30)
    c.setStrokeColor(colors.HexColor('#d9e1e7'));c.line(44,45,W-44,45)
    paragraph(c,'HPS GPR  |  13-27 September 2026  |  Saved-result review',44,30,800,8,color=MUTED)
    c.setFillColor(colors.HexColor(MUTED));c.setFont('Helvetica',9);c.drawRightString(W-44,20,str(page))


def block(c,label,text,x,y,width,size=11):
    y=paragraph(c,label.upper(),x,y,width,9,color=TEAL,font='Helvetica-Bold')-7
    return paragraph(c,html.escape(text),x,y,width,size)-16


def build_pdf(catalog,cards):
    OUT.mkdir(parents=True,exist_ok=True)
    base=HERE/'qa/layout.pdf'
    c=canvas.Canvas(str(base),pagesize=(W,H),pageCompression=1)
    c.setTitle('HPS GPR recent studies: plot logbook, 13-27 September 2026')
    c.setAuthor('Emrys Peets | compiled study logbook')
    placements=[]
    header(c,'Reading guide','HPS GPR study and plot logbook','Forty study and presentation entries; 26 selected plot pages with sources and speaking notes.',1)
    y=block(c,'What the recent series established','Native signal MC motivates campaign-specific shape and window choices. Paired injections and clean-training controls reveal response loss that a lower observed upper limit can hide.',44,564,430,14)
    y=block(c,'What to present first','Use the 12-minute route on page 2. Pages 4-29 are reusable plot cards; the final pages locate the complete study packages and dated presentation snapshots.',44,y,430,13)
    block(c,'What remains conditional','Saved templates, null sources, kernel states and method choices define each experiment. These studies do not by themselves establish physical efficiency, unconditional coverage, a calibrated coupling exclusion or discovery.',44,y,430,12)
    paragraph(c,'Independent local-to-global result',536,564,430,17,font='Times-Bold',color=NAVY)
    y=520
    for name,count,p,ci in [('2016','89 / 1,024','0.08780','[0.07038, 0.10587]'),('2021 10%','231 / 1,024','0.22634','[0.20032, 0.25244]'),('Common coupling','75 / 1,024','0.07415','[0.05804, 0.09095]')]:
        c.setFillColor(colors.HexColor('#f0f5f7'));c.roundRect(528,y-75,430,83,5,fill=1,stroke=0)
        paragraph(c,name,545,y-3,400,13,font='Helvetica-Bold')
        paragraph(c,f'B count: {count}   |   add-one p: <b>{p}</b>',545,y-26,400,12)
        paragraph(c,'Exact 95% binomial interval: '+ci,545,y-47,400,10,color=MUTED)
        y-=100
    paragraph(c,'v6.4.4: fixed local maps from 1,024 A scans; independent 1,024 B scans. Each quoted interval conditions on A and the null source. Combined is a joint one-coupling fit. See P23-P26.',536,y-3,410,11)
    c.showPage()
    header(c,'Presentation route','A twelve-minute explanation','Eight stops, about 90 seconds each. Use the other cards to answer follow-up questions.',2)
    route=[('P09','Start with the selected signal-MC distributions.','Their broad tails and displaced cores motivate changes to the extraction model.'),('P12','Compare both shapes at a fixed expected yield.','Use independent pilot and evaluation cohorts; distinguish raw recovery from the paired increment.'),('P17','Show the controlled training-contamination experiment.','Removing only training-bin signal nearly restores direct-MC response in the displayed checks.'),('P20','Keep campaign-specific shape evidence.','2016 has a smaller core shift and must not inherit the 2021 law.'),('P22','Explain why a lower limit can be misleading.','Reduced response largely cancels the narrower window\'s smaller raw spread.'),('P19','Show what changes in the observed joint fit.','The common-coupling local minimum moves when the 2021 template changes; keep that release distinct.'),('P23','Give the independent whole-scan probability.','Combined p=0.07415, conditional on the fixed local maps, source and declared grid.'),('P25','Finish with the mass correlations.','Overlapping fits explain why raw grid counts and a single resolution count cannot replace coherent scan toys.')]
    for i,(pid,heading,text) in enumerate(route):
        col=i//4;row=i%4;x=44+col*476;y=565-row*124
        page=3+int(pid[1:])
        paragraph(c,f'{i+1:02d} / {pid} / PAGE {page}',x,y,440,9,color=TEAL,font='Helvetica-Bold')
        y=paragraph(c,html.escape(heading),x,y-20,440,14,18,font='Helvetica-Bold')-6
        paragraph(c,html.escape(text),x,y,435,11,15)
    c.showPage()
    header(c,'Definitions','Keep these distinctions visible','The versions are a chain of conditional studies, not repeated independent confirmations.',3)
    distinctions=[('Raw recovery and paired response','Raw recovery is mean(Ahat/A). Paired response is mean[(Ahat_signal+background-Ahat_background)/A], using the same background toy. A null subtraction cancels an offset; it does not correct response.'),('Expected yield and realized counts','v6.2 fixes exact N and samples multinomial categories. v6.3.1 fixes expected A=z*s0 from an independent pilot and lets Poisson counts fluctuate. Its pulls use expected A.'),('Reference resolution and MC core width','The original mask uses +/-2.25 sigma_ref around the hypothesis or shifted center. MC u=(mass-core center)/core width is another coordinate. A value expressed in u cannot be treated as sigma_ref.'),('Fixed source and fitted background','A fixed GP mean may generate the toys while each toy still recomputes its conditional GP prediction. Holding that prediction fixed inside the likelihood (C=0) is a further, different change.'),('Local and global ordering','v6.4.3 calibrates max(raw q0). v6.4.4 freezes A local rank maps and calibrates min(local rank p) with independent B. Their global probabilities answer different questions.'),('Common coupling and other combinations','The joint physical model uses one coupling with campaign-specific normalizations and nuisance blocks. Independent amplitudes, Fisher/Stouffer, and selecting among searches are separate hypotheses.')]
    for i,(title,text) in enumerate(distinctions):
        x=44+(i%2)*476;y=568-(i//2)*156
        block(c,title,text,x,y,430,12)
    paragraph(c,'Finite ranks: zero exceedances gives an add-one floor and a binomial bound. An empty accepted yield grid is not a physical zero limit. Uncertainty bars retain the source study\'s definition; MC sampling uncertainty omits unmodeled systematic effects.',44,93,920,10.5)
    c.showPage()
    for card in cards:
        study=catalog[card['study']];page=3+int(card['id'][1:])
        header(c,card['id']+' / '+study['series']+' / '+study['folder'].split('_2026')[0],card['title'],card['takeaway'],page)
        png=HERE/card['asset_png'];im=Image.open(png);aspect=im.width/im.height
        if aspect<1.20:
            box=(44,74,590,518);tx=665;tw=299;y=564
            y=block(c,'Read the plot',card['read'],tx,y,tw)
            y=block(c,'Suggested explanation',card['say'],tx,y,tw)
            y=block(c,'Claim boundary',card['boundary'],tx,y,tw,10.5)
            if y<82:raise ValueError('Caption overflow '+card['id'])
        else:
            box=(44,214,920,378)
            block(c,'Read the plot',card['read'],44,191,438,10.5)
            y=block(c,'Suggested explanation',card['say'],522,191,440,10.5)
            y=block(c,'Claim boundary',card['boundary'],522,y,440,10)
            if y<60:raise ValueError('Wide caption overflow '+card['id'])
        if card.get('asset_pdf'):
            placements.append(dict(page=page-1,pdf=HERE/card['asset_pdf'],box=box))
        else:
            x,y,bw,bh=box;s=min(bw/im.width,bh/im.height)
            c.drawImage(str(png),x+(bw-im.width*s)/2,y+(bh-im.height*s)/2,width=im.width*s,height=im.height*s,preserveAspectRatio=True)
        x=44
        for label,source in [('Original figure',ROOT/card['source_png']),('Source data / definition',ROOT/card['source_evidence']),('Study',ROOT/study['path'])]:
            c.setFillColor(colors.HexColor(TEAL));c.setFont('Helvetica',8);c.drawString(x,54,label)
            width=c.stringWidth(label,'Helvetica',8);c.linkURL(url(source),(x,51,x+width,62),relative=0)
            x+=width+24
        c.setFillColor(colors.HexColor(MUTED));c.drawRightString(W-44,54,'Figure SHA-256: '+card['sha256'][:16])
        c.showPage()
    groups=[('Context and earlier significance','Context','Significance'),('Signal templates, recovery and observed fits','Templates','Recovery','Observed'),('Global calibration and presentation history','Global','Presentation')]
    for j,(title,*series) in enumerate(groups):
        page=30+j
        header(c,'Study finder',title,'Full explanations and links are in STUDY_LOG.md; every catalogue entry has a provenance-backed package.',page)
        selected=[x for x in catalog.values() if x['series'] in series]
        for i,study in enumerate(selected):
            x=44+(i%2)*476;y=564-(i//2)*60
            paragraph(c,html.escape(study['title']),x,y,435,12,font='Helvetica-Bold')
            text=study['folder'].replace('_202609',' / Sep ')
            paragraph(c,html.escape(text),x,y-21,435,8.5,color=MUTED)
            c.linkURL(url(ROOT/study['path']),(x,y-40,x+435,y+3),relative=0)
        if j==2:
            paragraph(c,'Archive and reproducibility',44,244,920,18,font='Times-Bold',color=NAVY)
            paragraph(c,'The publication snapshot preserves 40 study/presentation entries, complete saved inputs and ledgers, and all member payloads of 53 delivery ZIPs. Large raw files are stored losslessly compressed. The archive README explains verification, restoring large files, and rebuilding ZIP containers with identical member bytes.',44,211,900,12)
            paragraph(c,'This logbook is rebuilt from saved artifacts only. No new scientific fits or toys were run for it. The original study workspace remains separate from the publication checkout. Figures remain linked to original sources, with selected PNG/PDF copies and hashes for presentation reuse.',44,142,900,12)
        c.showPage()
    c.save()
    reader=PdfReader(base);writer=PdfWriter()
    for p in reader.pages:writer.add_page(p)
    for item in placements:
        original=PdfReader(item['pdf']).pages[0]
        x,y,bw,bh=item['box'];fw=float(original.mediabox.width);fh=float(original.mediabox.height)
        scale=min(bw/fw,bh/fh)
        transform=Transformation().scale(scale).translate(x+(bw-fw*scale)/2,y+(bh-fh*scale)/2)
        writer.pages[item['page']].merge_transformed_page(original,transform)
    writer.add_metadata({'/Title':'HPS GPR recent studies: plot logbook, 13-27 September 2026','/Author':'Emrys Peets','/Subject':'Saved-result study catalogue and presentation plot guide'})
    for title,page in [('Start here',0),('12-minute presentation route',1),('Definitions',2)]:writer.add_outline_item(title,page)
    for card in cards:writer.add_outline_item(card['id']+' '+card['title'],2+int(card['id'][1:]))
    writer.add_outline_item('Study finder',29)
    final=OUT/'HPS_GPR_Recent_Studies_Plot_Logbook.pdf'
    with final.open('wb') as f:writer.write(f)
    return final,len(reader.pages)


def main():
    HERE.joinpath('assets').mkdir(exist_ok=True)
    HERE.joinpath('qa').mkdir(exist_ok=True)
    entries=json.loads((HERE/'catalog.json').read_text())
    cards=json.loads((HERE/'plot_cards.json').read_text())
    catalog={x['folder']:x for x in entries}
    for entry in entries:
        d=ROOT/entry['path'];assert d.is_dir(),d
        report=report_for(entry)
        entry['report']=rel(report) if report and report.exists() else None
        intro=d/'README.md'
        if not intro.exists():intro=d/('CHANGELOG.md' if entry['series']=='Presentation' else 'method_audit.json')
        entry['readme']=rel(intro) if intro.exists() else None
        entry['plot_ids']=[x['id'] for x in cards if x['study']==entry['folder']]
    for card in cards:
        study=catalog[card['study']];source=ROOT/study['path']/card['figure'];evidence=ROOT/study['path']/card['evidence']
        assert source.is_file(),source
        assert evidence.is_file(),evidence
        stem=card['id']+'_'+source.stem
        dest=HERE/'assets'/(stem+'.png');shutil.copy2(source,dest)
        card.update(asset_png=dest.relative_to(HERE).as_posix(),source_png=rel(source),source_evidence=rel(evidence),sha256=digest(source))
        vector=source.with_suffix('.pdf')
        if vector.exists():
            dest=HERE/'assets'/(stem+'.pdf');shutil.copy2(vector,dest)
            card['asset_pdf']=dest.relative_to(HERE).as_posix();card['source_pdf']=rel(vector)
    checks=claim_checks(catalog)
    final,pages=build_pdf(catalog,cards)
    (HERE/'catalog.json').write_text(json.dumps(entries,indent=2,ensure_ascii=False)+'\n')
    (HERE/'plot_cards.json').write_text(json.dumps(cards,indent=2,ensure_ascii=False)+'\n')
    summary=['# Recent HPS GPR study log\n','Snapshot: 27 September 2026. Primary scope: 13-27 September; older dependencies and the previously pushed v5.0.5 note are retained. This is a saved-result publication and reading guide.\n',
             'Start with the [plotbook PDF]('+mdlink(final)+'), [illustrated plot guide](PLOTBOOK.md), [searchable local gallery](index.html), or [presentation notes](PRESENTATION_NOTES.md). The catalogue JSON is the machine-readable index.\n',
             'Study versions can change the template, mask, source and scan ordering. Read each result under its own definition. The v6.3.6 note incorporates several later appendices; these are not additional independent replications.\n']
    for series in ['Context','Significance','Templates','Recovery','Observed','Global','Presentation']:
        summary.append('## '+series+'\n')
        for e in entries:
            if e['series']!=series:continue
            summary += ['### '+e['title']+'\n','`'+e['folder']+'`\n',
                        '**Question.** '+e['question']+'\n','**Accomplished.** '+e['accomplishment']+'\n','**Interpretation.** '+e['limit']+'\n']
            links=['[Full package]('+mdlink(ROOT/e['path'])+')']
            if e['readme']:links.append('[Definitions / log]('+mdlink(ROOT/e['readme'])+')')
            if e['report']:links.append('[Report PDF]('+mdlink(ROOT/e['report'])+')')
            if e['plot_ids']:links+=['['+pid+'](PLOTBOOK.md#'+pid.lower()+')' for pid in e['plot_ids']]
            summary.append(' | '.join(links)+'\n')
    (HERE/'STUDY_LOG.md').write_text('\n'.join(summary))
    plotmd=['# Selected plots and speaking notes\n','[Study log](STUDY_LOG.md) | [PDF plotbook]('+mdlink(final)+') | [12-minute route](PRESENTATION_NOTES.md)\n']
    for card in cards:
        plotmd += ['<a id="'+card['id'].lower()+'"></a>\n','## '+card['id']+' - '+card['title']+'\n','**'+card['takeaway']+'**\n','!['+card['title']+']('+card['asset_png']+')\n','**Read the plot.** '+card['read']+'\n','**Suggested explanation.** '+card['say']+'\n','**Claim boundary.** '+card['boundary']+'\n']
        links=['[PNG]('+card['asset_png']+')','[Source data / definition]('+mdlink(ROOT/card['source_evidence'])+')','[Study package]('+mdlink(ROOT/catalog[card['study']]['path'])+')']
        if card.get('asset_pdf'):links.insert(1,'[Original vector PDF]('+card['asset_pdf']+')')
        plotmd+=[' | '.join(links)+'\n']
    (HERE/'PLOTBOOK.md').write_text('\n'.join(plotmd).replace('\\\n','\n'))
    htmlcards=[]
    for card in cards:
        e=catalog[card['study']];esc=html.escape
        links='<a href="'+card['asset_png']+'">PNG</a>'
        if card.get('asset_pdf'):links+=' <a href="'+card['asset_pdf']+'">Vector PDF</a>'
        links+=' <a href="'+mdlink(ROOT/card['source_evidence'])+'">Source / data</a>'
        htmlcards.append('<article data-series="'+e['series']+'"><span class="tag">'+card['id']+' / '+e['series']+'</span><h2>'+esc(card['title'])+'</h2><p class="take">'+esc(card['takeaway'])+'</p><a href="'+card['asset_png']+'"><img loading="lazy" src="'+card['asset_png']+'" alt="'+esc(card['title'])+'"></a><p><b>Read:</b> '+esc(card['read'])+'</p><p><b>Say:</b> '+esc(card['say'])+'</p><p class="boundary"><b>Boundary:</b> '+esc(card['boundary'])+'</p><nav>'+links+'</nav><small>'+esc(e['folder'])+'</small></article>')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>HPS GPR study plotbook</title><style>
    :root{font-family:system-ui,sans-serif;color:#20313d;background:#edf2f5}body{max-width:1260px;margin:auto;padding:30px}header{padding:12px 0 25px}h1{font:44px Georgia,serif;color:#15324a;margin:8px 0}h2{font:25px Georgia,serif;margin:8px 0}p{line-height:1.5}.tag{color:#147a81;font-size:12px;font-weight:700;letter-spacing:.06em}input,select{padding:12px;font-size:16px;border:1px solid #bacad4;border-radius:5px}input{min-width:340px}section{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}article{background:white;padding:24px;border:1px solid #d7e1e7;border-radius:8px}article[hidden]{display:none}img{width:100%;max-height:490px;object-fit:contain;background:white}.take{font-weight:600}.boundary{background:#f1f6f6;padding:12px;font-size:14px}a{color:#116c78;margin-right:12px}small{display:block;margin-top:16px;color:#60717b;font-size:11px;overflow-wrap:anywhere}nav{margin:15px 0}#count{color:#566675}@media(max-width:850px){section{grid-template-columns:1fr}body{padding:18px}input{min-width:0;width:85%}}@media print{section{display:block}article{break-inside:avoid}input,select{display:none}}</style>
    <header><span class="tag">13-27 SEPTEMBER 2026 / SAVED-RESULT REVIEW</span><h1>HPS GPR study plotbook</h1><p>26 selected plots with findings, reading instructions, speaking notes and claim boundaries. Forty study and presentation entries are indexed in the study log.</p><nav><a href="STUDY_LOG.md">Complete study log</a><a href="PRESENTATION_NOTES.md">Presentation route</a><a href="'''+mdlink(final)+'''">PDF plotbook</a></nav><input id="search" aria-label="Search plots" placeholder="Search: leakage, global, 2016, response..."><select id="series" aria-label="Filter series"><option value="">All series</option>'''+''.join('<option>'+s+'</option>' for s in sorted(set(e['series'] for e in entries)))+'''</select><p id="count"></p></header><section>'''+''.join(htmlcards)+'''</section><script>const q=document.getElementById('search'),s=document.getElementById('series'),a=[...document.querySelectorAll('article')];function filter(){let n=0;for(const c of a){const show=(!s.value||c.dataset.series===s.value)&&c.textContent.toLowerCase().includes(q.value.toLowerCase());c.hidden=!show;n+=show;}document.getElementById('count').textContent=n+' of '+a.length+' plots';}q.addEventListener('input',filter);s.addEventListener('change',filter);filter();</script></html>'''
    (HERE/'index.html').write_text(doc)
    source_manifest=[dict(id=x['id'],source=x['source_png'],sha256=x['sha256'],evidence=x['source_evidence'],evidence_sha256=digest(ROOT/x['source_evidence'])) for x in cards]
    (HERE/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
    print(json.dumps(dict(pdf=rel(final),pages=pages,plots=len(cards),studies=len(entries),numerical_checks=len(checks)),indent=2))


if __name__=='__main__':main()
