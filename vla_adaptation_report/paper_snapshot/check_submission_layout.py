#!/usr/bin/env python3
"""Check a built ICLR initial submission; requires pdftotext and pdfinfo.

The backmatter starts after clearpage, so any deferred main-text figures/tables
are included in the measured main-page count. Run after latexmk completes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

PAPER = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-main-pages', type=int, default=9)
    parser.add_argument('--require-main-pages', type=int,
                        help='Additionally require this exact main-text page count')
    parser.add_argument('--out', type=Path, default=PAPER/'submission_layout_receipt.json')
    args = parser.parse_args()
    if args.max_main_pages <= 0:
        raise ValueError('Page limit must be positive')
    aux = (PAPER/'iclr_draft.aux').read_text()
    match = re.search(r'\\newlabel\{backmatter:firstpage\}\{\{[^}]*\}\{(\d+)\}', aux)
    if not match:
        raise ValueError('Build the draft with its explicit backmatter boundary first')
    backmatter = int(match.group(1))
    main_pages = backmatter - 1
    if not 1 <= main_pages <= args.max_main_pages:
        raise ValueError(f'Main paper has {main_pages} pages; limit is {args.max_main_pages}')
    if args.require_main_pages is not None and main_pages != args.require_main_pages:
        raise ValueError(f'Main paper has {main_pages} pages; requested exactly {args.require_main_pages}')
    pdf = PAPER/'iclr_draft.pdf'
    extracted = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], check=True,
                               capture_output=True, text=True).stdout
    pages = extracted.split('\f')
    if pages and not pages[-1].strip():
        pages.pop()
    if 'ai use statement' not in pages[backmatter - 1].lower():
        raise ValueError('PDF and auxiliary file disagree about the backmatter boundary')
    info = subprocess.run(['pdfinfo', str(pdf)], check=True, capture_output=True, text=True).stdout
    total = int(re.search(r'^Pages:\s+(\d+)$', info, re.M).group(1))
    if total != len(pages):
        raise ValueError('PDF page extraction and pdfinfo disagree')
    log = (PAPER/'iclr_draft.log').read_text()
    errors = re.findall(r'^.*(?:Overfull|undefined|LaTeX Warning).*$' ,log,re.M)
    if errors:
        raise ValueError('Resolve final build warnings: ' + '\n'.join(errors))
    blg = PAPER/'iclr_draft.blg'
    if blg.exists() and re.search(r'Warning--|error message', blg.read_text(), re.I):
        raise ValueError('Resolve BibTeX warnings/errors')
    manifest = json.loads((PAPER/'official_style_manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if hashlib.sha256((PAPER/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Official style file was changed: ' + name)
    sources = ['iclr_draft.tex','iclr_draft.pdf','theory_appendix.tex',
               'authority_appendix.tex','refs.bib','fig_interface.pdf','fig_video_evidence.pdf',
               'fig_theory_analysis.pdf','fig_results_analysis.pdf',
               'fig_tracking_analysis.pdf','fig_convergence.pdf',
               'official_style_manifest.json',Path(__file__).name]
    receipt = dict(status='passed',initial_submission_main_limit=args.max_main_pages,
                   required_main_pages=args.require_main_pages,
                   main_pages=main_pages,backmatter_first_page=backmatter,total_pdf_pages=total,
                   deferred_main_floats_counted=True,official_style_matches=True,
                   undefined_references_or_citations=False,overfull_boxes=False,
                   bibliography_warnings=False,
                   source_hashes={n:hashlib.sha256((PAPER/n).read_bytes()).hexdigest() for n in sources},
                   policy='https://iclr.cc/Conferences/2027/AuthorGuidelines',
                   limitations='Layout and build checks do not establish scientific acceptance or certify anonymization of a separate code archive.')
    args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','main_pages','backmatter_first_page','total_pdf_pages']},indent=2))


if __name__ == '__main__':
    main()
