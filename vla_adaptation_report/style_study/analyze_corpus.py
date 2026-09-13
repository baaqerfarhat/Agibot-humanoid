#!/usr/bin/env python3
"""Reproducible structural inventory, not a claim of automated quality grading.

PDF reading order and citation styles vary. Measurements with ambiguous
boundaries are left missing; bibliography counts are deliberately not inferred
from years or citation callouts. The companion qualitative reports cover them.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent


def records(path):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    for key in ('papers', 'records', 'results'):
        if isinstance(data.get(key), list):
            return data[key]
    raise ValueError(f'Unknown manifest structure: {path}')


def inventory(row):
    path = Path(row['pdf_path'])
    with fitz.open(path) as doc:
        text_pages = [p.get_text(sort=True) for p in doc]
        lines = []
        for page_index, page in enumerate(doc):
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    spans = line.get('spans', [])
                    text = ''.join(s['text'] for s in spans).strip()
                    if text:
                        lines.append(dict(page=page_index + 1, y=line['bbox'][1], x=line['bbox'][0],
                                          text=text, size=max(s['size'] for s in spans),
                                          bold=any(s['flags'] & 16 for s in spans)))
        lines.sort(key=lambda x: (x['page'], x['y'], x['x']))
        refs = next((l for l in lines if re.sub(r'\s', '', l['text']).upper() in ('REFERENCES', 'BIBLIOGRAPHY')), None)
        main = [l for l in lines if refs is None or (l['page'], l['y']) < (refs['page'], refs['y'])]
        figures, tables = {}, {}
        for l in main:
            for label, out in [('Figure', figures), ('Table', tables)]:
                m = re.match(r'^' + label + r'\s+(\d+)\s*[:.](?:\s|$)', l['text'], re.I)
                if m:
                    out.setdefault(m[1], l['page'])
        # Conservative section-head candidates, reported so they can be audited.
        heads = [dict(page=l['page'], text=l['text']) for l in main
                 if re.match(r'^\d+(?:\.\d+)*\s+[A-Z]', l['text']) and l['bold'] and len(l['text']) < 140]
        raw = path.read_bytes()
        return dict(year=row['year'], title=row['title'], openreview_id=row.get('openreview_id'),
                    designations=row.get('designations', []), pdf_path=str(path),
                    sha256=hashlib.sha256(raw).hexdigest(), pdf_pages=len(doc),
                    reference_start_pdf_page=refs['page'] if refs else None,
                    reference_heading_y=round(refs['y'], 2) if refs else None,
                    detected_main_figures=len(figures) if refs else None,
                    detected_main_tables=len(tables) if refs else None,
                    figure_caption_pages=figures, table_caption_pages=tables,
                    section_heading_candidates=heads,
                    extracted_words=sum(len(re.findall(r'\b[\w-]+\b', p)) for p in text_pages),
                    full_text_searchable=all(len(p.strip()) > 50 for p in text_pages),
                    scope='Caption anchors before first References heading; not manually verified counts, page-limit determinations, or a writing-quality score.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    rows = []
    for year in range(2023, 2027):
        rows += records(HERE / f'manifest_{year}.json')
    out, errors = [], []
    cache_path = HERE / 'structure_metrics.json'
    prior = json.loads(cache_path.read_text()) if cache_path.exists() else []
    cache = {(r['year'], r.get('openreview_id'), r['sha256']): r for r in prior}
    for index, row in enumerate(rows, 1):
        try:
            path = Path(row['pdf_path'])
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            item = cache.get((row['year'], row.get('openreview_id'), actual_hash))
            if item is None:
                item = inventory(row)
            if row.get('sha256'):
                assert item['sha256'] == row['sha256'], 'Manifest hash mismatch'
            out.append(item)
        except Exception as e:
            errors.append(dict(year=row['year'], title=row['title'], error=str(e)))
        if index % 50 == 0:
            print(f'Structure checked {index}/{len(rows)}; parsed/cache-valid {len(out)}', flush=True)
    (HERE / 'structure_metrics.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    summaries = {}
    for year in range(2023, 2027):
        subset = [x for x in out if x['year'] == year]
        d = dict(pdf_count=len(subset), public_designations=dict(Counter(k for x in subset for k in x['designations'])))
        for key in ('pdf_pages', 'reference_start_pdf_page', 'detected_main_figures', 'detected_main_tables'):
            values = [x[key] for x in subset if x[key] is not None]
            d[key] = dict(n=len(values), median=float(np.median(values)), q25=float(np.percentile(values, 25)),
                          q75=float(np.percentile(values, 75)), min=min(values), max=max(values)) if values else None
        summaries[year] = d
    receipt = dict(manifest_count=len(rows), verified_pdf_count=len(out), errors=errors,
                   duplicate_year_ids=len(rows) - len({(r['year'], r.get('openreview_id')) for r in rows}),
                   summaries=summaries,
                   metric_limitations='PDF layouts differ. Captions are conservative regex detections; reference start is not main-page count. No inferred bibliography counts or acceptance causality.',
                   source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('manifest_*.json')})
    (HERE / 'corpus_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if errors and not args.allow_incomplete:
        raise SystemExit('Incomplete corpus: inspect corpus_receipt.json')


if __name__ == '__main__':
    main()
