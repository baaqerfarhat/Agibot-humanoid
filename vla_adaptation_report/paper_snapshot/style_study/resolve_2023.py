#!/usr/bin/env python3
"""Broaden cached arXiv queries for unmatched 2023 titles; retain candidates.

Without --apply this only records possible sources for manual inspection.
With --apply, only the original exact title/author rule is applied automatically.
"""
import argparse
import concurrent.futures as cf
import difflib
import json

from collect_2023 import HERE, ROOT, associate, download, norm, query_arxiv, save_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    rows = json.loads((HERE / 'manifest_2023.json').read_text())
    missing = [r for r in rows if not r.get('pdf_url')]
    entries = []
    for i in range(0, len(missing), 6):
        chunk = missing[i:i + 6]
        entries.extend(query_arxiv([' '.join(r['title'].split()[:5]) for r in chunk], 'arxiv_broad_%03d' % i))
        print('Broader queries', min(i + 6, len(missing)), '/', len(missing), flush=True)
    entries = list({e['arxiv_id']: e for e in entries}.values())
    candidates = []
    for r in missing:
        ranked = sorted(entries, key=lambda e: difflib.SequenceMatcher(None, norm(e['title']), norm(r['title'])).ratio(), reverse=True)[:3]
        candidates.append(dict(openreview_id=r['openreview_id'], title=r['title'], authors=r['authors'],
                               candidates=[dict(e, title_similarity=round(difflib.SequenceMatcher(None, norm(e['title']), norm(r['title'])).ratio(), 4)) for e in ranked]))
        if args.apply:
            associate(r, entries)
    (HERE / 'source_candidates_2023.json').write_text(json.dumps(candidates, indent=2, ensure_ascii=False) + '\n')
    if args.apply:
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            for r in pool.map(download, missing):
                print(r['title'], r['download_status'], flush=True)
        save_manifest(rows)


if __name__ == '__main__':
    main()
