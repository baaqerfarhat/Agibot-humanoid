#!/usr/bin/env python3
"""Reconcile official ICLR 2023 designations; retrieve public arXiv versions.

OpenReview currently requires browser verification. This collector does not
attempt that challenge. Alternate versions are explicitly labeled and matched
by title plus author surname, with raw official and arXiv records retained.
"""
import concurrent.futures as cf
import hashlib
import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import fitz
import requests
from bs4 import BeautifulSoup

ROOT = Path('/home/fengze/online_adaptation_research/iclr_2023_2026/2023')
HERE = Path(__file__).resolve().parent
AWARDS = 'https://blog.iclr.cc/2023/03/21/announcing-the-iclr-2023-outstanding-paper-award-recipients/'
CATALOG = 'https://iclr.cc/static/virtual/data/iclr-2023-orals-posters.json'
NS = {'a': 'http://www.w3.org/2005/Atom'}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower())


def save_manifest(rows):
    (HERE / 'manifest_2023.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')


def cached(url, path, params=None):
    if path.exists():
        return path.read_bytes()
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
            return r.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(4 * 2 ** attempt)


def parse_arxiv(raw):
    entries = []
    for e in ET.fromstring(raw).findall('a:entry', NS):
        title = ' '.join(e.findtext('a:title', namespaces=NS).split())
        aid = e.findtext('a:id', namespaces=NS).split('/abs/')[-1]
        if not re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})v\d+', aid):
            continue
        entries.append(dict(title=title, arxiv_id=aid,
            authors=[a.findtext('a:name', namespaces=NS) for a in e.findall('a:author', NS)],
            updated=e.findtext('a:updated', namespaces=NS),
            published=e.findtext('a:published', namespaces=NS)))
    return entries


def build():
    for p in ('raw', 'pdfs', 'text', 'events'):
        (ROOT / p).mkdir(parents=True, exist_ok=True)
    raw = cached(CATALOG, ROOT / 'raw/catalog.json')
    cached('https://iclr.cc/virtual/2023/events/oral', ROOT / 'raw/oral.html')
    data = json.loads(raw)['results']
    byid = {x['id']: x for x in data}
    posters = [x for x in data if x['eventtype'] == 'Poster' and x.get('paper_url')]
    bytitle = {norm(x['name']): x for x in posters}
    rows = {}
    for x in data:
        if x['eventtype'] != 'Oral' and x['decision'] != 'Accept: notable-top-5%':
            continue
        related = [byid[i] for i in x.get('related_events_ids', []) if i in byid and byid[i].get('paper_url')]
        p = x if x.get('paper_url') else next((p for p in related if norm(p['name']) == norm(x['name'])), bytitle.get(norm(x['name'])))
        assert p and norm(p['name']) == norm(x['name']), x['name']
        oid = parse_qs(urlparse(p['paper_url']).query)['id'][0]
        row = rows.setdefault(oid, dict(year=2023, title=p['name'], authors=[a['fullname'] for a in p['authors']],
            event_url='https://iclr.cc' + p['virtualsite_url'], openreview_id=oid,
            openreview_url=p['paper_url'], requested_pdf_url='https://openreview.net/pdf?id=' + oid,
            designations=[], award_source_urls=[], catalog_url=CATALOG,
            decision=p['decision'], abstract=p['abstract'], download_status='pending',
            openreview_download_status='403 browser verification; no bypass attempted',
            retrieved_at=datetime.now(timezone.utc).isoformat()))
        if x['eventtype'] == 'Oral':
            row['event_url'] = 'https://iclr.cc' + x['virtualsite_url']
            row['designations'].append('oral')
        if x['decision'] == 'Accept: notable-top-5%':
            row['designations'].append('notable_top_5_percent')
        if x['decision'] == 'Accept: notable-top-25%':
            row['designations'].append('notable_top_25_percent')
    blog = BeautifulSoup(cached(AWARDS, ROOT / 'raw/awards.html'), 'html.parser')
    category = None
    for tag in blog.find_all(['h2', 'a']):
        if tag.name == 'h2':
            t = tag.get_text(' ', strip=True)
            category = 'honorable_mention' if 'Honorable' in t else ('outstanding' if t == 'Outstanding Paper Awards' else None)
        elif category and 'openreview.net/forum?' in tag.get('href', ''):
            oid = parse_qs(urlparse(tag['href']).query)['id'][0]
            assert oid in rows, (oid, tag.get_text())
            rows[oid]['designations'].append(category)
            rows[oid]['award_source_urls'].append(AWARDS)
    result = sorted(rows.values(), key=lambda x: x['title'].lower())
    for x in result:
        x['designations'] = sorted(set(x['designations']))
    oldpath = HERE / 'manifest_2023.json'
    old = {x['openreview_id']: x for x in json.loads(oldpath.read_text())} if oldpath.exists() else {}
    for x in result:
        prior = old.get(x['openreview_id'], {})
        if prior.get('download_status') == 'downloaded':
            x.update(prior)
    return result


def query_arxiv(titles, label):
    path = ROOT / 'raw' / (label + '.xml')
    exists = path.exists()
    query = ' OR '.join('ti:"' + re.sub(r'[^\w\s-]', ' ', t) + '"' for t in titles)
    raw = cached('https://export.arxiv.org/api/query', path, {'search_query': query, 'max_results': 100})
    if not exists:
        time.sleep(3.2)  # arXiv API etiquette; process runs asynchronously.
    return parse_arxiv(raw)


def associate(row, entries):
    hits = [e for e in entries if norm(e['title']) == norm(row['title'])]
    surnames = {norm(a.split()[-1]) for a in row['authors']}
    hits = [e for e in hits if surnames & {norm(a.split()[-1]) for a in e['authors']}]
    if len(hits) == 1:
        row['arxiv_record'] = hits[0]
        row['pdf_url'] = 'https://arxiv.org/pdf/' + hits[0]['arxiv_id']
        row['version_note'] = 'Public arXiv version identified in arxiv_record; not verified identical to accepted OpenReview PDF.'
        row['identity_check'] = 'Exact normalized official/arXiv title and at least one matching author surname.'
        return True
    return False


def download(row):
    if row.get('download_status') == 'downloaded':
        return row
    if not row.get('pdf_url'):
        row['download_status'] = 'unresolved_public_pdf'
        return row
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', row['title']).strip('_')[:110]
    pdf = ROOT / 'pdfs' / (row['openreview_id'] + '__' + slug + '.pdf')
    txt = ROOT / 'text' / (row['openreview_id'] + '.txt')
    try:
        raw = cached(row['pdf_url'], pdf)
        assert raw.startswith(b'%PDF'), 'Response is not a PDF'
        with fitz.open(pdf) as doc:
            assert len(doc) > 0
            pages = len(doc)
            txt.write_text('\n\f\n'.join(p.get_text(sort=True) for p in doc))
        row.update(pdf_path=str(pdf), text_path=str(txt), sha256=hashlib.sha256(raw).hexdigest(),
                   bytes=len(raw), pages=pages, download_status='downloaded')
        row.pop('error', None)
    except Exception as e:
        row.update(download_status='download_failed', error=str(e))
    return row


def main():
    rows = build()
    print('Official unique corpus', len(rows), 'in-person orals', sum('oral' in x['designations'] for x in rows), flush=True)
    entries = []
    for i in range(0, len(rows), 8):
        entries.extend(query_arxiv([x['title'] for x in rows[i:i+8]], 'arxiv_batch_%03d' % i))
        print('arXiv title queries', min(i + 8, len(rows)), 'of', len(rows), flush=True)
    # One public record can appear in multiple query batches.
    entries = list({e['arxiv_id']: e for e in entries}.values())
    for row in rows:
        if row.get('download_status') != 'downloaded':
            associate(row, entries)
    save_manifest(rows)
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        fs = {pool.submit(download, x): x for x in rows}
        for i, f in enumerate(cf.as_completed(fs), 1):
            f.result()
            if i % 10 == 0 or i == len(rows):
                save_manifest(rows)
                print('PDF checked', i, 'downloaded', sum(x['download_status'] == 'downloaded' for x in rows), flush=True)
    save_manifest(rows)


if __name__ == '__main__':
    main()
