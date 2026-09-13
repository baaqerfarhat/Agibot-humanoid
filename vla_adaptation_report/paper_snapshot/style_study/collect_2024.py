#!/usr/bin/env python3
"""Collect every official ICLR 2024 oral and publicly named paper award.

Two request workers maximum. Uses the public ICLR proceedings; OpenReview's
browser-verification barrier is recorded, not bypassed. Metadata/text extraction
is not equivalent to a substantive reading. Re-running resumes valid files.
"""
from __future__ import annotations

import collections
import concurrent.futures
import datetime as dt
import hashlib
import json
import pathlib
import re
import time
import unicodedata
from urllib.parse import parse_qs, urljoin, urlparse

import fitz
import requests
from bs4 import BeautifulSoup

HERE = pathlib.Path(__file__).resolve().parent
BULK = pathlib.Path('/home/fengze/online_adaptation_research/iclr_2023_2026/2024')
URLS = {
    'oral_catalog.html': 'https://iclr.cc/virtual/2024/events/oral',
    'awards.html': 'https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/',
    'paper_catalog.json': 'https://iclr.cc/static/virtual/data/iclr-2024-orals-posters.json',
    'proceedings_index.html': 'https://proceedings.iclr.cc/paper_files/paper/2024',
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalize(title):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', title)
                  .encode('ascii', 'ignore').decode().lower())


def fetch(url, destination):
    if destination.exists() and destination.stat().st_size:
        return destination.read_bytes(), 'cached', None
    last = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(20, 150),
                                    headers={'User-Agent': 'ICLR-writing-corpus/1.0 (public academic PDFs)'})
            if response.status_code in (401, 403, 404):
                raise RuntimeError(f'HTTP {response.status_code}; no access-control retry')
            response.raise_for_status()
            temporary = destination.with_suffix(destination.suffix + '.partial')
            temporary.write_bytes(response.content)
            temporary.replace(destination)
            return response.content, 'downloaded', now()
        except RuntimeError:
            raise
        except (requests.RequestException, OSError) as exc:
            last = str(exc)
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(last)


def metadata(pdf, text_path):
    if not pdf.read_bytes().startswith(b'%PDF'):
        raise ValueError('Response is not a PDF')
    with fitz.open(pdf) as doc:
        if not len(doc) or doc.is_encrypted:
            raise ValueError('PDF is empty or encrypted')
        page_texts = [p.get_text(sort=True) for p in doc]
        text_path.write_text('\n\n'.join(
            f'===== PDF PAGE {i + 1} / {len(doc)} =====\n{text}'
            for i, text in enumerate(page_texts)))
        figures, tables, references, appendices, headings = [], [], [], [], []
        fonts = collections.Counter()
        for n, (page, text) in enumerate(zip(doc, page_texts), 1):
            lines = text.splitlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if re.fullmatch(r'REFERENCES|References|BIBLIOGRAPHY|Bibliography', line):
                    references.append(n)
                if re.fullmatch(r'APPENDIX|APPENDICES|Appendix|Appendices|SUPPLEMENTARY MATERIAL', line):
                    appendices.append(n)
                if re.fullmatch(r'\d+(?:\.\d+)*', line) and i + 1 < len(lines):
                    heading = lines[i + 1].strip()
                    if 2 < len(heading) < 110 and heading.upper() == heading:
                        headings.append({'page': n, 'number': line, 'heading': heading})
                for pattern, output in ((r'^Figure\s+(\d+)\s*:', figures),
                                        (r'^Table\s+(\d+)\s*:', tables)):
                    match = re.match(pattern, line, flags=re.I)
                    if match:
                        output.append({'page': n, 'number': int(match.group(1)),
                                       'caption_start': ' '.join(lines[i:i+7])[:550]})
            if n <= 9:
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines', []):
                        for span in line.get('spans', []):
                            fonts[(span['font'], round(span['size'], 1))] += len(span['text'])
        return {
            'pages': len(doc), 'text_characters': sum(map(len, page_texts)),
            'pdf_metadata': doc.metadata,
            'first_page_text': page_texts[0][:1800],
            'reference_heading_pages_heuristic': sorted(set(references)),
            'explicit_appendix_heading_pages_heuristic': sorted(set(appendices)),
            'section_headings_heuristic': headings,
            'figure_captions_heuristic': figures, 'table_captions_heuristic': tables,
            'top_fonts_first_9_pages': [{'font': k[0], 'points': k[1], 'characters': v}
                                       for k, v in fonts.most_common(6)],
            'measurement_limit': 'Automated text/layout observations; heading/caption detection is heuristic and full PDF pages include references/appendices.',
        }


def build_records():
    raw = BULK / 'raw'
    catalogs = {}
    for filename, url in URLS.items():
        data, status, when = fetch(url, raw / filename)
        catalogs[filename] = {'url': url, 'path': str(raw / filename), 'sha256': sha(data),
                              'bytes': len(data), 'status': status, 'retrieved_at_utc': when}
    soup = BeautifulSoup((raw / 'oral_catalog.html').read_text(), 'html.parser')
    cards = soup.select('.event-card')
    assert len(cards) == 86 and all(c['data-event-type'] == 'Oral' for c in cards)
    event_ids = {int(c['data-event-id']) for c in cards}
    static = json.loads((raw / 'paper_catalog.json').read_text())['results']
    by_event = {r['id']: r for r in static}
    oral_records = [by_event[i] for i in sorted(event_ids)]
    assert all(r['eventtype'] == 'Oral' and r['decision'].lower() == 'accept (oral)' for r in oral_records)

    awards = {}
    award_soup = BeautifulSoup((raw / 'awards.html').read_text(), 'html.parser').select_one('.entry-content')
    designation = None
    for node in award_soup.find_all(['h2', 'a']):
        if node.name == 'h2':
            designation = {'Award Winners': 'outstanding', 'Honorable mentions': 'honorable_mention'}.get(node.get_text(' ', strip=True))
        elif designation and 'openreview.net/forum?' in node.get('href', ''):
            oid = parse_qs(urlparse(node['href']).query)['id'][0]
            awards[oid] = designation
    assert collections.Counter(awards.values()) == {'outstanding': 5, 'honorable_mention': 11}

    proceedings = collections.defaultdict(list)
    ps = BeautifulSoup((raw / 'proceedings_index.html').read_text(), 'html.parser')
    for a in ps.find_all('a', href=True):
        if '-Abstract-Conference.html' in a['href']:
            proceedings[normalize(a.get_text(' ', strip=True))].append(urljoin(URLS['proceedings_index.html'], a['href']))
    records = {}
    for record in oral_records:
        oid = parse_qs(urlparse(record['paper_url']).query)['id'][0]
        if oid in records:
            raise ValueError(f'Duplicate oral OpenReview ID: {oid}')
        abstract_urls = proceedings[normalize(record['name'])]
        if len(abstract_urls) != 1:
            raise ValueError(f'No unique official proceedings title match: {record["name"]}')
        designations = ['oral'] + ([awards[oid]] if oid in awards else [])
        records[oid] = {
            'year': 2024, 'title': record['name'],
            'authors': [a['fullname'] for a in record['authors']],
            'event_url': urljoin('https://iclr.cc', record['virtualsite_url']),
            'openreview_id': oid, 'openreview_forum_url': record['paper_url'],
            'openreview_pdf_url': f'https://openreview.net/pdf?id={oid}',
            'proceedings_abstract_url': abstract_urls[0],
            'pdf_url': None, 'designations': designations,
            'award_source_urls': [URLS['awards.html']] if oid in awards else [],
            'designation_source_urls': [URLS['oral_catalog.html'], URLS['paper_catalog.json']],
            'pdf_path': str(BULK / 'pdfs' / f'{oid}.pdf'),
            'text_path': str(BULK / 'text' / f'{oid}.txt'),
            'sha256': None, 'bytes': None, 'pages': None, 'download_status': 'pending',
            'version_label': 'Public official ICLR 2024 proceedings PDF retrieved at collection time; not assumed identical to original submission or latest OpenReview revision.',
            'reading_status': 'automated_metadata_and_text_only',
        }
    if set(awards) - set(records):
        raise ValueError('Award paper outside oral collection must be added explicitly')
    return records, catalogs


def collect_one(record):
    oid = record['openreview_id']
    try:
        html, _, _ = fetch(record['proceedings_abstract_url'], BULK / 'raw' / f'proceedings_{oid}.html')
        soup = BeautifulSoup(html, 'html.parser')
        links = [urljoin(record['proceedings_abstract_url'], a['href'])
                 for a in soup.find_all('a', href=True)
                 if a['href'].endswith('-Paper-Conference.pdf')]
        if len(links) != 1:
            raise ValueError('Official abstract page lacks unique Conference paper link')
        record['pdf_url'] = links[0]
        data, source_status, downloaded_at = fetch(links[0], pathlib.Path(record['pdf_path']))
        extracted = metadata(pathlib.Path(record['pdf_path']), pathlib.Path(record['text_path']))
        record.update(sha256=sha(data), bytes=len(data), pages=extracted['pages'],
                      download_status='complete', download_source_status=source_status,
                      retrieved_at_utc=downloaded_at or now(), verified_at_utc=now(),
                      text_sha256=sha(pathlib.Path(record['text_path']).read_bytes()))
        meta_path = BULK / 'text' / f'{oid}.metadata.json'
        meta_path.write_text(json.dumps(extracted, indent=2, ensure_ascii=False) + '\n')
        record['metadata_path'] = str(meta_path)
        record['metadata_sha256'] = sha(meta_path.read_bytes())
    except Exception as exc:
        record['download_status'] = 'error'
        record['error'] = f'{type(exc).__name__}: {exc}'
    return record


def save(records, catalogs, finished=False):
    data = sorted(records.values(), key=lambda r: r['title'].casefold())
    (HERE / 'manifest_2024.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    summary = {
        'year': 2024, 'collection_started_at_utc': STARTED, 'updated_at_utc': now(),
        'collection_finished': finished, 'catalogs': catalogs,
        'counts': {'unique_papers': len(data), 'oral_event_cards': 86,
                   'outstanding': 5, 'honorable_mention': 11,
                   'complete': sum(r['download_status'] == 'complete' for r in data),
                   'errors': sum(r['download_status'] == 'error' for r in data)},
        'max_concurrent_requests': 2,
        'openreview_access': 'Direct PDF/API and browser-tool probes returned browser-verification403; public official proceedings used instead, without challenge bypass.',
        'count_resolution': '86 DOM event-card Oral elements and86 unique oral event URLs; title/detail links occur twice. Broad URL regex can include the unrelated navigation/login poster URL.',
        'candidate_shortlist_scope': 'Awards announcement reports initial44 candidates and roughly20 shortlisted, but names only5 winners and11 honorable mentions. No complete public44/20 name list is supplied; no extra candidate/finalist designations inferred.',
        'script_sha256': sha(pathlib.Path(__file__).read_bytes()),
    }
    (HERE / 'collection_2024.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')


STARTED = now()
if __name__ == '__main__':
    for directory in ('raw', 'pdfs', 'text'):
        (BULK / directory).mkdir(parents=True, exist_ok=True)
    records, catalogs = build_records()
    save(records, catalogs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(collect_one, r) for r in records.values()]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            records[record['openreview_id']] = record
            save(records, catalogs)
            print(f'{i:02d}/{len(records)} {record["download_status"]} {record["openreview_id"]} '
                  f'{record.get("pages")}p {record["title"]}', flush=True)
    save(records, catalogs, finished=True)
