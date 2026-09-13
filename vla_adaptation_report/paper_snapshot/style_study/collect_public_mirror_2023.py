#!/usr/bin/env python3
"""Retrieve publicly archived 2023 PDFs with explicit third-party provenance.

Official ICLR metadata supplies membership, title, and author identity. The
archive supplies bytes, not authoritative conference revision metadata.
No requests to, or workarounds for, OpenReview's access challenge are used.
"""
import concurrent.futures as cf
import difflib
import hashlib
import json
import re
from datetime import datetime, timezone
from urllib.parse import quote

import fitz
import requests

from collect_2023 import HERE, ROOT, cached, norm

REPO = 'Chelsea707/ICLR_OCR'
REV = 'ce5b4a0f363cbefda8004455b2a46dc04da78e7b'
RAW = ROOT / 'raw' / 'public_archive'


def directories():
    path = RAW / 'directories.json'
    if path.exists():
        return json.loads(path.read_text())
    url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/2023?limit=1000'
    entries = []
    while url:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        entries.extend(response.json())
        url = response.links.get('next', {}).get('url')
    path.write_text(json.dumps(entries, indent=2) + '\n')
    return entries


def fetch(row, dirs):
    oid = row['openreview_id']
    out = dict(openreview_id=oid, title=row['title'], authors=row['authors'],
               download_status='unresolved', source_kind='public_third_party_pdf_archive',
               archive_repo=REPO, archive_revision=REV,
               version_note='Public archived PDF; exact accepted/review revision and byte identity to OpenReview are not independently established.',
               retrieved_utc=datetime.now(timezone.utc).isoformat())
    wanted = norm(row['title'])
    matches = [d for d in dirs if norm(d['path'].split('/', 1)[1]) == wanted]
    if len(matches) != 1:
        ranked = sorted(dirs, key=lambda d: difflib.SequenceMatcher(None, norm(d['path'].split('/', 1)[1]), wanted).ratio(), reverse=True)[:3]
        out['directory_candidates'] = [dict(path=d['path'], similarity=difflib.SequenceMatcher(None, norm(d['path'].split('/', 1)[1]), wanted).ratio()) for d in ranked]
        # Only typography differences in the title are allowed automatically.
        if ranked and out['directory_candidates'][0]['similarity'] >= .98:
            matches = ranked[:1]
        else:
            return out
    directory = matches[0]['path']
    files_url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/' + quote(directory, safe='/')
    files = json.loads(cached(files_url, RAW / (oid + '_files.json')))
    pdfs = [f for f in files if f['path'].endswith('_origin.pdf')]
    if len(pdfs) != 1:
        out['error'] = 'Expected exactly one archived origin PDF'
        return out
    entry = pdfs[0]
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/' + quote(entry['path'], safe='/')
    path = ROOT / 'pdfs_archive' / (oid + '.pdf')
    txt = ROOT / 'text_archive' / (oid + '.txt')
    try:
        raw = cached(url, path)
        assert raw.startswith(b'%PDF'), 'Not a PDF'
        sha = hashlib.sha256(raw).hexdigest()
        expected = entry.get('lfs', {}).get('oid')
        if expected:
            assert sha == expected, 'Public archive LFS hash mismatch'
        with fitz.open(path) as doc:
            texts = [p.get_text(sort=True) for p in doc]
            first = '\n'.join(texts[:2])
            # Check the title before the first abstract, avoiding a title merely
            # mentioned as related work or in a bibliography.
            lead = re.split(r'abstract', first, maxsplit=1, flags=re.I)[0]
            lead_norm = norm(lead)
            title_match = wanted in lead_norm
            surnames = {norm(a.split()[-1]) for a in row['authors'] if len(norm(a.split()[-1])) >= 3}
            present = sorted(n for n in surnames if n in lead_norm)
            abstract_words = set(re.findall(r'[a-z]{4,}', row.get('abstract', '').lower()))
            first_words = set(re.findall(r'[a-z]{4,}', first.lower()))
            overlap = len(abstract_words & first_words) / max(1, len(abstract_words))
            identity = title_match and (bool(present) or overlap >= .65)
            txt.write_text('\n\f\n'.join(texts))
            out.update(pdf_url=url, source_url=files_url, local_pdf_path=str(path),
                local_text_path=str(txt), sha256=sha, bytes=len(raw), pages=len(doc),
                archive_file_sha256=expected, text_sha256=hashlib.sha256(txt.read_bytes()).hexdigest(),
                title_check_before_abstract=title_match, author_surnames_present=present,
                abstract_word_recall=round(overlap, 4), title_preview=lead[:2000],
                identity_evidence='Compared PDF title before abstract and author surnames (or abstract overlap for anonymous versions) to official ICLR metadata.',
                download_status='verified' if identity else 'downloaded_needs_manual_identity_check')
    except Exception as e:
        out['error'] = str(e)
    return out


def main():
    for p in (RAW, ROOT / 'pdfs_archive', ROOT / 'text_archive'):
        p.mkdir(parents=True, exist_ok=True)
    rows = json.loads((HERE / 'manifest_2023.json').read_text())
    selected = [r for r in rows if r['download_status'] != 'downloaded']
    dirs = [d for d in directories() if d['type'] == 'directory']
    print('Archive directories', len(dirs), 'requested papers', len(selected), flush=True)
    records = []
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        for out in pool.map(lambda row: fetch(row, dirs), selected):
            records.append(out)
            (HERE / 'manual_sources_2023_mirror.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
            print(len(records), out['title'], out['download_status'], flush=True)


if __name__ == '__main__':
    main()
