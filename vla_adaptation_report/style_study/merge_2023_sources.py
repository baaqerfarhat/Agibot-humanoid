#!/usr/bin/env python3
"""Merge verified public source recoveries without changing corpus membership."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import fitz

from collect_2023 import HERE, ROOT, norm, save_manifest


def main():
    rows = json.loads((HERE / 'manifest_2023.json').read_text())
    mirrors = {r['openreview_id']: r for r in json.loads((HERE / 'manual_sources_2023_mirror.json').read_text())}
    manual = {r['openreview_id']: r for r in json.loads((HERE / 'mirror_identity_checks_2023.json').read_text())}
    primary = {r['openreview_id']: r for letter in ('a', 'b')
               for r in json.loads((HERE / f'manual_sources_2023_{letter}.json').read_text())
               if r['download_status'] in ('complete', 'verified')}
    recovered = []
    for row in rows:
        oid = row['openreview_id']
        if oid not in mirrors:
            assert row['download_status'] == 'downloaded'
            row['source_kind'] = 'arxiv_author_preprint'
            continue
        mirror = mirrors[oid]
        assert mirror['download_status'] == 'verified' or manual.get(oid, {}).get('verified'), oid
        selected = mirror
        kind = 'public_third_party_pdf_archive'
        alt = primary.get(oid)
        if alt:
            # Prefer an author/institution copy only when its text identifies an
            # exact-title ICLR2023 conference paper. Keep all other versions as
            # explicit alternatives rather than silently substituting extensions.
            p = Path(alt.get('local_pdf_path') or alt['pdf_path'])
            assert hashlib.sha256(p.read_bytes()).hexdigest() == alt['sha256']
            with fitz.open(p) as doc:
                lead = doc[0].get_text()
            normalized = norm(lead)
            if ('publishedasaconferencepaperaticlr2023' in normalized
                    and norm(row['title']) in normalized):
                selected = alt
                kind = 'author_or_institution_conference_copy'
            row['alternate_primary_source'] = alt
        row['public_archive_source'] = mirror
        if oid in manual:
            row['public_archive_identity_check'] = manual[oid]
        pdf = Path(selected.get('local_pdf_path') or selected['pdf_path'])
        raw = pdf.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == selected['sha256']
        with fitz.open(pdf) as doc:
            texts = [p.get_text(sort=True) for p in doc]
            pages = len(doc)
        txt = ROOT / 'text' / (oid + '.txt')
        txt.write_text('\n\f\n'.join(texts))
        row.update(pdf_path=str(pdf), text_path=str(txt), pdf_url=selected['pdf_url'],
                   sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), pages=pages,
                   source_kind=kind, download_status='downloaded',
                   version_note=selected['version_note'], identity_check=selected['identity_evidence'],
                   text_sha256=hashlib.sha256(txt.read_bytes()).hexdigest())
        row.pop('error', None)
        recovered.append(dict(openreview_id=oid, source_kind=kind, sha256=row['sha256']))
    assert len(rows) == 270 and len({r['openreview_id'] for r in rows}) == 270
    assert all(r['download_status'] == 'downloaded' for r in rows)
    save_manifest(rows)
    result = dict(unique_papers=len(rows), downloaded=len(rows), recovered_papers=len(recovered),
                  missing=0, source_counts=dict(Counter(r['source_kind'] for r in rows)),
                  all_pdf_bytes=sum(r['bytes'] for r in rows), all_pdf_pages=sum(r['pages'] for r in rows),
                  recovered_records=recovered,
                  manifest_sha256=hashlib.sha256((HERE / 'manifest_2023.json').read_bytes()).hexdigest(),
                  limitations='2023 public alternatives are not asserted byte-identical to the accepted OpenReview revisions; see each record.')
    (HERE / 'collection_2023_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'recovered_records'}, indent=2))


if __name__ == '__main__':
    main()
