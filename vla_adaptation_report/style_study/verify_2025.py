"""Independently recheck the downloaded 2025 oral archive without network calls."""
from pathlib import Path
import datetime
import hashlib
import json
from collections import Counter
import fitz

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    rows = json.loads((HERE / 'manifest_2025.json').read_text())
    catalog = json.loads((HERE / 'catalog_2025.json').read_text())
    original = json.loads(Path(catalog['sources']['iclr-2025-orals-posters.json']['path']).read_text())['results']
    oral_ids = {r['id'] for r in original if r['eventtype'] == 'Oral'}
    assert len(rows) == len({r['openreview_id'] for r in rows}) == 213
    assert {r['openreview_mapping']['oral_event_id'] for r in rows} == oral_ids
    assert all(r['designations'][0] == 'oral' for r in rows)
    named = {r['openreview_id']: r['designations'][1:] for r in rows if len(r['designations']) > 1}
    assert named == {'HvSytvg3Jh':['outstanding'], 'tPNHOoZFl9':['outstanding'], '6Mxhg9PtDE':['outstanding'], 'HD6bWcj87Y':['honorable_mention'], 'Ha6RTeWMd0':['honorable_mention'], 'vo9t20wsmd':['honorable_mention']}
    verified = []
    for r in rows:
        assert r['download_status'] == 'downloaded_verified'
        for path, expected in [(r['pdf_path'], r['sha256']), (r['text_path'], r['text_sha256']), (r['proceedings_page_path'], r['proceedings_page_sha256'])]:
            assert sha(path) == expected, path
        assert Path(r['pdf_path']).stat().st_size == r['bytes']
        with fitz.open(r['pdf_path']) as doc:
            assert not doc.is_encrypted and len(doc) == r['pages']
            assert all(doc[i].get_text().strip() for i in range(len(doc)))
        assert r['pdf_url'].startswith('https://proceedings.iclr.cc/paper_files/paper/2025/file/')
        assert len(r['page_word_counts']) == r['pages']
        verified.append({'openreview_id':r['openreview_id'], 'pdf_sha256':r['sha256'], 'text_sha256':r['text_sha256'], 'pages':r['pages']})
    for source in catalog['sources'].values():
        assert sha(source['path']) == source['sha256']
    assert sha(HERE/'collect_2025.py') == catalog['script_sha256']
    notes=json.loads((HERE/'reading_notes_2025.json').read_text())
    assert notes['manifest_sha256'] == sha(HERE/'manifest_2025.json')
    for r in notes['papers']:
        assert sha(r['preview_path']) == r['preview_sha256']
    result={'year':2025,'complete':True,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'unique_papers':len(rows),'zero_missing_pdfs':True,'named_awards':named,
            'pdf_bytes':sum(r['bytes'] for r in rows),'pdf_pages':sum(r['pages'] for r in rows),
            'bindings':{p:sha(HERE/p) for p in ['manifest_2025.json','catalog_2025.json','collection_2025_receipt.json','collect_2025.py','reading_notes_2025.json','YEAR_2025.md','verify_2025.py']},
            'checks':'Every PDF/text/publication HTML hash; readable, nonempty text pages; official catalog and proceedings source bindings; unique oral event/OpenReview coverage; all six awards; preview hashes.',
            'verified_papers':verified}
    (HERE/'verification_2025.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['bindings','verified_papers']}))

if __name__ == '__main__':
    main()
