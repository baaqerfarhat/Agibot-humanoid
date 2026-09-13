#!/usr/bin/env python3
"""Assemble one complete Markdown study, its PDF index, and source receipts."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BULK = Path('/home/fengze/online_adaptation_research/iclr_2023_2026')
AWARDS = {
    2023: 'https://blog.iclr.cc/2023/03/21/announcing-the-iclr-2023-outstanding-paper-award-recipients/',
    2024: 'https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/',
    2025: 'https://blog.iclr.cc/2025/04/22/announcing-the-outstanding-paper-awards-at-iclr-2025/',
    2026: 'https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/',
}


def esc(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def nested(text, levels=1, drop_title=False):
    if drop_title:
        text = text.split('\n', 1)[1].lstrip('\n')
    return re.sub(r'^(#{1,5}) ', lambda m: '#' * min(6, len(m[1]) + levels) + ' ', text, flags=re.M)


def main():
    rows = [r for y in range(2023, 2027) for r in json.loads((HERE / f'manifest_{y}.json').read_text())]
    receipt = json.loads((HERE / 'corpus_receipt.json').read_text())
    metrics = json.loads((HERE / 'structure_metrics.json').read_text())
    byid = {(m['year'], m['openreview_id']): m for m in metrics}
    verified_statuses = {'downloaded', 'complete', 'downloaded_verified', 'ok'}
    downloaded = [r for r in rows if r.get('download_status') in verified_statuses and r.get('pdf_path')]
    assert receipt['manifest_count'] == len(rows)
    assert receipt['verified_pdf_count'] == len(downloaded), 'Run corpus analyzer after merging downloads'
    assert receipt['duplicate_year_ids'] == 0
    totals = Counter(d for r in rows for d in r['designations'])
    summary = [
        '# ICLR 2023–2026 oral and award paper writing study', '',
        'Collected 2026-09-08. Includes the complete section-by-section guide, all 34 targeted writing profiles, and a paper-by-paper PDF/structure index.', '',
        f'**Downloaded and hash-verified: {len(downloaded)}/{len(rows)} identified papers.** '
        f'The collection includes {totals["outstanding"]} Outstanding Paper Award recipients and '
        f'{totals["honorable_mention"]} honorable mentions, deduplicated within the oral/distinction corpus. '
        f'The indexed PDFs contain {sum(r["pages"] for r in downloaded):,} pages and '
        f'{sum(r["bytes"] for r in downloaded) / 1e9:.2f} GB of PDF bytes. '
        'These totals include references, appendices and alternate public versions.', '',
        f'Bulk PDFs, extracted text and raw source records: [{BULK}]({BULK}/). '
        'The bulk collection stays outside the source repository. '
        'The separate [PDF index](CORPUS_INDEX.md) offers the same download links without the long guide.', '',
        '## Contents', '',
        '- [Collection scope and award records](#collection-scope-and-award-records)',
        '- [Whole-corpus structural measurements](#whole-corpus-structural-measurements)',
        '- [Writing guide and manuscript application](#writing-guide-and-manuscript-application)',
        '- [Year-by-year close readings](#year-by-year-close-readings)',
        '- [Complete paper and PDF index](#complete-paper-and-pdf-index)', '',
        '## Collection scope and award records', '',
        '| Year | Unique collected papers | Scheduled oral events | Additional inclusion | Outstanding | Honorable mention |',
        '|---|---:|---:|---|---:|---:|',
    ]
    for y, scheduled, extra in [(2023, 248, '22 other notable-top-5% papers'), (2024, 86, 'None'),
                                (2025, 213, 'None'), (2026, 223, '1 oral-accepted paper with poster event only')]:
        subset = [r for r in rows if r['year'] == y]
        summary.append(f'| [{y} official catalog](https://iclr.cc/virtual/{y}/events/oral) | {len(subset)} | {scheduled} | {extra} | '
                       f'{sum("outstanding" in r["designations"] for r in subset)} | {sum("honorable_mention" in r["designations"] for r in subset)} |')
    summary += ['',
        'Scope is the **main conference**, excluding workshops, Tiny Papers, blogposts, invited talks and Test-of-Time awards to older publications. '
        'ICLR 2023 used separate acceptance and presentation labels; its scheduled orals include top-25% papers, while the extra top-5% entries are explicitly distinguished. '
        '2026 counts use directly retrieved official catalogs, not an older search-cache event count.', '',
        '**Candidates and finalists:** the official announcements describe larger pools but do not publish complete named lists. '
        '2023 describes 67 initial candidates and approximately 20 intermediate shortlisted papers; 2024 describes 44 and approximately 20; '
        '2025 describes 36 initial candidates; 2026 describes 36 initial candidates and five shortlisted papers, naming only three recognized titles. '
        'We include every named Outstanding/Honorable paper and do not fabricate candidate or finalist identities. '
        'Searches for additional public finalist records did not establish a separate complete main-conference list. '
        'A complete download of privately unnamed pools is therefore not something this collection can certify. '
        + ' '.join(f'[{y} award announcement]({u}).' for y, u in AWARDS.items()), '',
        '**Version and source distinction:** 2024–2026 PDFs come from the official proceedings. '
        'For 2023, OpenReview returned a browser-verification challenge; the collection uses author/arXiv copies and explicitly identified public-archive copies. '
        'Membership and award status come from official ICLR records, not from the archive. '
        'Title/author checks, source URLs, public archive commit and file hashes are retained. '
        'An alternate or archived PDF is not asserted to be byte-identical to the accepted OpenReview revision. '
        'These versions support a writing study but must not be mixed silently when interpreting publication layout or substantive revisions.', '',
        '### Every named award paper', '',
        '| Year | Official distinction | Paper | PDF |', '|---|---|---|---|',
    ]
    for r in rows:
        if not {'outstanding', 'honorable_mention'} & set(r['designations']):
            continue
        status = 'Outstanding' if 'outstanding' in r['designations'] else 'Honorable mention'
        link = f'[local PDF](<{r["pdf_path"]}>)' if r.get('pdf_path') else 'Not downloaded'
        summary.append(f'| {r["year"]} | [{status}]({AWARDS[r["year"]]}) | {esc(r["title"])} | {link} |')
    summary += ['', '## Whole-corpus structural measurements', '',
        'Every indexed PDF is opened and hash-checked; the analyzer extracts page geometry and text. '
        'The following are **conservative automated measurements**, not manually audited figure totals. '
        'Caption anchors are detected before the first References/Bibliography heading; wrapped captions, unnumbered images and unusual layouts can be missed. '
        'Reference-heading page is a physical PDF location, **not the number of main-text pages**. '
        'Some versions put substantive material after references. No bibliography size is inferred from citation regexes.', '',
        '| Year | PDFs | Total PDF pages, median [Q1,Q3] | First reference-heading PDF page, median | Detected figures before references, median [Q1,Q3] | Detected tables before references, median [Q1,Q3] |',
        '|---|---:|---|---:|---|---|',
    ]
    for y in range(2023, 2027):
        s = receipt['summaries'][str(y)]
        def spread(key):
            v = s[key]
            return f'{v["median"]:g} [{v["q25"]:g}, {v["q75"]:g}]' if v else 'undetected'
        rp = s['reference_start_pdf_page']
        summary.append(f'| {y} | {s["pdf_count"]} | {spread("pdf_pages")} | {rp["median"]:g} | {spread("detected_main_figures")} | {spread("detected_main_tables")} |')
    summary += ['',
        'The central implication is variation in how arguments are presented, not an optimum number of figures, tables or pages. '
        '2023 includes preprints and extended treatments; 2024–2026 published versions often use a different allowance from an initial submission. '
        'The precise caption-detection denominators, ranges, per-paper anchors and unresolved extraction fields are in '
        '[structure_metrics.json](structure_metrics.json) and [corpus_receipt.json](corpus_receipt.json). '
        'All papers receive this structural pass; 34 receive targeted qualitative readings. The latter are selected, not a random sample.', '',
        '## Writing guide and manuscript application', '',
        nested((HERE / 'writing_guide_body.md').read_text(), drop_title=True), '',
        '## Year-by-year close readings', '',
    ]
    for y in range(2023, 2027):
        summary += [nested((HERE / f'YEAR_{y}.md').read_text(), levels=2), '']
    index = ['# ICLR 2023–2026 complete paper and PDF index', '',
        'F/T = automatically detected numbered figure/table captions before the first reference heading; '
        'R = physical PDF page of that heading; “?” means undetected. These are structural aids, not manually verified counts or page-limit checks. '
        'Use the source URL and manifest version fields to distinguish proceedings, author preprints and public archived copies.', '']
    for y in range(2023, 2027):
        index += [f'## {y}', '', '| Paper | Public distinction | PDF / source | Pages; F/T; R |', '|---|---|---|---|']
        for r in sorted((x for x in rows if x['year'] == y), key=lambda x: x['title'].lower()):
            m = byid.get((y, r['openreview_id']), {})
            def v(k):
                return '?' if m.get(k) is None else m[k]
            links = f'[PDF](<{r["pdf_path"]}>) · [source]({r["pdf_url"]})' if r.get('pdf_path') else r.get('download_status', 'missing')
            designation = ', '.join(d.replace('_', ' ') for d in r['designations'])
            index.append(f'| {esc(r["title"])} | {designation} | {links} | {r.get("pages", "?")}; {v("detected_main_figures")}/{v("detected_main_tables")}; {v("reference_start_pdf_page")} |')
        index.append('')
    (HERE / 'CORPUS_INDEX.md').write_text('\n'.join(index).rstrip() + '\n')
    summary += ['## Complete paper and PDF index', '', nested('\n'.join(index[2:])), '',
                '## Reproduction and final artifact checks', '',
                'The collectors, per-year manifests, identity checks and corpus analyzer are in this directory. '
                'Run `python paper/style_study/analyze_corpus.py` after downloads are complete; '
                'then run `python paper/style_study/build_report.py` to assemble this report and index. '
                'Existing structural measurements are reused only when the PDF SHA-256 still matches.', '',
                'The manuscript build is checked with `python paper/check_submission_layout.py --require-main-pages 9`. '
                'See [the layout receipt](../submission_layout_receipt.json), [revision notes](../ICLR_REVISION_NOTES.md), '
                'and [figure provenance audit](../FIGURE_AUDIT.md).', '']
    (HERE / 'ICLR_WRITING_STUDY.md').write_text('\n'.join(summary).rstrip() + '\n')
    inputs = [HERE / 'writing_guide_body.md', HERE / 'corpus_receipt.json', HERE / 'structure_metrics.json']
    inputs += [HERE / f'YEAR_{y}.md' for y in range(2023, 2027)]
    inputs += [HERE / f'manifest_{y}.json' for y in range(2023, 2027)]
    outputs = [HERE / 'ICLR_WRITING_STUDY.md', HERE / 'CORPUS_INDEX.md']
    result = dict(papers=len(rows), downloaded=len(downloaded), outstanding=totals['outstanding'],
                  honorable_mentions=totals['honorable_mention'], close_reading_profiles=34,
                  bulk_directory=str(BULK),
                  input_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  output_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs})
    (HERE / 'report_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('papers', 'downloaded', 'outstanding', 'honorable_mentions', 'close_reading_profiles')}, indent=2))


if __name__ == '__main__':
    main()
