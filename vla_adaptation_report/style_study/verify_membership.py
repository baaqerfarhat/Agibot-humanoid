#!/usr/bin/env python3
"""Read-only, offline membership reconciliation independent of the collectors.

This checks public scope, not PDF availability or version equivalence. The 2023
manifest may acquire new download metadata without changing its membership hash.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
CORPUS = Path('/home/fengze/online_adaptation_research/iclr_2023_2026')
PATHS = {
    2023: ('raw/catalog.json', 'raw/oral.html', 'raw/awards.html'),
    2024: ('raw/paper_catalog.json', 'raw/oral_catalog.html', 'raw/awards.html'),
    2025: ('catalogs/iclr-2025-orals-posters.json', 'catalogs/oral.html', 'catalogs/awards.html'),
    2026: ('raw/orals_posters.json', 'raw/orals.html', 'raw/awards.html'),
}
AWARD_URLS = {
    2023: 'https://blog.iclr.cc/2023/03/21/announcing-the-iclr-2023-outstanding-paper-award-recipients/',
    2024: 'https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/',
    2025: 'https://blog.iclr.cc/2025/04/22/announcing-the-outstanding-paper-awards-at-iclr-2025/',
    2026: 'https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(data):
    return digest(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())


def normalize(value):
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]', '', value.casefold())


def paper_ids(event):
    links = [event.get('paper_url')] + [m.get('uri') for m in event.get('eventmedia', [])]
    found = set()
    for link in links:
        if not link:
            continue
        url = urlparse(link)
        if url.hostname != 'openreview.net' or url.path != '/forum':
            continue
        identifier = parse_qs(url.query).get('id', [''])[0]
        # Exclude conference-generated aliases such as 2025-Oral--7534-a5745671.
        if re.fullmatch('[A-Za-z0-9_-]{8,15}', identifier):
            found.add(identifier)
    return found


def read_awards(data):
    soup = BeautifulSoup(data, 'html.parser')
    article = soup.select_one('.entry-content') or soup.select_one('article') or soup
    current = None
    awards = {}
    for node in article.find_all(['h2', 'h3', 'a']):
        if node.name != 'a':
            heading = node.get_text(' ', strip=True).casefold()
            if 'honorable mention' in heading:
                current = 'honorable_mention'
            elif 'outstanding paper' in heading or heading == 'award winners':
                current = 'outstanding'
            else:
                current = None
        elif current:
            identifiers = paper_ids({'paper_url': node.get('href')})
            for identifier in identifiers:
                assert identifier not in awards, ('Duplicate award link', identifier)
                awards[identifier] = current
    assert awards, 'No named awards parsed'
    return awards


def verify_year(year):
    raw_paths = [CORPUS / str(year) / name for name in PATHS[year]]
    raw = [path.read_bytes() for path in raw_paths]
    manifest_path = HERE / f'manifest_{year}.json'
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    catalog = json.loads(raw[0])['results']
    by_event = {event['id']: event for event in catalog}
    assert len(by_event) == len(catalog), 'Duplicate catalog event IDs'
    by_paper = {row['openreview_id']: row for row in manifest}
    assert len(by_paper) == len(manifest), 'Duplicate manifest OpenReview IDs'
    assert all(row['year'] == year for row in manifest)

    soup = BeautifulSoup(raw[1], 'html.parser')
    scheduled = set()
    for link in soup.select('a[href]'):
        match = re.fullmatch(f'/virtual/{year}/oral/(\\d+)/?', urlparse(link['href'].strip()).path)
        if match:
            scheduled.add(int(match[1]))
    oral_events = {event['id'] for event in catalog if event['eventtype'] == 'Oral'}
    assert scheduled == oral_events, ('HTML/catalog oral mismatch', scheduled ^ oral_events)
    conference_source = f'https://openreview.net/group?id=ICLR.cc/{year}/Conference'

    # Resolve each eligible event using its own real link or an explicitly linked
    # same-UID, same-title event. This does not import the collector's mapping code.
    eligible = [event for event in catalog if event['eventtype'] == 'Oral'
                or event['decision'].casefold() == 'accept (oral)'
                or (year == 2023 and event['decision'] == 'Accept: notable-top-5%')]
    groups = defaultdict(list)
    mapping = []
    title_variants = {}
    for event in eligible:
        assert event['sourceurl'] == conference_source, ('Non-conference source', event['id'])
        candidates = paper_ids(event)
        related_used = []
        for related_id in event.get('related_events_ids', []):
            if related_id not in by_event:
                continue
            related = by_event[related_id]
            assert related['uid'] == event['uid'], ('Related event UID mismatch', event['id'])
            if normalize(related['name']) != normalize(event['name']):
                # Three manually inspected 2025 event/poster title revisions.
                # Explicit related-event links and shared UID are reinforced by
                # equality of the complete ordered author-name lists.
                assert year == 2025 and frozenset([event['id'], related_id]) in {
                    frozenset([31875, 30202]), frozenset([31893, 30408]),
                    frozenset([31756, 28077]),
                }, ('Unreviewed related title variant', event['id'])
                names = [normalize(a['fullname']) for a in event['authors']]
                assert names == [normalize(a['fullname']) for a in related['authors']]
                key = ','.join(map(str, sorted([event['id'], related_id])))
                title_variants[key] = dict(event_ids=sorted([event['id'], related_id]),
                    titles=[event['name'], related['name']], shared_uid=event['uid'],
                    complete_ordered_author_names_match=True)
            assert related['sourceurl'] == conference_source
            candidates.update(paper_ids(related))
            related_used.append(related_id)
        assert len(candidates) == 1, ('Ambiguous/missing real OpenReview ID', event['id'], candidates)
        identifier = next(iter(candidates))
        groups[identifier].append(event)
        mapping.append(dict(event_id=event['id'], openreview_id=identifier,
                            related_event_ids_checked=related_used))
    awards = read_awards(raw[2])
    expected = set(groups) | set(awards)
    missing = sorted(expected - set(by_paper))
    extra = sorted(set(by_paper) - expected)
    assert not missing and not extra, ('Membership mismatch', year, missing, extra)
    assert set(awards).issubset(groups), 'Award-only paper requires explicit identity audit'

    scheduled_papers = {oid for oid, events in groups.items() if any(e['id'] in scheduled for e in events)}
    additional = []
    designation_errors = []
    for identifier, events in groups.items():
        row = by_paper[identifier]
        designations = set()
        if identifier in scheduled_papers or year >= 2024:
            designations.add('oral')
        if year == 2023:
            for decision, designation in [('Accept: notable-top-5%', 'notable_top_5_percent'),
                                           ('Accept: notable-top-25%', 'notable_top_25_percent')]:
                if any(event['decision'] == decision for event in events):
                    designations.add(designation)
        if identifier in awards:
            designations.add(awards[identifier])
            assert AWARD_URLS[year] in row['award_source_urls'], ('Missing award provenance', identifier)
        if set(row['designations']) != designations:
            designation_errors.append(dict(openreview_id=identifier, expected=sorted(designations),
                                           actual=row['designations']))
        assert row['event_url'] in {'https://iclr.cc' + event['virtualsite_url'] for event in events}
        if identifier not in scheduled_papers:
            additional.append(dict(openreview_id=identifier, title=row['title'],
                                   decisions=sorted({event['decision'] for event in events}),
                                   event_ids=sorted(event['id'] for event in events),
                                   event_types=sorted({event['eventtype'] for event in events})))
    assert not designation_errors, designation_errors
    membership = [dict(openreview_id=identifier, year=year,
                       designations=sorted(by_paper[identifier]['designations']),
                       event_url=by_paper[identifier]['event_url']) for identifier in sorted(by_paper)]
    accepted_orals = {oid for oid, events in groups.items()
                      if any(e['decision'].casefold() == 'accept (oral)' for e in events)}
    return dict(
        year=year, status='passed', unique_manifest_papers=len(manifest),
        scheduled_oral_event_count=len(scheduled), unique_scheduled_oral_papers=len(scheduled_papers),
        additional_scope_papers=len(additional), additional_papers=additional,
        oral_acceptance_outside_scheduled_ids=sorted(accepted_orals - scheduled_papers),
        award_counts=dict(Counter(awards.values())), named_award_ids=awards,
        named_awards_missing=[], extra_manifest_ids=[], missing_manifest_ids=[],
        duplicate_openreview_ids=[], designation_errors=[],
        catalog_records=len(catalog), catalog_source_counts=dict(Counter(e['sourceurl'] for e in catalog)),
        eligible_record_source_counts=dict(Counter(e['sourceurl'] for e in eligible)),
        non_conference_records_in_selected_scope=0,
        all_scheduled_orals_mapped=True, html_and_json_oral_event_sets_identical=True,
        membership_sha256=canonical_hash(membership), membership=membership,
        event_to_paper_mapping=sorted(mapping, key=lambda item: item['event_id']),
        related_event_title_variants=list(title_variants.values()),
        sources=[dict(path=str(path), bytes=len(data), sha256=digest(data),
                      url=(f'https://iclr.cc/static/virtual/data/iclr-{year}-orals-posters.json'
                           if index == 0 else f'https://iclr.cc/virtual/{year}/events/oral'
                           if index == 1 else AWARD_URLS[year]))
                 for index, (path, data) in enumerate(zip(raw_paths, raw))],
        manifest_snapshot=dict(path=str(manifest_path), sha256=digest(manifest_raw), bytes=len(manifest_raw)),
    )


def main():
    years = [verify_year(year) for year in PATHS]
    all_ids = [row['openreview_id'] for year in years for row in year['membership']]
    assert len(set(all_ids)) == len(all_ids), 'Duplicate OpenReview ID across years'
    result = dict(
        schema_version=1, status='passed', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        script_path=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__).read_bytes()),
        audit_scope='Offline public membership and designation audit only; PDF retrieval/version integrity is excluded.',
        corpus_scope='All scheduled main-conference oral papers, all additional 2023 notable-top-5% papers, all official oral acceptances in 2024–2026, and all publicly named outstanding/honorable-mention papers in the saved official award posts.',
        limitations=[
            'This is completeness relative to the saved official sources, not proof that no unlisted event or private award shortlist exists.',
            '2023 virtual top-25% papers lacking scheduled oral events are outside the defined scope unless independently included by an award; top-25% designation alone is not represented as an oral acceptance.',
            'Unpublished candidate/finalist lists are not inferred from reported committee shortlist sizes.',
            'A source-linked main-conference event establishes membership; no workshop, TinyPaper, TMLR, JMLR, or blog-track entry is selected.',
            'Manifest full-file snapshots may change as download metadata are updated; membership_sha256 binds only year, real OpenReview ID, verified designations, and event URL.',
        ],
        totals=dict(unique_papers=sum(y['unique_manifest_papers'] for y in years),
                    unique_openreview_ids_across_years=len(set(all_ids)),
                    scheduled_oral_papers=sum(y['unique_scheduled_oral_papers'] for y in years),
                    additional_scope_papers=sum(y['additional_scope_papers'] for y in years),
                    publicly_named_award_papers=sum(sum(y['award_counts'].values()) for y in years)),
        years=years,
    )
    output = HERE / 'membership_audit.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=result['status'], totals=result['totals'],
                          per_year={y['year']:y['unique_manifest_papers'] for y in years},
                          output=str(output), sha256=digest(output.read_bytes())), indent=2))


if __name__ == '__main__':
    main()
