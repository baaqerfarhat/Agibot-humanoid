#!/usr/bin/env python3
"""Collect all official ICLR 2026 oral acceptances and publicly named awards.

At most two concurrent HTTP requests. Official proceedings PDFs are used
because current OpenReview access is challenged; no access-control workaround.
Raw catalogs, PDFs and page-delimited text remain outside the git repository.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import unicodedata
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup
import fitz
import requests

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO/"paper/style_study"
DEFAULT = Path("/home/fengze/online_adaptation_research/iclr_2023_2026/2026")
SOURCES = {
    "orals.html": "https://iclr.cc/virtual/2026/events/oral",
    "orals_posters.json": "https://iclr.cc/static/virtual/data/iclr-2026-orals-posters.json",
    "awards.html": "https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/",
    "proceedings.html": "https://proceedings.iclr.cc/paper_files/paper/2026",
}
AWARDS = {
    "Yxz92UuPLQ": ("outstanding", "Transformers are Inherently Succinct"),
    "VKGTGGcwl6": ("outstanding", "LLMs Get Lost In Multi-Turn Conversation"),
    "yRtgZ1K8hO": ("honorable_mention", "The Polar Express: Optimal Matrix Sign Methods and their Application to the Muon Algorithm"),
}
DEEP = list(AWARDS) + ["bFYfV6c9zu", "IZHk6BXBST", "mIeKe74W43",
                      "yDmb7xAfeb", "AufVSUgMUo", "EJ680UQeZG"]
# Read from the downloaded first pages. Proceedings metadata retain the catalog
# title while these public PDF headers contain typography or title revisions.
PRINTED_TITLES = {
    "ItFuNJQGH4": "p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding",
    "g88nt4ieTG": "Diffusion Language Models Know the Answer Before Decoding",
    "vH7OAPZ2dR": "GLASS Flows: Transition Sampling for Alignment of Flow and Diffusion Models",
    "gpsczXOsHn": "Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Minimization",
    "3eTr9dGwJv": "MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Model for Embodied Task Planning",
    "7xjoTuaNmN": "Data Recipes for Reasoning Models",
    "EJ680UQeZG": "Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers",
    "EA80Zib9UI": "Safety-Guided Flow: A Unified Framework for Negative Guidance in Safe Generation",
    "P0GOk5wslg": "Speculative Actions: A Lossless Framework for Faster Agentic Systems",
}
LOCAL = threading.local()
LOG_LOCK = threading.Lock()


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(value):
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub("[^a-z0-9]", "", value.lower())


def fetch(url, destination, log):
    if destination.exists() and destination.stat().st_size:
        return dict(status="cached", retrieved_utc=datetime.fromtimestamp(
            destination.stat().st_mtime, timezone.utc).isoformat())
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers["User-Agent"] = "ICLR-public-proceedings-research/1.0"
    for attempt in range(4):
        when = utc()
        try:
            response = LOCAL.session.get(url, timeout=(20, 90))
            record = dict(utc=when, url=url, status=response.status_code,
                          bytes=len(response.content), attempt=attempt+1)
            with LOG_LOCK, log.open("a") as stream:
                stream.write(json.dumps(record)+"\n")
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()  # no retries/workarounds for challenges
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix+".part")
            temporary.write_bytes(response.content)
            temporary.replace(destination)
            return dict(status="downloaded", retrieved_utc=when,
                        content_type=response.headers.get("content-type"),
                        final_url=response.url)
        except requests.RequestException as error:
            if getattr(error, "response", None) is not None:
                raise
            with LOG_LOCK, log.open("a") as stream:
                stream.write(json.dumps(dict(utc=when, url=url, attempt=attempt+1,
                                             error=str(error)))+"\n")
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Retry budget exhausted")


def real_ids(record):
    urls = [record.get("paper_url")]+[m.get("uri") for m in record.get("eventmedia", [])]
    found = set()
    for url in urls:
        if not url or "openreview.net/forum?" not in url:
            continue
        identifier = parse_qs(urlparse(url).query).get("id", [""])[0]
        # Conference event IDs such as 2026-Oral--22526-187fa8f1 are synthetic.
        if re.fullmatch(r"[A-Za-z0-9_-]{8,15}", identifier):
            found.add(identifier)
    return found


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    corpus = args.corpus_dir.resolve()
    raw = corpus/"raw"
    for path in (raw, corpus/"pdf", corpus/"text", OUTPUT):
        path.mkdir(parents=True, exist_ok=True)
    log = raw/"requests_2026.jsonl"
    source_receipts = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fetch, url, raw/name, log): (name, url)
                   for name, url in SOURCES.items()}
        for future in as_completed(futures):
            name, url = futures[future]
            source_receipts[name] = dict(url=url, path=str(raw/name),
                sha256=digest(raw/name), **future.result())
    events = json.loads((raw/"orals_posters.json").read_text())["results"]
    catalog = BeautifulSoup((raw/"orals.html").read_bytes(), "html.parser")
    oral_urls = {urljoin("https://iclr.cc", a["href"].strip())
                 for a in catalog.find_all("a", href=True)
                 if re.fullmatch(r"/virtual/2026/oral/\d+", a["href"].strip())}
    awards_html = BeautifulSoup((raw/"awards.html").read_bytes(), "html.parser")
    award_links = {parse_qs(urlparse(a["href"]).query).get("id", [""])[0]: a.get_text(" ", strip=True)
                   for a in awards_html.find_all("a", href=True) if "openreview.net/forum?" in a["href"]}
    for identifier, (_, title) in AWARDS.items():
        if norm(award_links.get(identifier, "")) != norm(title):
            raise ValueError("Award source changed: "+identifier)
    groups = {}
    for event in events:
        if event["eventtype"] == "Oral" or event["decision"] == "Accept (Oral)":
            identifiers = real_ids(event)
            if len(identifiers) != 1:
                raise ValueError("Missing/ambiguous real ID: "+event["name"])
            groups.setdefault(next(iter(identifiers)), []).append(event)
    if not set(AWARDS).issubset(groups):
        raise ValueError("Public award outside oral catalog: add it explicitly")
    proceedings = BeautifulSoup((raw/"proceedings.html").read_bytes(), "html.parser")
    by_title = {}
    for link in proceedings.find_all("a", title="paper title"):
        by_title.setdefault(norm(link.get_text(" ", strip=True)), []).append(link["href"])

    rows = []
    for identifier, group in groups.items():
        oral = [event for event in group if event["eventtype"] == "Oral"]
        primary = oral[0] if oral else group[0]
        alternatives = by_title.get(norm(primary["name"]), [])
        if len(alternatives) != 1:
            raise ValueError("Proceedings title match not unique: "+primary["name"])
        abstract = urljoin("https://proceedings.iclr.cc", alternatives[0])
        designations = ["oral"]+([AWARDS[identifier][0]] if identifier in AWARDS else [])
        rows.append(dict(
            year=2026, title=primary["name"],
            authors=[author["fullname"] for author in primary["authors"]],
            event_url=urljoin("https://iclr.cc", primary["virtualsite_url"]),
            openreview_id=identifier, openreview_url="https://openreview.net/forum?id="+identifier,
            pdf_url=None, designations=designations,
            award_source_urls=[SOURCES["awards.html"]] if identifier in AWARDS else [],
            pdf_path=str(corpus/"pdf"/(identifier+".pdf")),
            text_path=str(corpus/"text"/(identifier+".txt")),
            sha256=None, bytes=None, pages=None, download_status="pending",
            proceedings_url=abstract,
            designation_evidence=dict(official_decision="Accept (Oral)",
                source_url=SOURCES["orals_posters.json"], has_scheduled_oral=bool(oral),
                event_ids=[event["id"] for event in group]),
            source_version="Official ICLR 2026 proceedings PDF, retrieved public version; not asserted to equal latest OpenReview revision.",
            errors=[]))
    rows.sort(key=lambda row: row["title"].casefold())
    manifest = OUTPUT/"manifest_2026.json"
    write_json(manifest, rows)

    def download(row):
        identifier = row["openreview_id"]
        try:
            abstract_path = raw/"abstracts"/(identifier+".html")
            fetch(row["proceedings_url"], abstract_path, log)
            html = BeautifulSoup(abstract_path.read_bytes(), "html.parser")
            metadata = {}
            for tag in html.find_all("meta"):
                metadata.setdefault(tag.get("name"), []).append(tag.get("content"))
            if norm(metadata["citation_title"][0]) != norm(row["title"]):
                raise ValueError("Proceedings citation title does not match catalog")
            pdf_url = metadata["citation_pdf_url"][0]
            if not pdf_url.startswith("https://proceedings.iclr.cc/"):
                raise ValueError("Unexpected proceedings PDF host")
            row.update(pdf_url=pdf_url, proceedings_authors=metadata.get("citation_author", []),
                       proceedings_publication_date=metadata.get("citation_publication_date", [None])[0])
            result = fetch(pdf_url, Path(row["pdf_path"]), log)
            with fitz.open(row["pdf_path"]) as document:
                if not document.is_pdf or document.page_count == 0:
                    raise ValueError("Response is not a nonempty PDF")
                page_texts = [page.get_text("text", sort=True) for page in document]
                row["pages"] = document.page_count
                row["title_on_first_page_normalized"] = norm(row["title"]) in norm(page_texts[0])
                printed = PRINTED_TITLES.get(identifier, row["title"])
                row["printed_title"] = printed
                if norm(printed) not in norm(page_texts[0]):
                    raise ValueError("Unresolved printed title; inspect first page")
                row["title_verification"] = ("matches_catalog" if row["title_on_first_page_normalized"]
                                             else "verified_printed_title_variant")
            text = "\n".join(f"\n===== PDF PAGE {number} =====\n{value}"
                             for number, value in enumerate(page_texts, 1))
            Path(row["text_path"]).write_text(text)
            row.update(download_status="ok", sha256=digest(row["pdf_path"]),
                       bytes=Path(row["pdf_path"]).stat().st_size,
                       text_sha256=digest(row["text_path"]), text_characters=len(text),
                       retrieved_utc=result["retrieved_utc"], retrieval=result["status"])
        except Exception as error:
            row["download_status"] = "error"
            row["errors"].append(repr(error))
        return row

    priority = {identifier: index for index, identifier in enumerate(DEEP)}
    ordered = sorted(rows, key=lambda row: (priority.get(row["openreview_id"], 100), row["title"]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, row) for row in ordered]
        for number, future in enumerate(as_completed(futures), 1):
            row = future.result()
            write_json(manifest, rows)
            if number <= 9 or number % 10 == 0 or row["download_status"] != "ok":
                print(f"{number}/{len(rows)} {row['download_status']} {row['openreview_id']} {row['title'][:80]}", flush=True)

    summary = dict(
        year=2026, completed_utc=utc(), source_receipts=source_receipts,
        scheduled_oral_urls=len(oral_urls), unique_oral_acceptances=len(rows),
        supplemental_poster_only_oral=[row["title"] for row in rows
                                      if not row["designation_evidence"]["has_scheduled_oral"]],
        outstanding=2, honorable_mention=1, unnamed_shortlist_papers=2,
        shortlist_scope="Official committee had a five-paper shortlist and a 36-paper longlist; only three recognized titles are publicly named in its announcement. No unnamed finalist/candidate designations inferred.",
        pdf_version_scope="Public official proceedings files. OpenReview latest-revision comparison unavailable because of access challenge reported in this collection session.",
        successful=sum(row["download_status"] == "ok" for row in rows),
        verified_printed_title_variants=[row["openreview_id"] for row in rows
                                        if row.get("title_verification") == "verified_printed_title_variant"],
        errors=[row["openreview_id"] for row in rows if row["download_status"] != "ok"],
        total_pdf_bytes=sum(row["bytes"] or 0 for row in rows),
        total_pages=sum(row["pages"] or 0 for row in rows),
        requested_deep_read_ids=DEEP,
        automated_scope="Metadata collection, title mapping, PDF validation and full-text extraction for every row; not a claim of reading every paper in depth.",
        script_sha256=digest(__file__))
    write_json(OUTPUT/"metadata_2026.json", summary)
    print(json.dumps({key:summary[key] for key in ("successful","errors","total_pdf_bytes","total_pages")}), flush=True)


if __name__ == "__main__":
    main()
