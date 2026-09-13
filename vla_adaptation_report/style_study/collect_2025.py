"""Archive every ICLR 2025 main-conference oral and publicly named award paper.

Two concurrent requests at most. Oral aliases are joined to their linked poster
records for real OpenReview IDs. PDFs come from official published proceedings;
OpenReview's download endpoint returned a 403 challenge during this collection.
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup
import fitz
import requests

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/fengze/online_adaptation_research/iclr_2023_2026/2025")
ORAL = "https://iclr.cc/virtual/2025/events/oral"
STATIC = "https://iclr.cc/static/virtual/data/iclr-2025-orals-posters.json"
AWARDS = "https://blog.iclr.cc/2025/04/22/announcing-the-outstanding-paper-awards-at-iclr-2025/"
PROCEEDINGS = "https://proceedings.iclr.cc/paper_files/paper/2025"
MANIFEST = HERE / "manifest_2025.json"
ALIASES = {
    "AlphaEdit: Null-Space Constrained Model Editing for Language Models":
        "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models",
    "SymmetricDiffusers: Learning Discrete Diffusion Models over Finite Symmetric Groups":
        "SymmetricDiffusers: Learning Discrete Diffusion on Finite Symmetric Groups",
    "Classic but Everlasting: Traditional Gradient-Based Algorithms Converges Fast Even in Time-Varying Multi-Player Games":
        "Classic but Everlasting: Traditional Gradient-Based Algorithms Converge Fast Even in Time-Varying Multi-Player Games",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())


def fetch(url: str) -> bytes:
    errors = []
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(15, 90), headers={"User-Agent": "Mozilla/5.0 (ICLR public proceedings research archive)"})
            if response.status_code in (401, 403, 404):
                raise PermissionError(f"HTTP {response.status_code}: {url}")
            response.raise_for_status()
            time.sleep(.15)
            return response.content
        except PermissionError:
            raise
        except (requests.RequestException, OSError) as exc:
            errors.append(str(exc))
            if attempt < 4:
                time.sleep(min(2**attempt, 16))
    raise RuntimeError("; ".join(errors))


def cached_catalog(name: str, url: str) -> bytes:
    path = ROOT / "catalogs" / name
    if not path.exists():
        path.write_bytes(fetch(url))
    return path.read_bytes()


def make_records() -> tuple[list[dict], dict]:
    for directory in ["catalogs", "pdfs", "texts", "events"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    raw_sources = {}
    for name, url in [("oral.html", ORAL), ("awards.html", AWARDS),
                      ("iclr-2025-orals-posters.json", STATIC), ("proceedings.html", PROCEEDINGS)]:
        data = cached_catalog(name, url)
        raw_sources[name] = {"url": url, "path": str(ROOT / "catalogs" / name), "sha256": sha(data), "bytes": len(data)}
    catalog = json.loads((ROOT / "catalogs/iclr-2025-orals-posters.json").read_text())
    rows = catalog["results"]
    by_id = {r["id"]: r for r in rows}
    oral_html = BeautifulSoup((ROOT / "catalogs/oral.html").read_text(), "html.parser")
    oral_ids = {int(c["data-event-id"]) for c in oral_html.select('.event-card[data-event-type="Oral"]')}
    oral_rows = [r for r in rows if r["eventtype"] == "Oral"]
    assert oral_ids == {r["id"] for r in oral_rows}
    award_html = BeautifulSoup((ROOT / "catalogs/awards.html").read_text(), "html.parser")
    award_records = {}
    article = award_html.find("article")
    for link in article.find_all("a", href=True):
        if "openreview.net/forum?id=" not in link["href"]:
            continue
        heading = link.find_previous(["h2", "h3"]).get_text(" ", strip=True).lower()
        if "outstanding papers" in heading:
            designation = "outstanding"
        elif "honorable mentions" in heading:
            designation = "honorable_mention"
        else:
            raise ValueError(f"Unknown award designation: {heading}")
        oid = parse_qs(urlparse(link["href"]).query)["id"][0]
        award_records[oid] = {"designation": designation, "award_title": link.get_text(" ", strip=True).rstrip(".")}
    assert len(award_records) == 6
    pub_html = BeautifulSoup((ROOT / "catalogs/proceedings.html").read_text(), "html.parser")
    publications = {}
    for link in pub_html.select('a[href*="-Abstract-Conference.html"]'):
        key = normalize(link.get_text(" ", strip=True))
        if key in publications:
            raise ValueError(f"Duplicate normalized proceedings title: {key}")
        publications[key] = link
    records = {}
    for event in oral_rows:
        linked = [by_id[i] for i in event["related_events_ids"] if i in by_id
                  and by_id[i]["uid"] == event["uid"] and by_id[i]["eventtype"] == "Poster"]
        assert len(linked) == 1
        poster = linked[0]
        assert normalize(ALIASES.get(event["name"], event["name"])) == normalize(poster["name"])
        oid = parse_qs(urlparse(poster["paper_url"]).query)["id"][0]
        assert not oid.startswith("2025-Oral-") and oid not in records
        published_title = ALIASES.get(event["name"], event["name"])
        publication = publications[normalize(published_title)]
        paper_authors = publication.parent.select_one(".paper-authors").get_text(" ", strip=True)
        title = publication.get_text(" ", strip=True)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:90]
        record = {
            "year": 2025, "title": title, "authors": paper_authors.split(", "),
            "event_title": event["name"], "event_url": urljoin("https://iclr.cc", event["virtualsite_url"]),
            "poster_event_url": urljoin("https://iclr.cc", poster["virtualsite_url"]),
            "openreview_id": oid, "openreview_url": poster["paper_url"],
            "pdf_url": None, "proceedings_url": urljoin("https://proceedings.iclr.cc", publication["href"]),
            "designations": ["oral"], "award_source_urls": [],
            "pdf_path": str(ROOT / "pdfs" / f"{oid}_{slug}.pdf"),
            "text_path": str(ROOT / "texts" / f"{oid}_{slug}.txt"),
            "sha256": None, "bytes": None, "pages": None, "download_status": "pending",
            "version": "Official published proceedings PDF retrieved publicly; not assumed to be the original submission or review-time version.",
            "authors_source": PROCEEDINGS,
            "oral_source_urls": [ORAL, STATIC],
            "openreview_mapping": {"oral_event_id": event["id"], "poster_event_id": poster["id"], "shared_uid": event["uid"]},
            "title_mapping_note": "Published title differs from event/award title; explicit verified alias retained." if event["name"] in ALIASES else None,
        }
        if oid in award_records:
            record["designations"].append(award_records[oid]["designation"])
            record["award_source_urls"] = [AWARDS]
            record["award_title"] = award_records[oid]["award_title"]
        records[oid] = record
    assert set(award_records) <= set(records)
    metadata = {"year": 2025, "retrieved_utc": now(), "sources": raw_sources,
                "oral_events": len(oral_rows), "unique_openreview_ids": len(records),
                "named_outstanding": 3, "named_honorable_mentions": 3,
                "unnamed_initial_award_pool": 36,
                "award_scope": "The official announcement names only 3 winners and 3 honorable mentions; no additional candidates/finalists inferred.",
                "openreview_pdf_access": "One direct test returned HTTP 403 challenge; no repeated 403 requests. Official proceedings used instead.",
                "title_aliases": ALIASES, "script_sha256": sha(Path(__file__).read_bytes())}
    return list(records.values()), metadata


def download_record(record: dict) -> dict:
    result = dict(record)
    try:
        oid = result["openreview_id"]
        event_path = ROOT / "events" / f"{oid}_proceedings.html"
        if not event_path.exists():
            event_path.write_bytes(fetch(result["proceedings_url"]))
        soup = BeautifulSoup(event_path.read_text(), "html.parser")
        links = [a for a in soup.find_all("a", href=True) if a.get_text(" ", strip=True) == "Paper"]
        assert len(links) == 1
        result["pdf_url"] = urljoin("https://proceedings.iclr.cc", links[0]["href"])
        result["proceedings_page_sha256"] = sha(event_path.read_bytes())
        result["proceedings_page_path"] = str(event_path)
        pdf_path = Path(result["pdf_path"])
        pdf = pdf_path.read_bytes() if pdf_path.exists() else fetch(result["pdf_url"])
        assert pdf.startswith(b"%PDF-")
        with fitz.open(stream=pdf, filetype="pdf") as document:
            assert not document.is_encrypted and len(document) > 0
            texts = [page.get_text(sort=True) for page in document]
            result["pages"] = len(document)
            result["page_word_counts"] = [len(re.findall(r"\b[\w'-]+\b", t)) for t in texts]
        if not pdf_path.exists():
            partial = pdf_path.with_suffix(".pdf.partial")
            partial.write_bytes(pdf)
            partial.replace(pdf_path)
        text = "\n\f\n".join(f"[PDF PAGE {i+1}]\n{t}" for i, t in enumerate(texts))
        Path(result["text_path"]).write_text(text)
        result.update(sha256=sha(pdf), bytes=len(pdf), text_sha256=sha(text.encode()),
                      download_status="downloaded_verified", retrieved_utc=now())
        # Metadata-only structure proxies; not a claim that every paper was read.
        ref_page = next((i+1 for i, t in enumerate(texts) if any(re.sub(r"\s+", "", line).upper() == "REFERENCES" for line in t.splitlines())), None)
        figure_matches = [(i+1, m.group(2)) for i, t in enumerate(texts)
                          for m in re.finditer(r"(?im)^\s*(Figure|Fig\.)\s+(\d+(?:\.\d+)?)\s*[:.]", t)]
        table_matches = [(i+1, m.group(1)) for i, t in enumerate(texts)
                         for m in re.finditer(r"(?im)^\s*Table\s+(\d+(?:\.\d+)?)\s*[:.]", t)]
        result["text_structure"] = {"references_first_heading_page": ref_page,
                                    "figure_caption_matches": figure_matches, "table_caption_matches": table_matches,
                                    "first_page_figure_caption": any(p == 1 for p, _ in figure_matches),
                                    "first_three_pages_figure_caption": any(p <= 3 for p, _ in figure_matches)}
    except Exception as exc:
        result.update(download_status="error", error=f"{type(exc).__name__}: {exc}", attempted_utc=now())
    return result


def write_manifest(records: list[dict]) -> None:
    partial = MANIFEST.with_suffix(".json.partial")
    partial.write_text(json.dumps(sorted(records, key=lambda r: (r["title"].casefold(), r["openreview_id"])), indent=2) + "\n")
    partial.replace(MANIFEST)


def main() -> None:
    records, metadata = make_records()
    write_manifest(records)
    (HERE / "catalog_2025.json").write_text(json.dumps(metadata, indent=2) + "\n")
    priority = ["6Mxhg9PtDE", "tPNHOoZFl9", "HvSytvg3Jh", "HD6bWcj87Y", "Ha6RTeWMd0", "vo9t20wsmd"]
    records.sort(key=lambda r: (priority.index(r["openreview_id"]) if r["openreview_id"] in priority else
                                6 if any(x in r["title"] for x in ["Geometry of Neural Reinforcement", "Flat Reward in Policy"]) else 7,
                                r["title"]))
    by_id = {r["openreview_id"]: r for r in records}
    print(json.dumps({"stage": "catalog_ready", "orals": len(records), "awards": len(priority)}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download_record, r): r["openreview_id"] for r in records}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            by_id[result["openreview_id"]] = result
            write_manifest(list(by_id.values()))
            if done <= 8 or done % 10 == 0 or result["download_status"] == "error":
                print(json.dumps({"done": done, "total": len(records), "status": result["download_status"],
                                  "id": result["openreview_id"], "title": result["title"],
                                  "pages": result["pages"], "error": result.get("error")}), flush=True)
    completed = list(by_id.values())
    ok = [r for r in completed if r["download_status"] == "downloaded_verified"]
    stats = {"year": 2025, "complete": len(ok) == len(records), "expected_unique_papers": len(records),
             "downloaded_verified": len(ok), "errors": [r for r in completed if r["download_status"] != "downloaded_verified"],
             "total_bytes": sum(r["bytes"] for r in ok), "total_pages": sum(r["pages"] for r in ok),
             "manifest_sha256": sha(MANIFEST.read_bytes()), "catalog_sha256": sha((HERE / "catalog_2025.json").read_bytes()),
             "script_sha256": sha(Path(__file__).read_bytes()), "finished_utc": now()}
    (HERE / "collection_2025_receipt.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k:v for k,v in stats.items() if k != "errors"}), flush=True)


if __name__ == "__main__":
    main()
