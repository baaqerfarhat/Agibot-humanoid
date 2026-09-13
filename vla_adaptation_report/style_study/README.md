# ICLR 2023–2026 writing corpus

Start with [ICLR_WRITING_STUDY.md](ICLR_WRITING_STUDY.md): the complete guide,
34 targeted paper profiles, award scope, structural measurements, manuscript
application, and every indexed PDF link. [CORPUS_INDEX.md](CORPUS_INDEX.md) is a
shorter navigation entry point containing only the paper index.

Bulk PDFs/text/raw records are stored outside git at
[/home/fengze/online_adaptation_research/iclr_2023_2026/](/home/fengze/online_adaptation_research/iclr_2023_2026/).
There are 793 unique selected papers: 270/86/213/224 by year. This includes all
scheduled main-conference oral papers, all 34 named award/honorable recipients,
22 additional 2023 top-5% papers, and one 2026 oral acceptance with only a poster
event. Complete unnamed award longlists/shortlists are not publicly recoverable.

## Files and provenance

- `manifest_YEAR.json`: authoritative membership and PDF source/hash records.
- `collect_YEAR.py`: year-specific catalog reconciliation and collection.
- `verify_membership.py`, `membership_audit.json`: independent coverage checks.
- `analyze_corpus.py`, `structure_metrics.json`, `corpus_receipt.json`: PDF
  integrity and conservative structural measurements; cached by PDF hash.
- `YEAR_*.md`: selected writing profiles with primary links and exact PDF pages.
- `writing_guide_body.md`: section-by-section synthesis and manuscript changes.
- `build_report.py`: assembles the complete guide and PDF index.
- `report_receipt.json`: guide input/output hashes and coverage counts.

For 2023, the initial arXiv collector recovered 227 papers. Manual primary-source
searches and a pinned public PDF archive recovered the remaining 43. The selected
2023 copies comprise 227 arXiv versions, eight author/institution conference
copies, and 35 explicitly labeled public-archive copies. Other recovered versions
remain available as alternatives. The public archive supplies file bytes; official
ICLR records supply membership and awards. No archived or alternate PDF is
asserted byte-identical to an accepted OpenReview revision.

2023 source recovery is recorded in `manual_sources_2023_{a,b,mirror}.json`,
`mirror_identity_checks_2023.json`, and `collection_2023_receipt.json`.
The archive is pinned to commit `ce5b4a0f363cbefda8004455b2a46dc04da78e7b`;
downloaded LFS hashes and title/author identity were checked. It is accessed as a
separate public source, without bypassing OpenReview's browser challenge.

## Reproduction

From the repository root, with Python, requests, BeautifulSoup, NumPy and PyMuPDF:

```sh
python paper/style_study/collect_2023.py
python paper/style_study/collect_2024.py
python paper/style_study/collect_2025.py
python paper/style_study/collect_2026.py
python paper/style_study/merge_2023_sources.py
python paper/style_study/verify_membership.py
python paper/style_study/analyze_corpus.py
python paper/style_study/build_report.py
```

Collectors reuse local cached sources. The checked-in manifests and raw source
hashes describe this dated snapshot, not a live promise that a remote endpoint
will remain unchanged. The 2023 merge reuses the manually verified alternate
downloads; exact manual source records remain in the repository for recovery.
The HTML/PDF caches must be present at the recorded paths for a completely
offline rebuild. `collect_public_mirror_2023.py` records automatic archive
recovery; two typography/title variants received additional human-readable
identity checks.

No claim is made that all 793 papers were closely read or that a layout choice
caused an award. All PDFs are structurally examined; 34 papers receive targeted
qualitative readings. Appendix proofs are not independently audited by this
writing study. The manuscript's nine-page check is a separate operation:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr_draft.tex
python check_submission_layout.py --require-main-pages 9
```
