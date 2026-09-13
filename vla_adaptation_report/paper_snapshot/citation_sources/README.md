# Citation source records

Raw exports and source pages retain their original whitespace. Local Git attributes
exclude these archived `.bib` and `.html` payloads from whitespace linting; the
manuscript bibliography and authored files retain normal checks.

These records support the 2026-09-08 bibliography audit. They preserve source metadata; they are not new papers or independent replications.

- `manifest.json` records the original source URL, source type, byte count, retrieval time and SHA-256 for **48 files covering 30 citation keys**, including **29 raw BibTeX exports**.
- `*.proceedings.bib`, `*.pmlr.bib`, `*.rss.bib` and `*.neurips.bib` contain official conference citation exports. Associated HTML files preserve the pages from which embedded exports were extracted.
- `*.crossref.bib` and `*.crossref.json` contain publisher-deposited DOI metadata. These are Crossref responses, not Google Scholar exports or handwritten replacements.
- `*.datacite.bib` contains arXiv-issued DOI metadata. These sources do not establish peer-reviewed publication.
- `ioannou1996.author_cv.pdf` is the author's USC publication record confirming the original Prentice-Hall 1996 book and later Dover reprint. No publisher BibTeX for that original edition was obtained.
- `integration.json` maps local citation keys to raw exports and records mechanical normalization or checked exceptions. Original exports remain unchanged.
- `local_materials.json` identifies the supplied MAGIC, HMAC and mirror-descent PDFs and TeX sources by path and hash; those files were not changed or copied here.
- `validation.json` records a warning-free check of all 50 entries with the repository's ICLR 2027 BibTeX style. `source_verification.json` verifies every manifest hash and each integrated entry's hash.

The main version corrections are MAGIC's final **2025** T-RO citation, HMAC's final pages **18309–18315**, published CoRL/RSS citations for the VLA policies and DATT/RMA, and Shim's published **2020** encyclopedia contribution. Stable local citation keys sometimes retain an older year. See [the complete audit](../REFERENCE_AUDIT.md) for metadata, source limits and appropriate scientific use.

Raw exports may contain publisher metadata quirks. The integration record identifies normalization of HTML superscripts, Unicode punctuation, proper-name capitalization and math; omission of abstracts and other optional fields; removal of GR00T's standalone colon delimiter; and version-specific authorship. No Google Scholar retrieval is claimed.
