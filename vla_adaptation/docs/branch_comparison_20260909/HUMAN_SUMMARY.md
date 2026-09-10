Compared `main` (`7ca2f26`) with `review/dual-track-audit` (`cda6c92`).

- Fixed statistical errors; added fault cleanup and stronger reset/calibration checks.
- Diagnosed gripper normalization bias; a complete fix remains unverified.
- Expanded Panda/ALOHA joint-fault tests; weighted correction helps some conditions but harms others.
- Added and tuned matched estimators and composite adaptation; superiority over Kalman remains unproven.
- Added ARX system identification using measured-motion history and one shared prediction/correction input map.
- Added a constant-basis descriptor prototype: Spatial success improved **112/140 → 135/140**.
- Added theory, analysis figures, citation records and paper-layout checks.
- Preserve main’s newer GR1 and WidowX work when merging.

[Detailed Claude handoff](CLAUDE_HANDOFF.md)
