# Draft release notes — NOT PUBLISHED

Proposed semantic release: v2.0.0 (subject to author review).

Adds a versioned `ijphm_2026/` folder with the 2026 retrospective near-endpoint
C-MAPSS experiments (FD001--FD004), independent mathematical/audit code,
NASA-original source conversion, pinned observed dependencies and an auditable
17-significant-digit export improvement in Stage13. Original source lineage is
recorded in `metadata/source_lineage.csv`.

Prior v1.0.0 RUL workflows, datasets, figures, and DOI remain unchanged.
No NASA raw datasets or per-cycle prediction outputs are bundled.

Known limitation: exact model probabilities can vary numerically between
independent runs even where primary AP conclusions are stable. No prospective
validation, no true nonfailure controls, and no universal RPD dominance over
its components are claimed. Full publication readiness requires author review.
