# Before committing and releasing

- [ ] Review all source files and provenance against frozen IJPHM_020 hashes.
- [ ] Review Stage13 `%.17g` patch; ensure only the serialization changed.
- [ ] Confirm authorial rights and approve applying existing root BSD-3-Clause license to this added code.
- [ ] Confirm no NASA raw CSVs, predictions, large run outputs, credentials or environment paths are staged.
- [ ] Run 25 tests in a new environment and compare one real-data replay with the published audit.
- [ ] Keep the historical v1.0.0 tag/release and historical DOI unchanged.
- [ ] Update root README to link this new workflow; avoid suggesting old RUL workflows reproduce IJPHM tables.
- [ ] Review root CITATION.cff for a new *version-specific* release DOI after deposition; do not guess a DOI.
- [ ] Create a new tag/release only after reviewing proposed Git diff and confirming its version number.
- [ ] Cross-check all statements against accepted manuscript; do not claim predictive deployment validation.
