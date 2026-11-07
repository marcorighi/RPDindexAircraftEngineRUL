# IJPHM 2026: RPDgep and aircraft-engine degradation (release candidate)

**Status:** candidate for repository review; not a published release or DOI.

This directory is intended to be added *alongside* the existing v1.0.0
workflow of `marcorighi/RPDindexAircraftEngineRUL`, **without replacing it**.
The older release (19 August 2026) contains a distinct RUL/test-set protocol;
its existing DOI `10.5281/zenodo.22012996` refers to that archival release.
Do not cite that version as containing these newly added Stage 10/13/18 scripts.

## Scientific scope

Retrospective grouped cross-validation on the official NASA C-MAPSS *training*
trajectories FD001--FD004 (709 engines). A binary indicator marks a cycle within
30 cycles of the observed trajectory endpoint. The software tests whether
adding the global RPDgep prefix feature to a sensor+age+OLS baseline improves
ranking of these retrospective near-endpoint cycles. It **does not** estimate
remaining useful life on unseen operational engines or claim prospective failure
prediction. Each dataset's engines end in failure; no true nonfailure controls.

The RPD indices predate this release. Original research code:
https://github.com/marcorighi/RPDindex ; optimized implementation:
https://github.com/marcorighi/RPDindexOptimized .

## Included pipeline

- `03_code/10_failure_proximity/` -- frozen prefix RPD feature computation.
- `03_code/13_nested_alarm/` -- grouped nested thresholds; *only* the
  predictions CSV formatting has been changed from 12 to 17 significant digits.
- `03_code/18_rpd_component_ablation/` -- nine-model RPD/component ablation.
- `03_code/19_stage18_independent_audit/` and `20_manuscript_stage18_v04/`
  -- independent verification and mathematical tests.
- `scripts/` -- NASA source preparation, isolated environment, tests, replay,
  output integrity/packaging and preliminary reference checks.
- `metadata/source_lineage.csv` -- original SHA-256 and release SHA-256 for
  each of the 17 scientific source files.
- `reference/` -- compact original-vs-replica summaries, not predictions.

## Requirements

The audited e2m replication used Python 3.14.7 and the pinned package snapshot
`requirements-observed.txt` (NumPy, pandas, SciPy, scikit-learn, etc.). This is
an observed **version snapshot**, not a universal hash-locked environment.
Machine-level floating-point variation can alter individual probabilities.

### 1. Validate distributed source code

```bash
python3 scripts/verify_source_lineage.py
sha256sum -c SHA256SUMS
```

### 2. Build an isolated Python environment and run tests

```bash
bash scripts/setup_environment.sh
bash scripts/run_tests.sh
```

The expected suite contains 25 synthetic/mathematical tests in four groups.

### 3. Obtain training data from NASA

Download the original *6. Turbofan Engine Degradation Simulation Data Set.zip*
from NASA Prognostics Center of Excellence, for example:

https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

The expected outer ZIP SHA-256 is
`c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.

```bash
./.venv/bin/python scripts/prepare_nasa_data.py /path/to/NASA_turbofan_original.zip
```

The script extracts `CMAPSSData.zip` and writes training CSV files in local
`02_raw_data/` after checking source SHA-256, row counts and motor/cycle keys.
The generated CSVs are deliberately excluded from Git. The raw data's
redistribution rights must be established independently; public availability
alone is not permission to redistribute it.

### 4. Recompute the frozen experiment

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
bash scripts/replay.sh FD001 both
# After FD001 works, optionally: bash scripts/replay.sh all both
```

The reproduction creates outputs under `04_results/` and SHA-256 verified ZIPs
under `04_results/ZIP/`. The reference checker is a *coarse regression screen*,
not a claim of bitwise identical predictions or full inferential validation.
No code is executed against the preexisting v1.0.0 results.

## Interpretation and known limitations

Across the original four datasets, the addition of RPDgep increases pooled
Average Precision relative to the baseline. Replicas on independently
regenerated NASA CSVs preserve that direction, but not all model probabilities
and secondary-model APs agree to the final bit. RPDgep does **not** universally
outperform the underlying `gamma,eta` components. Stage 13 alarm outcomes are
retrospective, not field alarm false-positive rates.

## Release and citation

This is an **unpublished candidate**. A new GitHub tag and a *version-specific*
Zenodo record should be created **only after** license, provenance, and source
review. Until then, do not attach the historical v1.0.0 DOI as if it identified
this directory. See `RELEASE_CHECKLIST.md`.
