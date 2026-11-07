# Validation scope

- Frozen source hash audit: 17 files, with 16 unchanged and one documented
  Stage13 export-precision variant.
- Four groups totaling 25 synthetic/mathematical tests were run in a separate
  Python 3.13.5 environment during release-candidate construction.
- The official NASA four-dataset clean-room rerun was performed previously
  on e2m with Python 3.14.7, logged in IJPHM_031. The candidate bundle itself
  has not been independently rerun on all 709 real engine trajectories.
- Reference metrics are stored as compact tables; no raw NASA observations,
  per-cycle predictions, or proprietary model binaries are included.
