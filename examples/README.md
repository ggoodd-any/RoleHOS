# Curated reviewer examples

The six local examples demonstrate good segmentation results: two each from ID, OOD and mini-HOI4D. We evaluated 64 evenly spaced positions in each sorted image list with the published full model, seed 0, and shortlisted high-overlap predictions containing both hands and manipulated objects (at least 1,500 pixels each at 448 × 448). We then inspected RGB, ground truth and predictions, selecting varied scenes and interaction roles. The six selected examples were also checked with seeds 1 and 2.

These are deliberately selected successful cases, not a representative sample or a new benchmark. Full-test and paper metrics are unchanged. The selection score is a per-image mean IoU over evaluable foreground roles with nonempty prediction/ground-truth union (Mini excludes LO); it is not the paper's dataset-level mIoU. `manifest.json` records this score, role IoUs, sample identity, candidate index, checkpoint identity and image/label hashes.

The Gradio interface accepts any RGB upload; model predictions and visual overlays are generated at 448 × 448. The downloadable grayscale PNG retains class IDs 0–5. Ground-truth copies in `labels/` support offline inspection and are never provided to the network.

Public code archives omit image/label copies until redistribution terms are confirmed. Users with the licensed dataset can restore the exact gallery with `python -m tools.import_examples --data-root /path/to/egohos`. Original dataset terms apply, not the root MIT license.
