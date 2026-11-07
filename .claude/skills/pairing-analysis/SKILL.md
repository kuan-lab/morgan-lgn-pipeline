---
name: pairing-analysis
description: Run synapse pairing and the cell B connectivity analysis on a graph segmentation version, then compare it with the previous version and check whether changes are real or merges. Use for "redo the network analysis", "pair on vN", "compare v3 and v4", "are these new inputs real".
---

# Pairing + network analysis

Folder: `/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/`.

## Run

1. `mkdir pairing_graph_vN` and copy `chunks.csv` and `skeleton_nodes.npy` from `pairing_graph_v3/`.
2. Copy `run_pairing_graph_v4.sh` to `_vN.sh` and set `PAIR_RUN`, `GRAPH_SEG` (the vN sharded segmentation) and the job name.
   It runs `pair_chunks.sh` (5,395 chunks; each runs `box_pipeline/step3_seg_and_pair.py`), then `collect_cellB.py`,
   `analyze_cellB.py`, `check_pairs.py` and `check_triad_rgc_lin.py`. All of these read `PAIR_RUN`.
   About 6 h on bigmem with 48 workers. Chunks with a `.done` flag are skipped on rerun.
3. Not run by the chain, so run them separately if needed: the bouton-level clustering (`bouton_clustering.py`,
   `bouton_connectivity.py`), the FFI estimate (`ffi_signal.py`) and the local check.

## Compare with the previous version

`compare_pairing_v3_v4.py` (v3/v4 are hard-coded; copy it and change `A`/`B` for other pairs). It writes
`pairing_graph_vN/analysis/compare_v3/`:
- `input_sets.csv`: kept / new / lost inputs onto cell B, distance to the skeleton, and the VAST label;
- `post_label_by_set.csv`;
- `merge_sites_v3_object_into_cellB.csv`;
- `sites_to_view.csv`;
- `rgc_new_in_v4.csv` / `rgc_lost_from_v3.csv`.

Signs of a merge rather than a real gain:
- new inputs are far from the skeleton (a real input's median is ~0.4 µm; merges are ~2+ µm);
- only ~1% carry the VAST cell 201 label (real inputs: ~45%);
- they come from a few large objects that had no skeleton nodes in the earlier version.

Don't report "more inputs" before running this check. Then make check files (skill `wk-check-files`) so Nghia
can confirm by eye.

## Report

A v_old vs v_new table of headline numbers (`cellB_summary.json`), the merge test above, which analyses changed
significance, and the full path of the output folder.
