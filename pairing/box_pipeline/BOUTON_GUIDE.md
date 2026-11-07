# Boutons and RGC bouton candidates: what the code does and where

How the pipeline turns segmentation + synapses (+ mito) into a table of boutons, and which boutons it calls RGC candidates. Scripts are in `box_pipeline/` unless stated; settings in `box_pipeline/config.py`. Status (2026-10-16): working on the depth boxes; **no ground truth yet** for bouton boundaries or for the RGC call.

Three different things get called "bouton" in this project; keep them apart:

| Name | What it is | Where it is made |
|---|---|---|
| **Bouton (box pipeline)** | the part of a presynaptic segment around its synapses, cut at thin necks | `step3_seg_and_pair.py` → `boutons.csv`, `bouton_labels.tif` |
| **RGC candidate** | a whole presynaptic segment that passes the mito rule | `make_rgc_tifs.py` → `rgc_candidates_v2<tag>.csv` |
| **Bouton → cell B pair (whole cell)** | all active zones from one presynaptic segment onto cell B, grouped | `cellB_connectivity/check_pairs.py` → `cellB_rgc_boutons.csv` |

## Inputs

From the box pipeline steps 1–3 (see `README.md`): neuron segmentation `seg.tif`, synapse instances with pre/post pairing (`synapses.csv`), and for the RGC call the mito probability `mito_prob.tif` (mito v3, a voxel is mito if probability > 0.5).

## Step A: cut presynaptic segments into boutons (`step3_seg_and_pair.py`, last section)

For every segment that is presynaptic to at least one synapse:
1. Distance transform inside the segment, in nm (8 × 8 × 30).
2. **Cores** = voxels deeper than `BOUTON_NECK_NM` = 100 nm from the segment's surface; cores under 100 voxels are dropped.
3. Watershed from the cores within the segment, so each core grows into one bouton. A segment with no core stays one bouton.
4. Each synapse goes to the bouton containing its presynaptic point (the ray's landing point), or the nearest bouton voxel.
5. Only boutons with at least one synapse are kept.

Outputs per box:
- `bouton_labels.tif`
- `boutons.csv`, with columns:
  - `bouton_id`, `pre_seg_id`
  - `vol_vox`, `vol_um3`
  - `syn_count`, `syn_ids`
  - centre in local and wk coordinates
  - `truncated_at_box` — use only `0` for size statistics
  - `pre_seg_n_pre`, `pre_seg_n_post` — how many synapses the whole segment is pre / post to; a weak RGC hint only

## Step B: statistics (`analyze_boxes.py`)

Pools all boxes of the current run into `analysis/<RUN>/`:
- `all_boutons.csv`, `all_synapses.csv`, `per_box.csv`, `per_depth.csv`;
- `by_depth.png`: bouton size, synapses per bouton, multisynaptic fraction, density.

Planned bouton features are listed in the memory notes: size, number of active zones, active-zone size, density, multisynaptic fraction.

## Step C: RGC candidates (`make_rgc_tifs.py`)

```bash
python make_rgc_tifs.py --mito 0.045 --tag _mito045     # 4.5 % = what the latest runs used
```

A whole waterz segment is a candidate if:

| Criterion | Value |
|---|---|
| Presynaptic to at least 1 synapse | ≥ 1 (without it there are far too many false positives) |
| Mito fraction of the segment | ≥ `--mito`. `RGC_CANDIDATE_RULE.md` documents 10 %; later runs and the mito v4 comparison used **4.5 %** |
| Volume | ≥ 0.1 µm³, no upper limit (a cap wrongly removed two large mito-rich axons) |
| Not mostly postsynaptic | post count ≤ max(pre count, 2) |

Output per box: `rgc_candidates_v2<tag>.csv` (segment, volume, mito fraction, pre/post counts, wk coordinates) and label tifs for checking in webKnossos. The reasoning and the sensitivity table are in `RGC_CANDIDATE_RULE.md`.

Related scripts:
- `rgc_threshold_sweep.py`: how many candidates at each mito cutoff.
- `rgc_depth.py _mito045`: candidates compared across depth.
- `compare_mito_v4.py <box>`: mito v3 against v4 on one box. v4 did not separate RGCs better.
- `analysis/boxes_z300_noblank/rgc_2to10pct_to_label.csv`: the ambiguous 2–10 % group, waiting for labels.

## Whole cell B (no mito pass there)

- A bouton → cell B pair is all active zones whose presynaptic point lies in the same presynaptic segment. `cellB_connectivity/check_pairs.py` writes `pairing/analysis/cellB_rgc_boutons.csv`: 491 RGC active zones collapse to 210 pairs, a median of 2 active zones per pair.
- RGC identity on cell B comes from the manual VAST labels, not from mito. Untraced RGC boutons are therefore missed.
- The presynaptic segment is the chunk-local one. The old global segmentation over-merges, so its IDs are not used to define a bouton.

## Known problems

- **Segmentation errors pass straight through.** A bouton merged with its target fails the "not mostly postsynaptic" test; a bouton split from its mito fails the mito test; thin axons fragment into several candidates.
- **The 100 nm neck threshold was never tested** against hand-drawn bouton boundaries.
- **The mito cutoff is tuned on a handful of examples** (one myelin segment at 6.3 %; Nghia's labels on a few segments). There's no recall or precision measurement yet.
- **"RGC" is an assumption:** a large, mito-rich, multisynaptic terminal. Nobody has confirmed these aren't another large terminal type.
- **Blank sections:** the boxes remove black sections only, not white or noise frames (see `cellB_connectivity/empty_frames.py`).
