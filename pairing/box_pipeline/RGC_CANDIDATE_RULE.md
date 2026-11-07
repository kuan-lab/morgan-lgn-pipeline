# How RGC bouton candidates are picked (box pipeline, 2026-10-02)

> **2026-10-16:** later runs (depth comparison, mito v4 comparison) used a **4.5 %** mito cutoff (`make_rgc_tifs.py --mito 0.045`), not the 10 % below. Overview of the bouton steps: `BOUTON_GUIDE.md`.

This is a rule of thumb, not a classifier. Nothing here has been checked against ground truth yet; that is what we would like help with.

## Data it runs on

Nine boxes of Morgan LGN EM, each 10 × 10 × 9 µm (1250 × 1250 × 300 voxels at 8 × 8 × 30 nm), three each at 50, 100 and 200 µm below the bottom of the myelin layer. Blank (black) sections are removed before inference and their labels copied back from the neighbouring section afterwards.

| Input | Model | How it is used |
|---|---|---|
| Neuron segments | v4 affinity model + waterz (2D fragments, threshold 0.2) | the unit that gets classified |
| Synapses | synapse v6ft, checkpoint 25000 | instances: seed > 0.9, foreground > 0.2, seed ≥ 50 voxels, instance ≥ 300 voxels |
| Mito | mito v3, checkpoint 80000 | a voxel counts as mito if probability > 0.5 |

Each synapse gets a post segment (the segment it overlaps most) and a pre segment (the first other segment hit along the synapse's normal, stepping 30 nm at a time). Segments under 1000 voxels cannot be pre or post.

## The rule

A whole waterz segment is an RGC candidate if all four hold:

| Criterion | Value | Why |
|---|---|---|
| Presynaptic to at least 1 synapse | ≥ 1 | without it there are too many false positives |
| Mito fraction of the segment | ≥ 10 % of its voxels | separates mito-packed terminals from everything else |
| Volume | ≥ 0.1 µm³ (52,083 voxels), no upper limit | drops fragments |
| Not mostly postsynaptic | post count ≤ max(pre count, 2) | drops dendrites that were wrongly given a few "pre" synapses |

There is no other clean-up: no morphological filtering, no myelin filter, no manual curation. Synapses with quality flags (long ray, single section, cut by the box) still count toward the pre and post counts.

## Where the numbers come from

- **10 % mito** was raised from 5 % after one myelin segment (6.3 % mito) got through. It is tuned on that single example.
- **≥ 1 synapse as pre** was kept on purpose after looking at masks without it.
- **No upper size limit**: a 10 µm³ cap wrongly removed two large mito-rich axons (18 and 33 µm³, 22 and 27 synapses as pre).

## How sensitive the count is to the mito cutoff

All nine boxes pooled, other criteria fixed. 5,827 segments pass the non-mito criteria; 5,076 of them contain no mito at all.

| Mito cutoff | Candidates | of which ≥ 0.5 µm³ | of which pre to ≥ 2 synapses |
|---|---|---|---|
| any mito (> 0) | 751 | – | – |
| 1 % | 415 | 322 | 245 |
| 2 % | 380 | 304 | 236 |
| 5 % | 344 | 287 | 226 |
| **10 % (current)** | **291** | 255 | 206 |
| 15 % | 235 | 212 | 174 |
| 20 % | 152 | 138 | 111 |
| 30 % | 24 | 20 | 13 |

The distribution has two groups: 336 segments with under 1 % mito, and a broad group between 5 % and 30 % that peaks at 20–30 %. The 10 % cutoff sits inside the second group, so it does remove real-looking candidates: 89 segments fall between 2 % and 10 %, median 0.55 µm³, 30 of them pre to two or more synapses. A cutoff near 2 % would sit in the gap between the two groups instead.

## Known weaknesses

- **Myelin.** The synapse and mito models both fire on some myelin. Mean EM brightness of the segment does not separate it.
- **Segmentation errors pass straight through.** A bouton merged with a dendrite fails the "not mostly postsynaptic" test; a bouton split from its mito fails the mito test. Thin axons fragment, so one axon can appear as several candidates.
- **Box edges.** Segments cut by the box are kept and marked `truncated_at_box`. In `d050_a` there is also a 7-section blank gap (z 172–178) where segments are cut; the candidate table does not flag that cut.
- **"RGC" is an assumption.** Large, mito-rich, multisynaptic terminals are what RGC boutons should look like in LGN, but no one has confirmed these are RGC and not another large terminal type.
- **Per-box rate varies a lot** (8 to 51 candidates per box), so recall may differ between regions.

## What ground truth would let us measure

1. **Recall:** in one or two boxes, every RGC bouton marked (a point per bouton is enough), so we can count how many the rule misses.
2. **Precision:** for a sample of our candidates, a yes/no on whether each is an RGC bouton. The per-candidate table already has wk coordinates.
3. **Mito model check:** it marks 0.4–2.7 % of tissue as mito depending on the box. A small mito annotation in one of these boxes would tell us if it under-calls.

## Files

`box_pipeline/boxes_z300_noblank/<box>/`
- `rgc_candidates_v2.tif`: mask at 8 × 8 × 30 nm, value = segment ID
- `rgc_candidates_v2.csv`: volume, mito fraction, pre and post counts, wk coordinates per candidate
- `seg.tif`, `syn_inst.tif`, `mito_prob.tif`, `em.tif`: the inputs, same grid
- `box.json`: box position (`wk_origin` is the top-left corner in wk 4 × 4 × 30 coordinates)

`box_pipeline/analysis/boxes_z300_noblank/`
- `rgc_candidates_v2_wk.nml`: one node per candidate plus the nine bounding boxes
- `rgc_presyn_segments.csv`: every segment passing the non-mito criteria, with its mito fraction
- `rgc_mito_sweep.csv`: the table above

Scripts: `make_rgc_tifs.py` (the rule), `rgc_threshold_sweep.py` (the sweep).
