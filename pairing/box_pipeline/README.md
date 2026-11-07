# box_pipeline

Characterize synapses and boutons in small EM boxes sampled at different depths.
Canonical home for inference + pairing code; older `run_infer_*.sh` / `scripts_scratch/` copies are superseded.

## Run

```bash
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
cd box_pipeline
# 1. add a row to boxes.csv: name,depth_um,wk_x,wk_y,wk_z   (box center, wk 4x4x30 coords)
python submit.py              # all boxes
python submit.py d100_a       # one box
python submit.py --cpu-only d100_a   # redo seg + pairing only
```

Each box submits a GPU job, then a CPU job that starts when the GPU job succeeds. Logs are in `logs/`.

## Steps

| Step | File | What |
|---|---|---|
| 1 | `step1_fetch_em.py` | BossDB `morgan2020/lgn/em` at 8x8x30 → `em.tif` |
| 2a | `infer_neuron_aff.py` | neuron affinity, v4 finetune, core+halo tiling → `aff.tif` (3,Z,Y,X) |
| 2b | `infer_synapse.py` | synapse prob, jj837 v6ft → `syn_prob.tif` |
| 2c | `infer_synapse.py` (mito checkpoint) | mito prob, jj837 mito v3 → `mito_prob.tif` |
| 3 | `step3_seg_and_pair.py` | waterz 2D thr 0.2 → `seg.tif`; binary watershed → `syn_inst.tif`; pre/post pairing; bouton split |

All settings (box size, model paths, thresholds) are in `config.py`.

## Outputs (`<RUN>/<name>/`)

`RUN` in `config.py` names the output folder, so a new box size never overwrites old results. `boxes/` = first run (100 sections, no mito); `boxes_z300/` = 300 sections + mito. Aggregate stats: `python analyze_boxes.py` → `analysis/<RUN>/`.

- `synapses.csv`: one row per synapse. `post_id` is the seg with most overlap, `pre_id` is the first seg hit along the AZ normal ray. `quality` flags: `long_ray` (>180 nm, likely FP), `single_slice`, `truncated_at_box`, `no_pre`.
- `synapses_wk.nml`: post→pre pairs in full-volume wk coords (load on the Morgan dataset).
- `boutons.csv`: pre segs split into boutons at 100 nm necks; only boutons with ≥1 synapse are kept. Use `truncated_at_box == 0` for size stats.
- `summary.json`: per-box counts, median bouton volume, multisynaptic fraction.

Local coords (`cz,cy,cx`) are 8x8x30 voxels inside the box. `wk_*` = `wk_origin + (2x, 2y, z)`.

## Known limits

- Neuron seg fragments thin neurites, so some pre/post IDs are fragments of the same cell.
- `pre_seg_n_pre` / `pre_seg_n_post` in `boutons.csv` are only a weak RGC hint. Actual RGC calls need mito/morphology (Jaeyee's mito work).
