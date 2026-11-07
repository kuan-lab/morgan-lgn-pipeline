# morgan-lgn-pipeline

Code used to reconstruct the synaptic inputs of LGN cell B (VAST cell 201) in the Morgan mouse LGN EM volume
(BossDB `morgan2020/lgn/em`, 8 × 8 × 30 nm). It goes from a skeleton of one cell to a table of every synapse onto it
and who made it: neuron segmentation, synapse detection, pre/post pairing, connectivity analysis.

This is a record of the code as it was run on the Yale Misha cluster (2026-08 to 2026-11), not a packaged library.
Paths to data, models and environments are absolute cluster paths and are left as they were, so every script shows
exactly what it read and wrote. The neuron segmentation part is expected to be reworked.

## Pipeline

| # | Stage | Folder | Entry point | Output |
|---|---|---|---|---|
| 1 | Tile plan: 256-px tiles within 3 µm of the cell | `neuron_seg/tile_plan` | `run_build_tile_plan_naive.sh` | `cellB_tile_plan_3um.json` |
| 2 | Fetch EM for those tiles from BossDB | `neuron_seg/inference` | `run_fetch_em_3um.sh` (SLURM array) | sparse EM precomputed |
| 3 | Neuron affinity, tile by tile | `neuron_seg/inference` | `run_blob_infer_256_neuron_v4_3um.sh`, then `run_shard_neuron_v4_3um.sh` | `cellB_neuron_v4_3um_sharded` (3 channels: z, y, x) |
| 4 | Blank / empty sections: find them, re-infer the affected tiles without them | `neuron_seg/blank_sections` | `empty_frames.py`, `find_blank_blocks.py`, `run_patch_rebuild.sh` | `cellB_neuron_v4_3um_noblank_patch` |
| 5 | Graph segmentation (Tim's magneton stages): 2D supervoxels → waterz edges per core → global merge → relabel | `neuron_seg/graph_seg` | `run_full_v4.sh percore`, then `run_full_v4.sh global` | `graph_cellB_v4_seg_sharded` |
| 6 | Synapse probability, same tile plan | `synapse` | `run_blob_infer_synapse_v6ft_3um.sh`, `run_shard_syn_v6ft_3um.sh` | `cellB_pred_v6ft_3um_sharded` |
| 7 | Synapse instances + pre/post pairing per chunk | `pairing` | `run_pairing_graph_v4.sh` (calls `pair_chunks.sh` → `box_pipeline/step3_seg_and_pair.py`) | `pairing_graph_v4/chunks/*` |
| 8 | Collect, analyse, check against manual annotation | `analysis` | run at the end of `run_pairing_graph_v4.sh` | `pairing_graph_v4/analysis/*` |

`pairing/box_pipeline/` is also a standalone pipeline for small boxes (fetch, neuron + synapse + mito inference,
segmentation, pairing, boutons); see its own `README.md`. `tools/make_sharded_pyramid.py` shards any output for webKnossos.

### How the graph segmentation is wrapped

`neuron_seg/graph_seg/run_cellB_graph.py` runs Tim's stages without editing his code. It adds four things:

1. the blank-section patch pasted over the tiles it re-inferred;
2. supervoxels outside the tile plan set to background;
3. gzip-compressed output;
4. a myelin rule, which the config chooses.

The cores are 256 × 1024 × 1024 voxels with a halo of 32 × 128 × 128.

## Myelin settings (graph segmentation)

| Config | Rule | Known problem |
|---|---|---|
| `configs/config_v1_nomask.yaml` | none: every voxel gets a supervoxel | myelin becomes objects and merges cells |
| `configs/config_v3_voxel04.yaml` | voxels with mean affinity ≤ 0.4 → background | merges cells through thin low-affinity membranes |
| `configs/config_v4_open_r6.yaml` | only low regions that hold a 6-px disk (per-section opening) → background | fewer merges in a test box; on whole cell B about twice as many false inputs as v3 |

The current cell B results use **v3**. `neuron_seg/mask_test/` holds the box test behind the table (`eval_box.py`), and the
replay that finds the first cross-cell merge (`trace_first_cross.py`). It also has a permutation test
(`order_test.py`): it showed that some merges depend only on waterz's processing order, because 8-bit affinity
gives many tied scores.

## Things that bite

- **Coordinates.** webKnossos (wk) is 4 × 4 × 30 nm. The pipeline runs at 8 nm, so absolute x, y = wk / 2. The cell B
  intermediates (EM, affinity) are stored local, at absolute − (25114, 33058, 134). Segmentation outputs keep the
  absolute offset, and every new output must copy the source's `voxel_offset`, or it shows up at (0, 0, 0) in wk.
- **Blank sections** must be removed before inference. Some empty sections are white or noise, not black, so detect
  them by contrast (`empty_frames.py`). The gap wk z 2349–2357 is cut.
- **Never run dense inference on the whole cell.** Use the tile plan; dense output fills the quota.
- **Memory.** The edges stage needs ~20 GB per worker on dense cores, and the global merge needs ~1.3 TB (bigmem).
- **Waterz** needs `precompile_waterz.py` first when `HOME` is redirected to node scratch.
- **Synapse attribution:** a pre-side ray longer than 180 nm almost always means a false positive or a cut at the box
  edge. These are flagged and excluded.

## Dependencies (on Misha)

- magneton, not copied here:
  - graph stages: Tim's checkout `/gpfs/radev/home/yf354/magneton`, commit `b5b7115` (2026-10-14);
  - inference and box pipeline: kuan-lab/magneton commit `c59f012` (2026-07-05), at `cell_A_full/magneton_local` and `ntl5/magneton_morgan`.
- Conda envs: `/gpfs/radev/home/yf354/.conda/envs/magneton` (everything except sharding), `segneuron_cpu` (sharding).
- Models:
  - neuron: `cell_A_full/finetune_morgan_chunk1_v4_no_myelin/checkpoint_08000.pth.tar`, fine-tuned from `models/neuron_serial_section_morgan_v2.pth.tar`;
  - synapse: `jj837/synapse_model/training/checkpoints_v6ft/checkpoint_25000.pth.tar`;
  - mito: `models/mito_v3_jj837`.
- Data: EM from BossDB, cell B skeleton `CellB_skeleton.nml`, VAST manual segmentation + `segment_groups.csv` (cell identity and type).

Outputs live on `/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/` (large volumes) and
`/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/` (tables).
