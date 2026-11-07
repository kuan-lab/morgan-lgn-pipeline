---
name: tile-inference
description: Run neuron-affinity or synapse inference for a whole traced cell (or any large region) the sparse way: tile plan around the skeleton, EM fetch, per-tile inference, sharding with the right voxel_offset, decode check, blank-section patch. Use when asked to infer a new cell (e.g. cell 129), re-infer cell B, or build/rebuild the blank-section patch.
---

# Whole-cell sparse inference

Never use magneton's dense `--inference` for a whole cell. Ask the user before submitting GPU work, and give the tile
count, GPU-hours and disk estimate. All scripts are in `/gpfs/radev/project/kuan/ntl5/cell_A_full/` unless noted.

## Steps

1. **Tile plan**: `run_build_tile_plan_naive.sh` with `SKEL_NML=<cell skeleton>`, `SEG_PC=<VAST cell precomputed>`, `DIL=3000`
   (3 µm, so partner boutons are covered), `DST=magneton_inference/<cell>_tile_plan_3um.json`. Tiles are 256 px; the
   plan JSON has `tiles: [x0, y0, z0, x1, y1, z1]` in local 8 nm coords.
2. **EM fetch**: `run_fetch_em_3um.sh` (SLURM array, `--pad-vox 128,128,8` halo). Copy it and change the plan and `--dst`.
   The halo matters: without it, the model reads zeros at tile edges and hallucinates.
3. **Inference**:
   - neuron: `run_blob_infer_256_neuron_v4_3um.sh` (`blob_infer_from_plan_3ch.py`, v4 checkpoint);
   - synapse: `run_blob_infer_synapse_v6ft_3um.sh` (`blob_infer_from_plan.py`, jj837 v6ft ck25k).

   Before submitting, check GPU availability (`/hpc info gpu`). `gpu_devel` often starts at once for jobs under 4 h.
4. **Shard**: `run_shard_neuron_v4_3um.sh` / `run_shard_syn_v6ft_3um.sh` (`make_sharded_pyramid.py`, then the
   `voxel_offset` patch). The offset must be the source EM's, at every mip; cell B uses (25114, 33058, 134) at 8_8_30.
5. **Decode check**: read a few windows at tile centres and confirm nonzero values. Shard files can look normal and still
   decode to zeros.
6. **Blank sections**: `cellB_connectivity/empty_frames.py`, which finds black *and* white/noise frames by contrast. Then
   `find_blank_blocks.py` and `list_longgap_blocks.py` give the tile lists, and `run_patch_rebuild.sh` re-infers those tiles
   without the bad sections (`infer_cellB_noblank.py`, long-gap rule 3) into `<cell>_neuron_v4_3um_noblank_patch`.
   Graph segmentation pastes this patch over the affinity, so every graph run needs it. Never delete it.

## Report

The paths of the plan, EM, raw output, sharded output and patch (full absolute paths), the tile counts, and the
decode-check result.
