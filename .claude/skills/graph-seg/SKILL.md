---
name: graph-seg
description: Run Tim's graph instance segmentation on our tile-plan affinity (whole cell B, a new cell, or a test box): choose the myelin config, set up a new version's paths, submit the per-core and global stages with the right memory, recover from OOM, verify. Use for "redo the segmentation", "new graph version", "test a myelin rule on a box".
---

# Graph segmentation (cell B wrapper around magneton)

Folder: `/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB/`. Driver: `run_cellB_graph.py`
(stages `fragments`, `edges`, then `merge_graph_fast.py` and `relabel_fast.py`). It wraps Tim's stages without editing
them: it pastes the blank-section patch, sets everything outside the tile plan to background, gzips the output, and
applies the myelin rule from the config.

## Myelin configs

| Config | Rule | Known problem |
|---|---|---|
| `config_cellB.yaml` (v1) | no mask | myelin becomes objects |
| `config_cellB_v2.yaml` (v2/v3) | mean affinity ≤ 0.4 → background | merges through thin membranes |
| `config_cellB_v4.yaml` (v4) | `myelin_open_radius: 6`, only thick low bands | merges onto cell B, about 2× the false inputs of v3 |

v3 is the reference. Some merges depend only on waterz's processing order (8-bit ties; see
`masktest/order_test.py`), so a single merge on a test box is not proof that a rule is better.

## Test a rule on a box first

`masktest/run_masktest.sh` runs 4 cores around `REG`, then `eval_box.py`, which reports merges against v1, myelin
left, and thin processes kept. Copy a `masktest/config_*.yaml` and change only the rule and the paths. Set
`NO_PATCH=1` only if the box has no patched tiles.

## New full version

1. Copy the config to `config_cellB_vN.yaml` and change *every* path: `fragments`, `output`, `fragments_dir`,
   `relabel_dir`, `edges_dir`, `lut_dir`. Copy `run_full_v4.sh` and `verify_full_v4.py`, replacing v4 with vN.
2. Check there are no stale done flags: `ckpt_relabel_fast_<basename of output>` must not exist. Stale flags once made
   every relabelled score 0.
3. Check memory: `scontrol show node r817u09n05 | grep AllocMem` (and n06). Slurm counts *requested* memory.
4. Submit three jobs chained with `afterok`:
   - `run_full_vN.sh percore` (fragments + edges): `-w <node> -c 28 --mem=1900G --export=ALL,NW=24`. Edges needs ~20 GB
     per worker on dense cores; 690 GB with 24 workers was OOM-killed.
   - `run_full_vN.sh global` (merge, relabel, shard, verify): bigmem 1450 G; v2 peaked at 1.23 TiB.
   - `run_pairing_graph_vN.sh` (see skill `pairing-analysis`).
5. On OOM: both stages resume per core, so resubmit with fewer workers or more memory. Never delete `edges/`.
6. Verify: `verify_full_vN.py` must print PASS (the sharded copy matches, every mip decodes, soma IDs as expected).

Fragments take ~5.5 h and edges ~8 h for whole cell B. Record the job chain in `cellB_connectivity/vN_chain_jobs.txt`.
