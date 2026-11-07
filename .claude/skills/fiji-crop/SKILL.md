---
name: fiji-crop
description: Cut TIFF stacks for Fiji around a wk coordinate (EM, affinity as the graph saw it, r6 myelin mask, v3/v4 segmentation, the centre object) to inspect a merge or split. Use when Nghia gives a wk coordinate and asks to see the affinity / crop / tif.
---

# Fiji crop around a wk coordinate

```bash
cd /gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity
srun -p devel -c 2 --mem=24G -t 30 /gpfs/radev/home/yf354/.conda/envs/magneton/bin/python -u export_fiji_merge.py <wk x> <wk y> <wk z> [half xy px = 300] [half z = 20]
```

- Run it on devel, not the login node, which kills Python that loads volumes.
- Output: `/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/pairing_graph_v4/analysis/compare_v3/view_merge_<x>_<y>_<z>_fiji/`:
  - `em.tif`;
  - `aff_mean.tif`, `aff_z.tif`, `aff_xy.tif`: the affinity with the blank-section patch pasted, as the graph saw it;
  - `mask_r6.tif`;
  - `v3.tif`, `v4.tif`: float32 renumbered, with IDs in `*_ids.csv` (ImageJ has no uint32);
  - `v4_centre_object.tif`.
- The script prints the object at the centre in v3 and v4, the v3 objects under the v4 centre object, and how many
  patch tiles the box touches.
- Check whether a joined v3 object holds cell B skeleton nodes: `grep <id> pairing_graph_v3/cellB_skel_segs.csv`.

Tell Nghia the **full folder path**, which slice is the centre (slice 0 = wk z − half z), and the pixel → wk mapping
the script prints. To see segmentation and EM together in wk instead, use the sharded layers directly.
