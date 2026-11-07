#!/usr/bin/env python3
"""Phase 1: build the blob-hugging tile plan for cell B.

Streams cell 201 seg (64x64x60 nm) per-Z-slab, dilates by 500 nm, projects
onto XY, computes 2D connected components, tiles each CC with 256x256x32
tiles centered on the CC bbox. Emits a JSON with the tile list.

The mask is NEVER materialized as a big 3D array — we work per-slab.

Output JSON schema:
  {
    "wk_bbox_origin_vox": [xmin, ymin, zmin],
    "wk_bbox_size_vox":   [sx, sy, sz],
    "inf_res_nm":         [8, 8, 30],
    "tile_vox":           [32, 128, 128],  # (z, y, x) matches model INPUT_SIZE
    "n_tiles":            N,
    "tiles": [
      # each tile is (ox0, oy0, oz0, ox1, oy1, oz1) in LOCAL bbox vox at inf_res
      [ox0, oy0, oz0, ox1, oy1, oz1],
      ...
    ]
  }
"""
import argparse, os, json, sys
import numpy as np
import xml.etree.ElementTree as ET
from scipy.ndimage import binary_dilation, label as cc_label
from cloudvolume import CloudVolume


SCALE_SKEL_NM = (4, 4, 30)   # cellB_skel.nml uses voxel * (4,4,30) nm
INF_RES_NM = (8, 8, 30)
TILE_VOX = (256, 256, 32)    # (x, y, z) at 8x8x30 nm


def load_skel_bbox_nm(nml_path, pad_nm=2000):
    tree = ET.parse(nml_path); root = tree.getroot()
    xs, ys, zs = [], [], []
    for n in root.findall('.//node'):
        xs.append(int(n.get('x'))*SCALE_SKEL_NM[0])
        ys.append(int(n.get('y'))*SCALE_SKEL_NM[1])
        zs.append(int(n.get('z'))*SCALE_SKEL_NM[2])
    x0, x1 = min(xs)-pad_nm, max(xs)+pad_nm
    y0, y1 = min(ys)-pad_nm, max(ys)+pad_nm
    z0, z1 = min(zs)-pad_nm, max(zs)+pad_nm
    return (x0, y0, z0), (x1, y1, z1)


def main():
    global TILE_VOX
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skel-nml", default="cellB_skel.nml")
    ap.add_argument("--seg-precomputed", default="precomputed_cell201_merged")
    ap.add_argument("--seg-target-id", type=int, default=1)
    ap.add_argument("--seg-dilation-nm", type=float, default=500)
    ap.add_argument("--dst", required=True, help="Output JSON path (or partial JSON path if --slab-range set)")
    ap.add_argument("--slab-range", default=None,
                    help="Optional 'START,STOP' subset of slabs (0-indexed, "
                         "half-open) for SLURM array parallelism.")
    ap.add_argument("--tile-vox", default=None,
                    help="Tile size at inference res (8x8x30 nm), 'X,Y,Z'. "
                         f"Defaults to {TILE_VOX}.")
    args = ap.parse_args()

    if args.tile_vox:
        TILE_VOX = tuple(int(v) for v in args.tile_vox.split(","))
        assert len(TILE_VOX) == 3, "--tile-vox must be X,Y,Z"

    # Bbox in nm (skel bbox + small pad)
    (xn0, yn0, zn0), (xn1, yn1, zn1) = load_skel_bbox_nm(args.skel_nml, pad_nm=2000)
    print(f"target bbox nm: x[{xn0}..{xn1}] y[{yn0}..{yn1}] z[{zn0}..{zn1}]")
    print(f"  extent: {(xn1-xn0)/1000:.1f} x {(yn1-yn0)/1000:.1f} x {(zn1-zn0)/1000:.1f} um")

    # Snap to wk 4x4x30 vox
    xmin, ymin, zmin = xn0//4, yn0//4, zn0//30
    xmax, ymax, zmax = (xn1+3)//4, (yn1+3)//4, (zn1+29)//30
    print(f"wk bbox 4x4x30 vox: [{xmin},{ymin},{zmin}] to [{xmax},{ymax},{zmax}]  "
          f"size: {xmax-xmin} x {ymax-ymin} x {zmax-zmin}")

    # Inference-resolution grid (8x8x30 nm) size relative to bbox
    vx = (xmax-xmin)*4 // INF_RES_NM[0]      # nm / 8 nm per vox
    vy = (ymax-ymin)*4 // INF_RES_NM[1]
    vz = (zmax-zmin)*30 // INF_RES_NM[2]
    print(f"inference grid (8x8x30 nm): {vx} x {vy} x {vz}")

    # Slabs at 8x8x30 nm resolution
    tile_x, tile_y, tile_z = TILE_VOX
    print(f"tile size (x,y,z): {tile_x} x {tile_y} x {tile_z}")

    # Set up seg reader
    seg_cv = CloudVolume(f"file://{args.seg_precomputed}", mip=0, progress=False,
                         fill_missing=True)
    seg_res = tuple(int(r) for r in seg_cv.resolution)
    bmin = tuple(int(v) for v in seg_cv.bounds.minpt)
    bmax = tuple(int(v) for v in seg_cv.bounds.maxpt)

    # We need seg dilation of `seg_dilation_nm` in XYZ. Z-dilation crosses
    # tile-slab boundaries — read enough Z to cover it.
    seg_z_pad = int(np.ceil(args.seg_dilation_nm / seg_res[2])) + 1
    seg_xy_pad = int(np.ceil(args.seg_dilation_nm / seg_res[0])) + 1

    # For each Z-slab of the OUTPUT tile grid (each slab = tile_z z-vox at 8x8x30 nm)
    # → produce list of blob tiles.
    tiles = []
    n_slabs = (vz + tile_z - 1) // tile_z
    if args.slab_range:
        slab_start, slab_stop = [int(v) for v in args.slab_range.split(",")]
        slab_stop = min(slab_stop, n_slabs)
        print(f"processing slab range [{slab_start}, {slab_stop}) of {n_slabs} total")
        slab_iter = range(slab_start, slab_stop)
    else:
        slab_iter = range(n_slabs)
    for slab_idx in slab_iter:
        # Slab z-range in LOCAL bbox vox at 8x8x30 nm
        loc_z0 = slab_idx * tile_z
        loc_z1 = min((slab_idx+1) * tile_z, vz)
        # Convert to absolute nm
        nm_z0 = zn0 + loc_z0 * INF_RES_NM[2]
        nm_z1 = zn0 + loc_z1 * INF_RES_NM[2]
        # Extended nm range for dilation
        nm_z0_ext = nm_z0 - args.seg_dilation_nm
        nm_z1_ext = nm_z1 + args.seg_dilation_nm

        # Convert to seg (64x64x60 nm) vox range (absolute, accounting for offset)
        sx0 = max(bmin[0], xn0 // seg_res[0] - seg_xy_pad)
        sx1 = min(bmax[0], (xn1 + seg_res[0]-1) // seg_res[0] + seg_xy_pad)
        sy0 = max(bmin[1], yn0 // seg_res[1] - seg_xy_pad)
        sy1 = min(bmax[1], (yn1 + seg_res[1]-1) // seg_res[1] + seg_xy_pad)
        sz0 = max(bmin[2], int(nm_z0_ext // seg_res[2]))
        sz1 = min(bmax[2], int((nm_z1_ext + seg_res[2]-1) // seg_res[2]))
        if sx1<=sx0 or sy1<=sy0 or sz1<=sz0:
            continue

        seg_slab = np.squeeze(np.asarray(seg_cv[sx0:sx1, sy0:sy1, sz0:sz1]))
        if seg_slab.ndim == 2: seg_slab = seg_slab[:,:,None]
        seg_bin = (seg_slab == args.seg_target_id)
        n_pos = int(seg_bin.sum())
        if n_pos == 0:
            print(f"  slab {slab_idx+1}/{n_slabs}: no cell B seg — skip", flush=True)
            continue

        # Fast dilation: iterative 3x3x3 (Manhattan-ish, close to spherical for
        # small radii). O(N * K * iters) with K=27 << O(N * K^3) for a big kernel.
        # Iterations = max dilation in voxels needed to cover seg_dilation_nm.
        n_iter = int(np.ceil(args.seg_dilation_nm / min(seg_res)))
        import time as _t
        _t0 = _t.time()
        seg_dil = binary_dilation(seg_bin, iterations=n_iter)
        _t1 = _t.time()
        print(f"    seg_dil iters={n_iter} took {_t1-_t0:.1f}s", flush=True)
        # Restrict to the CENTRAL z-range (drop the pad we read for dilation context)
        # Compute central z indices within seg_dil
        seg_slab_z0_nm = sz0 * seg_res[2]
        central_z0 = max(0, int((nm_z0 - seg_slab_z0_nm) // seg_res[2]))
        central_z1 = min(seg_dil.shape[2], int((nm_z1 - seg_slab_z0_nm + seg_res[2]-1) // seg_res[2]))
        if central_z1 <= central_z0:
            continue
        seg_central = seg_dil[:, :, central_z0:central_z1]

        # Project onto XY at seg resolution
        proj2d_seg = seg_central.any(axis=2)   # (seg_x, seg_y) bool
        if not proj2d_seg.any():
            print(f"  slab {slab_idx+1}/{n_slabs}: no dilated mask in central range")
            continue

        # Upsample seg's projection to inference-resolution grid (8x8 nm)
        # Simple nearest-neighbor: for each grid vox, find seg vox it's in
        seg_ratio_x = seg_res[0] / INF_RES_NM[0]   # 64/8 = 8
        seg_ratio_y = seg_res[1] / INF_RES_NM[1]   # 8
        # Compute abs bbox in seg vox that corresponds to local grid vox 0..vx
        # In-nm bbox is [xn0, xn1)
        # seg vox = nm / seg_res, so local grid 0 -> nm xn0 -> seg (xn0/seg_res[0])
        seg_x_start = xn0 / seg_res[0]     # float
        seg_y_start = yn0 / seg_res[1]
        # Grid indices at inf-res
        gx = (np.arange(vx) + 0.5) * INF_RES_NM[0] / seg_res[0] + seg_x_start - sx0
        gy = (np.arange(vy) + 0.5) * INF_RES_NM[1] / seg_res[1] + seg_y_start - sy0
        ix = np.clip(gx.astype(int), 0, proj2d_seg.shape[0]-1)
        iy = np.clip(gy.astype(int), 0, proj2d_seg.shape[1]-1)
        proj2d_inf = proj2d_seg[np.ix_(ix, iy)]   # (vx, vy) bool at inf-res

        if not proj2d_inf.any():
            continue

        # 2D CC on this projected mask
        cc, n_cc = cc_label(proj2d_inf)
        slab_tile_count = 0
        for cc_id in range(1, n_cc+1):
            xs, ys = np.where(cc == cc_id)
            xlo, xhi = int(xs.min()), int(xs.max())+1
            ylo, yhi = int(ys.min()), int(ys.max())+1
            wx, wy = xhi - xlo, yhi - ylo
            ntx = max(1, (wx + tile_x - 1) // tile_x)
            nty = max(1, (wy + tile_y - 1) // tile_y)
            total_x = ntx * tile_x; total_y = nty * tile_y
            gx0 = max(0, min(vx - total_x, xlo - (total_x - wx) // 2))
            gy0 = max(0, min(vy - total_y, ylo - (total_y - wy) // 2))
            for i in range(ntx):
                ox0 = gx0 + i*tile_x
                ox1 = min(ox0 + tile_x, vx)
                for j in range(nty):
                    oy0 = gy0 + j*tile_y
                    oy1 = min(oy0 + tile_y, vy)
                    tiles.append([ox0, oy0, loc_z0, ox1, oy1, loc_z1])
                    slab_tile_count += 1
        print(f"  slab {slab_idx+1}/{n_slabs}  z_loc[{loc_z0}..{loc_z1})  "
              f"CCs: {n_cc}  tiles: {slab_tile_count}  running total: {len(tiles)}",
              flush=True)

    # Dedupe (same tile bbox from different CCs in same slab)
    tiles_uniq = sorted({tuple(t) for t in tiles})
    tiles = [list(t) for t in tiles_uniq]
    print(f"\ntotal tiles (dedup): {len(tiles)}")

    dense_grid = ((vx + tile_x - 1)//tile_x) * ((vy + tile_y - 1)//tile_y) * n_slabs
    print(f"dense grid equivalent: {dense_grid}   savings: "
          f"{100*(1 - len(tiles)/dense_grid):.1f}%")

    # Fetch cost estimate (bytes uint8 per tile)
    bytes_per_tile = tile_x * tile_y * tile_z   # uint8
    total_bytes = len(tiles) * bytes_per_tile
    print(f"\nEM to fetch: {total_bytes/1e9:.2f} GB uint8 across {len(tiles)} tiles")

    # Write JSON
    plan = {
        "wk_bbox_origin_vox": [int(xmin), int(ymin), int(zmin)],
        "wk_bbox_size_vox":   [int(xmax-xmin), int(ymax-ymin), int(zmax-zmin)],
        "inf_res_nm":         list(INF_RES_NM),
        "tile_vox":           list(TILE_VOX),   # (x, y, z)
        "seg_dilation_nm":    args.seg_dilation_nm,
        "n_tiles":            len(tiles),
        "n_slabs":            n_slabs,
        "dense_grid_tiles":   dense_grid,
        "tiles":              tiles,
    }
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    with open(args.dst, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"\nwrote plan: {args.dst}")


if __name__ == "__main__":
    main()
