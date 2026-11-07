#!/usr/bin/env python3
"""Naive tile plan builder — downsample-first, cheap dilate.

For each grid tile (32x32x16 seg vox = 256x256x32 vox @ 8x8x30 nm), reduce
"any seg voxel = target?" to one bool → tile-shape mini-mask (few MB).
Dilate that mini-mask by ceil(seg_dilation_nm / tile_nm) tiles in each dim.
Include every True tile.

Cell-agnostic: pass any skel NML + any precomputed seg + a target ID.

Output JSON matches build_cellB_tile_plan.py's schema.
"""
import argparse, os, json, time
import numpy as np
import xml.etree.ElementTree as ET
from scipy.ndimage import binary_dilation
from cloudvolume import CloudVolume


SCALE_SKEL_NM = (4, 4, 30)
INF_RES_NM = (8, 8, 30)


def load_skel_bbox_nm(nml_path, pad_nm=2000):
    tree = ET.parse(nml_path); root = tree.getroot()
    xs, ys, zs = [], [], []
    for n in root.findall('.//node'):
        xs.append(int(n.get('x')) * SCALE_SKEL_NM[0])
        ys.append(int(n.get('y')) * SCALE_SKEL_NM[1])
        zs.append(int(n.get('z')) * SCALE_SKEL_NM[2])
    return (min(xs)-pad_nm, min(ys)-pad_nm, min(zs)-pad_nm), \
           (max(xs)+pad_nm, max(ys)+pad_nm, max(zs)+pad_nm)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skel-nml", required=True)
    ap.add_argument("--seg-precomputed", required=True)
    ap.add_argument("--seg-target-id", type=int, default=1)
    ap.add_argument("--seg-dilation-nm", type=float, default=3000)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tile-vox", default="256,256,32",
                    help="X,Y,Z at inf res (default 256,256,32)")
    ap.add_argument("--skel-pad-nm", type=float, default=2000)
    ap.add_argument("--slab-z-tiles", type=int, default=32,
                    help="Read this many z-tile rows per slab (default 32)")
    args = ap.parse_args()

    TILE = tuple(int(v) for v in args.tile_vox.split(","))
    tile_x, tile_y, tile_z = TILE
    tile_nm = (tile_x * INF_RES_NM[0], tile_y * INF_RES_NM[1],
               tile_z * INF_RES_NM[2])
    print(f"tile size (nm): {tile_nm}")

    # Tile-level dilation in tiles: ceil(dilation_nm / tile_nm) per axis
    dil_tx = int(np.ceil(args.seg_dilation_nm / tile_nm[0]))
    dil_ty = int(np.ceil(args.seg_dilation_nm / tile_nm[1]))
    dil_tz = int(np.ceil(args.seg_dilation_nm / tile_nm[2]))
    print(f"tile-level dilation (dx,dy,dz): {dil_tx}, {dil_ty}, {dil_tz}")

    (xn0, yn0, zn0), (xn1, yn1, zn1) = load_skel_bbox_nm(
        args.skel_nml, pad_nm=args.skel_pad_nm)
    print(f"skel bbox nm: {(xn1-xn0)/1000:.1f} x {(yn1-yn0)/1000:.1f} x "
          f"{(zn1-zn0)/1000:.1f} um")

    xmin, ymin, zmin = xn0//4, yn0//4, zn0//30
    xmax, ymax, zmax = (xn1+3)//4, (yn1+3)//4, (zn1+29)//30
    vx = (xmax-xmin) * 4 // INF_RES_NM[0]
    vy = (ymax-ymin) * 4 // INF_RES_NM[1]
    vz = (zmax-zmin) * 30 // INF_RES_NM[2]
    print(f"inference grid: {vx} x {vy} x {vz}")

    seg_cv = CloudVolume(f"file://{args.seg_precomputed}", mip=0,
                         progress=False, fill_missing=True)
    seg_res = tuple(int(r) for r in seg_cv.resolution)
    bmin = tuple(int(v) for v in seg_cv.bounds.minpt)
    bmax = tuple(int(v) for v in seg_cv.bounds.maxpt)
    print(f"seg native res: {seg_res} nm/vox")

    # Grid tile counts
    nx = (vx + tile_x - 1) // tile_x
    ny = (vy + tile_y - 1) // tile_y
    nz = (vz + tile_z - 1) // tile_z
    dense = nx * ny * nz
    print(f"dense grid tiles: {nx} x {ny} x {nz} = {dense:,}")

    # Seg-vox per tile
    tsx = tile_nm[0] // seg_res[0]
    tsy = tile_nm[1] // seg_res[1]
    tsz = tile_nm[2] // seg_res[2]
    print(f"one grid tile = {tsx}x{tsy}x{tsz} seg vox")

    # Absolute seg-vox range covering skel bbox (no halo needed — dilation
    # is applied in tile space, not seg space)
    sx0 = max(bmin[0], xn0 // seg_res[0])
    sx1 = min(bmax[0], (xn0 // seg_res[0]) + nx * tsx)
    sy0 = max(bmin[1], yn0 // seg_res[1])
    sy1 = min(bmax[1], (yn0 // seg_res[1]) + ny * tsy)
    grid_seg_x0 = xn0 // seg_res[0]
    grid_seg_y0 = yn0 // seg_res[1]

    # mini_mask[tx,ty,tz] = does any seg vox in tile (tx,ty,tz) hit target?
    mini_mask = np.zeros((nx, ny, nz), dtype=bool)

    n_slabs = (nz + args.slab_z_tiles - 1) // args.slab_z_tiles
    t_total = time.time()
    for slab_ix in range(n_slabs):
        iz0 = slab_ix * args.slab_z_tiles
        iz1 = min(nz, iz0 + args.slab_z_tiles)
        # Seg-vox z range for this slab
        sz0 = max(bmin[2], zn0 // seg_res[2] + iz0 * tsz)
        sz1 = min(bmax[2], zn0 // seg_res[2] + iz1 * tsz)
        if sz1 <= sz0: continue

        t0 = time.time()
        seg = np.squeeze(np.asarray(seg_cv[sx0:sx1, sy0:sy1, sz0:sz1]))
        if seg.ndim == 2: seg = seg[:, :, None]
        seg_bin = (seg == args.seg_target_id)
        t1 = time.time()
        if not seg_bin.any():
            print(f"  slab {slab_ix+1}/{n_slabs}: empty  read {t1-t0:.1f}s",
                  flush=True)
            continue

        # Reduce to mini via block_reduce: for each (tsx,tsy,tsz) block, any()
        # Pad seg_bin to multiple of (tsx,tsy,tsz) so reshape works cleanly.
        sx_pad = (-seg_bin.shape[0]) % tsx
        sy_pad = (-seg_bin.shape[1]) % tsy
        sz_pad = (-seg_bin.shape[2]) % tsz
        if sx_pad or sy_pad or sz_pad:
            seg_bin = np.pad(seg_bin,
                             [(0, sx_pad), (0, sy_pad), (0, sz_pad)],
                             mode='constant')
        nxs = seg_bin.shape[0] // tsx
        nys = seg_bin.shape[1] // tsy
        nzs = seg_bin.shape[2] // tsz
        mini_slab = seg_bin.reshape(nxs, tsx, nys, tsy, nzs, tsz).any(
            axis=(1, 3, 5))
        t2 = time.time()

        # mini_slab is indexed by (tile_x_rel_to_grid_seg_x0, ...);
        # translate to grid-tile indices in mini_mask.
        # grid tile 0 in seg-vox starts at grid_seg_x0. Our slab starts at sx0.
        # So offset = (sx0 - grid_seg_x0) / tsx. For our default sx0 == grid_seg_x0.
        off_x = (sx0 - grid_seg_x0) // tsx
        off_y = (sy0 - grid_seg_y0) // tsy
        off_z = iz0  # slab starts exactly at iz0 in tile-z coords
        # place into mini_mask
        ex = min(mini_mask.shape[0] - off_x, mini_slab.shape[0])
        ey = min(mini_mask.shape[1] - off_y, mini_slab.shape[1])
        ez = min(mini_mask.shape[2] - off_z, mini_slab.shape[2])
        if ex <= 0 or ey <= 0 or ez <= 0:
            continue
        mini_mask[off_x:off_x+ex, off_y:off_y+ey, off_z:off_z+ez] |= \
            mini_slab[:ex, :ey, :ez]
        print(f"  slab {slab_ix+1}/{n_slabs} iz[{iz0}..{iz1}): "
              f"read {t1-t0:.1f}s  reduce {t2-t1:.1f}s  "
              f"seg_vox={int(seg_bin.sum())}  tiles_marked_so_far="
              f"{int(mini_mask.sum())}", flush=True)

    print(f"pre-dilate: {int(mini_mask.sum()):,} tiles have target seg voxels")

    # Anisotropic dilation. binary_dilation iterations with a cross default
    # is manhattan expansion of iterations vox. To get ~sphere-like anisotropy
    # with (dil_tx, dil_ty, dil_tz), take iterations = max, plus mask with a
    # per-axis check; simplest is repeated iterations = max(dil_tx,dil_ty,dil_tz)
    # then mask to axis-aligned box. Good enough for a naive plan.
    n_iter = max(dil_tx, dil_ty, dil_tz)
    if n_iter > 0:
        t0 = time.time()
        mini_dil = binary_dilation(mini_mask, iterations=n_iter)
        print(f"tile-level dilation iters={n_iter}: {time.time()-t0:.2f}s "
              f"({int(mini_dil.sum()):,} tiles)")
    else:
        mini_dil = mini_mask

    # Emit
    tiles = []
    idxs = np.argwhere(mini_dil)
    for ix, iy, iz in idxs:
        ox0 = int(ix) * tile_x; ox1 = min(ox0 + tile_x, vx)
        oy0 = int(iy) * tile_y; oy1 = min(oy0 + tile_y, vy)
        oz0 = int(iz) * tile_z; oz1 = min(oz0 + tile_z, vz)
        tiles.append([ox0, oy0, oz0, ox1, oy1, oz1])

    print(f"\ntotal tiles: {len(tiles):,} of {dense:,} dense "
          f"({100*(1-len(tiles)/dense):.1f}% saved)  "
          f"took {time.time()-t_total:.1f}s")
    print(f"EM to fetch: {len(tiles)*tile_x*tile_y*tile_z/1e9:.2f} GB uint8")

    plan = {
        "wk_bbox_origin_vox": [int(xmin), int(ymin), int(zmin)],
        "wk_bbox_size_vox":   [int(xmax-xmin), int(ymax-ymin), int(zmax-zmin)],
        "inf_res_nm":         list(INF_RES_NM),
        "tile_vox":           list(TILE),
        "seg_dilation_nm":    args.seg_dilation_nm,
        "n_tiles":            len(tiles),
        "n_slabs":            n_slabs,
        "dense_grid_tiles":   dense,
        "planner":            "naive_v2_downsample_first",
        "tile_dilation_iters": n_iter,
        "tiles":              tiles,
    }
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    with open(args.dst, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
