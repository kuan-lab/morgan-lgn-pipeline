#!/usr/bin/env python3
"""Phase 2: fetch raw EM from BossDB for each tile in the tile plan.

Writes into a local sparse precomputed (only tile-chunks materialized;
non-tile chunks remain absent — CloudVolume readers see zeros for those).

For SLURM array parallelism: --task-idx N --n-tasks K splits the tile list
into K roughly-equal shards; each task fetches its shard.
"""
import argparse, json, os, time
import numpy as np
from cloudvolume import CloudVolume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-plan", required=True)
    ap.add_argument("--src", required=True, help="BossDB URL (raw EM)")
    ap.add_argument("--src-mip-key", default="8_8_30")
    ap.add_argument("--dst", required=True, help="Local sparse precomputed dir")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--pad-vox", default="64,64,8",
                    help="Halo to fetch around each tile (X,Y,Z inf-vox). "
                         "Prevents zero-padding in Phase 3 model context.")
    args = ap.parse_args()
    pad_x, pad_y, pad_z = [int(v) for v in args.pad_vox.split(",")]

    with open(args.tile_plan) as f:
        plan = json.load(f)
    ox_wk, oy_wk, oz_wk = plan['wk_bbox_origin_vox']    # wk 4x4x30 vox
    sx_wk, sy_wk, sz_wk = plan['wk_bbox_size_vox']
    tile_vox = plan['tile_vox']                           # (x, y, z) at 8x8x30 nm
    inf_res = plan['inf_res_nm']                          # (8, 8, 30)
    tiles = plan['tiles']
    print(f"plan: {len(tiles)} tiles, tile size {tile_vox} vox @ {inf_res} nm")
    print(f"wk bbox: origin {(ox_wk, oy_wk, oz_wk)}  size {(sx_wk, sy_wk, sz_wk)}")

    # Local volume dimensions at inf-res
    vx = sx_wk * 4 // inf_res[0]
    vy = sy_wk * 4 // inf_res[1]
    vz = sz_wk * 30 // inf_res[2]
    print(f"local precomputed shape: {vx} x {vy} x {vz} vox @ {inf_res} nm")

    # Set up source
    probe = CloudVolume(args.src, mip=0, progress=False, use_https=True, fill_missing=True)
    scale_ix = None
    for i, sc in enumerate(probe.info['scales']):
        if sc['key'] == args.src_mip_key:
            scale_ix = i; break
    src = CloudVolume(args.src, mip=scale_ix, progress=False,
                       use_https=True, fill_missing=True, parallel=4)
    src_res = tuple(int(r) for r in src.resolution)
    assert src_res == tuple(inf_res), f"src {src_res} != plan {inf_res}"
    print(f"src at {src_res} nm, scale ix {scale_ix}")

    # Set up destination (create only on task 0 or if not exists)
    info = CloudVolume.create_new_info(
        num_channels=1, layer_type='image', data_type='uint8', encoding='raw',
        resolution=list(inf_res),
        voxel_offset=[0, 0, 0],
        chunk_size=[128, 128, 32],
        volume_size=[int(vx), int(vy), int(vz)],
    )
    os.makedirs(args.dst, exist_ok=True)
    dst = CloudVolume(f"file://{args.dst}", info=info, mip=0, compress=False,
                       fill_missing=True, parallel=1, non_aligned_writes=True)
    dst.commit_info()

    # Shard tile list
    my_tiles = tiles[args.task_idx::args.n_tasks]
    print(f"this task ({args.task_idx}/{args.n_tasks}): {len(my_tiles)} tiles")

    # BossDB source coords: wk bbox origin in wk vox → nm → src vox at inf res
    # since we're at inf_res 8x8x30 and wk is 4x4x30, src vox = wk*4/8 for xy, wk*30/30 for z
    src_ox_vox = ox_wk * 4 // inf_res[0]
    src_oy_vox = oy_wk * 4 // inf_res[1]
    src_oz_vox = oz_wk * 30 // inf_res[2]

    total_bytes = 0
    t0 = time.time()
    for idx, tile in enumerate(my_tiles):
        ox0, oy0, oz0, ox1, oy1, oz1 = tile
        # Expand by pad in each direction, clip to local volume bounds
        px0 = max(0, ox0 - pad_x); px1 = min(int(vx), ox1 + pad_x)
        py0 = max(0, oy0 - pad_y); py1 = min(int(vy), oy1 + pad_y)
        pz0 = max(0, oz0 - pad_z); pz1 = min(int(vz), oz1 + pad_z)
        # Fetch source at absolute vox (padded region)
        sx0 = src_ox_vox + px0; sx1 = src_ox_vox + px1
        sy0 = src_oy_vox + py0; sy1 = src_oy_vox + py1
        sz0 = src_oz_vox + pz0; sz1 = src_oz_vox + pz1
        arr = np.asarray(src[sx0:sx1, sy0:sy1, sz0:sz1])
        # Write padded region to local (adjacent tiles' halos overlap harmlessly)
        dst[px0:px1, py0:py1, pz0:pz1] = arr
        total_bytes += arr.nbytes
        if (idx+1) % 50 == 0 or idx == len(my_tiles)-1:
            dt = time.time() - t0
            mbps = total_bytes/1e6/max(dt, 0.01)
            print(f"  {idx+1}/{len(my_tiles)} tiles  {total_bytes/1e6:.0f} MB  "
                  f"{mbps:.0f} MB/s  elapsed {dt:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"\ndone: {len(my_tiles)} tiles, {total_bytes/1e6:.1f} MB, {dt:.0f}s "
          f"({total_bytes/1e6/max(dt,0.01):.0f} MB/s avg)")


if __name__ == "__main__":
    main()
