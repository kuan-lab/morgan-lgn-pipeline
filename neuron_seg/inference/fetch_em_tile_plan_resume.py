#!/usr/bin/env python3
"""Resume-capable variant of fetch_em_tile_plan.py — skips tiles whose padded
region is already fully covered on disk. Same CLI otherwise.
"""
import argparse, os, json, time
import numpy as np
from cloudvolume import CloudVolume


def chunk_files_for_bbox(dst_root, px0, px1, py0, py1, pz0, pz1,
                         chunk=(128, 128, 32), key="8_8_30"):
    """Enumerate expected raw-chunk file paths that cover [p*0, p*1)."""
    cx, cy, cz = chunk
    x0 = (px0 // cx) * cx
    y0 = (py0 // cy) * cy
    z0 = (pz0 // cz) * cz
    paths = []
    x = x0
    while x < px1:
        y = y0
        while y < py1:
            z = z0
            while z < pz1:
                fn = f"{x}-{x+cx}_{y}-{y+cy}_{z}-{z+cz}"
                paths.append(os.path.join(dst_root, key, fn))
                z += cz
            y += cy
        x += cx
    return paths


def all_chunks_exist(paths, min_size=1):
    for p in paths:
        try:
            st = os.stat(p)
        except FileNotFoundError:
            return False
        if st.st_size < min_size:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-plan", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--pad-vox", default="128,128,8", help="X,Y,Z pad")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    args = ap.parse_args()

    plan = json.load(open(args.tile_plan))
    tiles = plan['tiles']
    tile_vox = plan['tile_vox']
    inf_res = tuple(plan['inf_res_nm'])
    ox_wk, oy_wk, oz_wk = plan['wk_bbox_origin_vox']
    sx_wk, sy_wk, sz_wk = plan['wk_bbox_size_vox']
    vx = sx_wk * 4 // inf_res[0]
    vy = sy_wk * 4 // inf_res[1]
    vz = sz_wk * 30 // inf_res[2]

    pad_x, pad_y, pad_z = [int(v) for v in args.pad_vox.split(",")]

    src = CloudVolume(args.src, mip=0, progress=False, fill_missing=True,
                      use_https=True)
    src_res = tuple(int(r) for r in src.resolution)
    scale_ix = [i for i,s in enumerate(src.info['scales'])
                if tuple(int(r) for r in s['resolution']) == inf_res]
    if not scale_ix:
        raise RuntimeError(f"src has no scale at {inf_res} nm")
    src.mip = scale_ix[0]
    print(f"src at {src_res} nm, scale ix {scale_ix}")

    info = CloudVolume.create_new_info(
        num_channels=1, layer_type='image', data_type='uint8', encoding='raw',
        resolution=list(inf_res), voxel_offset=[0, 0, 0],
        chunk_size=[128, 128, 32], volume_size=[int(vx), int(vy), int(vz)],
    )
    os.makedirs(args.dst, exist_ok=True)
    dst = CloudVolume(f"file://{args.dst}", info=info, mip=0, compress=False,
                      fill_missing=True, parallel=1, non_aligned_writes=True)
    dst.commit_info()

    my_tiles = tiles[args.task_idx::args.n_tasks]
    print(f"this task ({args.task_idx}/{args.n_tasks}): {len(my_tiles)} tiles")

    src_ox_vox = ox_wk * 4 // inf_res[0]
    src_oy_vox = oy_wk * 4 // inf_res[1]
    src_oz_vox = oz_wk * 30 // inf_res[2]

    total_bytes = 0
    skipped = 0
    fetched = 0
    t0 = time.time()
    for idx, tile in enumerate(my_tiles):
        ox0, oy0, oz0, ox1, oy1, oz1 = tile
        px0 = max(0, ox0 - pad_x); px1 = min(int(vx), ox1 + pad_x)
        py0 = max(0, oy0 - pad_y); py1 = min(int(vy), oy1 + pad_y)
        pz0 = max(0, oz0 - pad_z); pz1 = min(int(vz), oz1 + pad_z)

        chunks = chunk_files_for_bbox(args.dst, px0, px1, py0, py1, pz0, pz1)
        if all_chunks_exist(chunks):
            skipped += 1
        else:
            sx0 = src_ox_vox + px0; sx1 = src_ox_vox + px1
            sy0 = src_oy_vox + py0; sy1 = src_oy_vox + py1
            sz0 = src_oz_vox + pz0; sz1 = src_oz_vox + pz1
            try:
                arr = np.asarray(src[sx0:sx1, sy0:sy1, sz0:sz1])
            except Exception as e:
                print(f"  [SKIP tile {idx}] fetch err: {e}", flush=True)
                continue
            dst[px0:px1, py0:py1, pz0:pz1] = arr
            total_bytes += arr.nbytes
            fetched += 1

        if (idx+1) % 50 == 0 or idx == len(my_tiles)-1:
            dt = time.time() - t0
            mbps = total_bytes/1e6/max(dt, 0.01)
            print(f"  {idx+1}/{len(my_tiles)} tiles  fetched={fetched} skipped={skipped}  "
                  f"{total_bytes/1e6:.0f} MB  {mbps:.0f} MB/s  elapsed {dt:.0f}s",
                  flush=True)

    dt = time.time() - t0
    print(f"\ndone: fetched={fetched} skipped={skipped} of {len(my_tiles)}, "
          f"{total_bytes/1e6:.1f} MB, {dt:.0f}s")


if __name__ == "__main__":
    main()
