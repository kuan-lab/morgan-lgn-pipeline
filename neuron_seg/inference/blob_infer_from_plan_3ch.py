#!/usr/bin/env python3
"""3-channel variant of blob_infer_from_plan.py for neuron affinity (z/y/x aff).
Same tile-plan-sparse approach as synapse — only writes chunks where the plan has tiles.

Difference from 1ch version:
  - Output precomputed initialized with num_channels=3
  - Model output has 3 channels: y[0, 0..2] instead of y[0, 0]
  - Blend + write per-channel
"""
import argparse, json, os, time
import numpy as np
import torch
from cloudvolume import CloudVolume


def load_cfg(cfg_path):
    from connectomics.config import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    return cfg


def load_model(cfg_path, checkpoint, device):
    from connectomics.model import build_model, update_state_dict
    cfg = load_cfg(cfg_path)
    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    state = update_state_dict(cfg, state, mode='test')
    inner = model.module if hasattr(model, "module") else model
    stripped = {(k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state.items()}
    inner.load_state_dict(stripped, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-plan", required=True)
    ap.add_argument("--em-precomputed", required=True)
    ap.add_argument("--out-precomputed", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input-size", default="32,256,256")
    ap.add_argument("--pad-size", default="8,64,64")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--num-channels", type=int, default=3, help="model OUT_PLANES")
    ap.add_argument("--save-channels", type=int, default=None, help="write only the first N channels (e.g. 3 nearest-neighbour of a 9-ch mito affinity model)")
    ap.add_argument("--gzip", action="store_true", help="gzip-compress output chunks")
    ap.add_argument("--contiguous", action="store_true", help="give each task a contiguous run of z-slabs (tasks never write the same chunk)")
    ap.add_argument("--resume", action="store_true", help="skip tiles already done (progress file in the output folder)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    n_save = args.save_channels or args.num_channels

    with open(args.tile_plan) as f:
        plan = json.load(f)
    tiles = plan['tiles']
    print(f"plan has {len(tiles)} tiles")

    tile_z, tile_y, tile_x = [int(v) for v in args.input_size.split(",")]
    pad_z, pad_y, pad_x = [int(v) for v in args.pad_size.split(",")]
    in_z, in_y, in_x = tile_z + 2*pad_z, tile_y + 2*pad_y, tile_x + 2*pad_x
    print(f"tile OUT {(tile_z,tile_y,tile_x)}  PAD {(pad_z,pad_y,pad_x)}  IN {(in_z,in_y,in_x)}  channels={args.num_channels}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = load_model(args.cfg, args.checkpoint, device)

    em_cv = CloudVolume(f"file://{args.em_precomputed}", mip=0, progress=False,
                        fill_missing=True)
    vx, vy, vz = tuple(int(v) for v in em_cv.volume_size)
    em_off = tuple(int(v) for v in em_cv.voxel_offset)
    print(f"EM volume: {vx}x{vy}x{vz}   voxel_offset {em_off}")

    # Output precomputed — MULTI-CHANNEL
    info = CloudVolume.create_new_info(
        num_channels=n_save, layer_type='image', data_type='uint8',
        encoding='raw', resolution=list(em_cv.resolution),
        voxel_offset=list(em_off), chunk_size=[128, 128, 32],
        volume_size=[vx, vy, vz],
    )
    os.makedirs(args.out_precomputed, exist_ok=True)
    out_cv = CloudVolume(f"file://{args.out_precomputed}", info=info, mip=0,
                         compress='gzip' if args.gzip else False, fill_missing=True,
                         non_aligned_writes=True)
    out_cv.commit_info()

    if args.contiguous:
        tiles = sorted(tiles, key=lambda t: (t[2], t[1], t[0]))
        z0s = np.array([t[2] for t in tiles]); cuts = np.searchsorted(z0s, np.unique(z0s))      # slab starts
        bounds = [0] + [int(cuts[np.searchsorted(cuts, len(tiles) * k / args.n_tasks)]) if np.searchsorted(cuts, len(tiles) * k / args.n_tasks) < len(cuts) else len(tiles)
                        for k in range(1, args.n_tasks)] + [len(tiles)]
        my_tiles = tiles[bounds[args.task_idx]:bounds[args.task_idx + 1]]
    else:
        my_tiles = tiles[args.task_idx::args.n_tasks]
    prog = os.path.join(args.out_precomputed, "progress", f"task{args.task_idx}_of{args.n_tasks}.txt")
    start = 0
    if args.resume and os.path.exists(prog):
        start = int(open(prog).read().split()[0]); print(f"resume: skipping {start} tiles already done")
    os.makedirs(os.path.dirname(prog), exist_ok=True)
    print(f"task {args.task_idx}/{args.n_tasks}: {len(my_tiles)} tiles")

    t0 = time.time()
    for idx, tile in enumerate(my_tiles):
        if idx < start: continue
        ox0, oy0, oz0, ox1, oy1, oz1 = tile
        vld_x = ox1 - ox0; vld_y = oy1 - oy0; vld_z = oz1 - oz0
        ix0 = ox0 - pad_x; ix1 = ox0 + tile_x + pad_x
        iy0 = oy0 - pad_y; iy1 = oy0 + tile_y + pad_y
        iz0 = oz0 - pad_z; iz1 = oz0 + tile_z + pad_z
        cx0, cx1 = max(0, ix0), min(vx, ix1)
        cy0, cy1 = max(0, iy0), min(vy, iy1)
        cz0, cz1 = max(0, iz0), min(vz, iz1)
        slab = np.squeeze(np.asarray(em_cv[
            cx0 + em_off[0]:cx1 + em_off[0],
            cy0 + em_off[1]:cy1 + em_off[1],
            cz0 + em_off[2]:cz1 + em_off[2]]))
        if slab.ndim == 2: slab = slab[:, :, None]
        pb = [max(0, -ix0), max(0, -iy0), max(0, -iz0)]
        pa = [max(0, ix1 - vx), max(0, iy1 - vy), max(0, iz1 - vz)]
        if any(pb) or any(pa):
            slab = np.pad(slab, [(pb[0], pa[0]), (pb[1], pa[1]), (pb[2], pa[2])],
                          mode='reflect')
        assert slab.shape == (in_x, in_y, in_z), \
            f"slab {slab.shape} != {(in_x, in_y, in_z)}"
        x = slab.transpose(2, 1, 0).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x).to(device)[None, None]
        with torch.no_grad():
            y = torch.sigmoid(model(x))   # (1, C, Z_in, Y_in, X_in)
        out_czyx = y[0].cpu().numpy()   # (C, Z, Y, X)
        assert out_czyx.shape[0] == args.num_channels, f"model out ch={out_czyx.shape[0]} != {args.num_channels}"
        cz = pad_z; cy = pad_y; cx = pad_x
        cropped = out_czyx[:n_save, cz:cz+vld_z, cy:cy+vld_y, cx:cx+vld_x]   # (C, Z, Y, X)
        out_slab_xyzc = (cropped * 255).astype(np.uint8).transpose(3, 2, 1, 0)   # (X, Y, Z, C)
        # Write at absolute (wk) coords: 0-based tile coord + em_off
        wx0, wx1 = ox0 + em_off[0], ox1 + em_off[0]
        wy0, wy1 = oy0 + em_off[1], oy1 + em_off[1]
        wz0, wz1 = oz0 + em_off[2], oz1 + em_off[2]
        prev = np.asarray(out_cv[wx0:wx1, wy0:wy1, wz0:wz1])   # (X, Y, Z, C)
        blended = np.maximum(prev, out_slab_xyzc)
        out_cv[wx0:wx1, wy0:wy1, wz0:wz1] = blended
        if (idx+1) % 50 == 0 or idx == len(my_tiles)-1:
            open(prog, "w").write(f"{idx+1} {len(my_tiles)}\n")
            dt = time.time() - t0
            print(f"  {idx+1}/{len(my_tiles)} tiles  elapsed {dt:.0f}s  "
                  f"({dt/(idx+1-start):.2f} s/tile)", flush=True)

    dt = time.time() - t0
    print(f"done: {len(my_tiles)} tiles in {dt:.0f}s")


if __name__ == "__main__":
    main()
