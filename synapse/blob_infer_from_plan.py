#!/usr/bin/env python3
"""Phase 3: run jj837 synapse model on each tile in the plan.
Reads EM from local sparse precomputed; writes sigmoid predictions to output
precomputed with non_aligned_writes.

SLURM-array-friendly: --task-idx N --n-tasks K shards the tile list."""
import argparse, json, os, time
import numpy as np
import torch
import tifffile
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
    ap.add_argument("--input-size", default="32,256,256",
                    help="model INPUT_SIZE = OUTPUT_SIZE (z,y,x)")
    ap.add_argument("--pad-size", default="8,64,64")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.tile_plan) as f:
        plan = json.load(f)
    tiles = plan['tiles']
    plan_tile_vox = plan['tile_vox']  # (x, y, z)
    print(f"plan has {len(tiles)} tiles; expected tile_vox {plan_tile_vox}")

    tile_z, tile_y, tile_x = [int(v) for v in args.input_size.split(",")]
    pad_z, pad_y, pad_x = [int(v) for v in args.pad_size.split(",")]
    in_z, in_y, in_x = tile_z + 2*pad_z, tile_y + 2*pad_y, tile_x + 2*pad_x
    print(f"tile OUT {(tile_z,tile_y,tile_x)}  PAD {(pad_z,pad_y,pad_x)}  IN {(in_z,in_y,in_x)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = load_model(args.cfg, args.checkpoint, device)

    em_cv = CloudVolume(f"file://{args.em_precomputed}", mip=0, progress=False,
                        fill_missing=True)
    vx, vy, vz = tuple(int(v) for v in em_cv.volume_size)
    em_off = tuple(int(v) for v in em_cv.voxel_offset)
    print(f"EM volume: {vx}x{vy}x{vz} vox   voxel_offset {em_off}")

    # Output precomputed
    info = CloudVolume.create_new_info(
        num_channels=1, layer_type='image', data_type='uint8', encoding='raw',
        resolution=list(em_cv.resolution),
        voxel_offset=[0, 0, 0],
        chunk_size=[128, 128, 32],
        volume_size=[vx, vy, vz],
    )
    os.makedirs(args.out_precomputed, exist_ok=True)
    out_cv = CloudVolume(f"file://{args.out_precomputed}", info=info, mip=0,
                         compress=False, fill_missing=True,
                         non_aligned_writes=True)
    out_cv.commit_info()

    my_tiles = tiles[args.task_idx::args.n_tasks]
    print(f"task {args.task_idx}/{args.n_tasks}: {len(my_tiles)} tiles")

    t0 = time.time()
    for idx, tile in enumerate(my_tiles):
        ox0, oy0, oz0, ox1, oy1, oz1 = tile
        vld_x = ox1 - ox0; vld_y = oy1 - oy0; vld_z = oz1 - oz0
        # Fetch padded EM tile
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
        # (X, Y, Z) -> (Z, Y, X) -> (1,1,Z,Y,X) tensor
        x = slab.transpose(2, 1, 0).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x).to(device)[None, None]
        with torch.no_grad():
            y = torch.sigmoid(model(x))
        out_zyx = y[0, 0].cpu().numpy()   # (Z_in, Y_in, X_in) in [0,1]
        # Crop central to valid output
        cz = pad_z; cy = pad_y; cx = pad_x
        cropped = out_zyx[cz:cz+vld_z, cy:cy+vld_y, cx:cx+vld_x]
        out_slab_xyz = (cropped * 255).astype(np.uint8).transpose(2, 1, 0)
        # MAX-blend so overlapping tiles keep the higher probability
        prev = np.asarray(out_cv[ox0:ox1, oy0:oy1, oz0:oz1])
        prev = np.squeeze(prev)
        blended = np.maximum(prev, out_slab_xyz)
        out_cv[ox0:ox1, oy0:oy1, oz0:oz1] = blended
        if (idx+1) % 50 == 0 or idx == len(my_tiles)-1:
            dt = time.time() - t0
            print(f"  {idx+1}/{len(my_tiles)} tiles  elapsed {dt:.0f}s  "
                  f"({dt/(idx+1):.2f} s/tile)", flush=True)

    dt = time.time() - t0
    print(f"done: {len(my_tiles)} tiles in {dt:.0f}s")


if __name__ == "__main__":
    main()
