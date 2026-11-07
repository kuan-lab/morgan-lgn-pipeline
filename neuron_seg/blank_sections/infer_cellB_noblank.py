#!/usr/bin/env python3
"""Cell B tile-plan inference of neuron affinity (v4) + mito (v3) with blank EM sections removed.

Per tile: read EM with extra z context, drop empty sections (black, or structureless: see empty_frames.py), run both models on the stacked
non-blank sections, then put every original z back:
  - a blank slot copies the outputs of its source section (first half of a run from below, second half from above)
  - aff_z of a slot = 255 if it copies the same source as the slot below (identical section),
    otherwise the model's z-affinity between the two consecutive non-blank sections
  - runs of >= LONG_GAP blank sections are not bridged: aff_z = 0 at the switch from below- to above-copies
Writes two sparse precomputed volumes (same grid as the EM cache), max-blended like blob_infer_from_plan_3ch.py.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from cloudvolume import CloudVolume

sys.path.insert(0, '/gpfs/radev/project/kuan/ntl5/cell_A_full')
from blob_infer_from_plan_3ch import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from empty_frames import empty_sections
BLANK_ZERO_FRAC, BLANK_MEAN, LONG_GAP, EXTRA_Z = 0.5, 40, 3, 10


def out_volume(path, em_cv, nch):
    info = CloudVolume.create_new_info(num_channels=nch, layer_type='image', data_type='uint8', encoding='raw',
                                       resolution=list(em_cv.resolution), voxel_offset=list(em_cv.voxel_offset),
                                       chunk_size=[128, 128, 32], volume_size=list(em_cv.volume_size))
    os.makedirs(path, exist_ok=True)
    cv = CloudVolume(f'file://{path}', info=info, mip=0, compress=False, fill_missing=True, non_aligned_writes=True)
    if not os.path.exists(os.path.join(path, 'info')):
        cv.commit_info()
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tile-plan', required=True)
    ap.add_argument('--em-precomputed', required=True)
    ap.add_argument('--out-neuron', required=True)
    ap.add_argument('--out-mito', default=None)
    ap.add_argument('--neuron-cfg', required=True); ap.add_argument('--neuron-ckpt', required=True)
    ap.add_argument('--mito-cfg', default=None); ap.add_argument('--mito-ckpt', default=None)
    ap.add_argument('--tiles-from', default=None, help='json with tile_indices (e.g. blank_blocks.json)')
    ap.add_argument('--tiles', default=None, help='comma-separated tile indices (testing)')
    ap.add_argument('--input-size', default='32,256,256'); ap.add_argument('--pad-size', default='8,64,64')
    ap.add_argument('--task-idx', type=int, default=0); ap.add_argument('--n-tasks', type=int, default=1)
    ap.add_argument('--max-tiles', type=int, default=0, help='debug: stop after this many tiles')
    ap.add_argument('--long-gap', type=int, default=LONG_GAP, help='blank runs of >= this many sections are cut, not bridged (large value = always bridge)')
    ap.add_argument('--force', action='store_true', help='redo tiles even if flagged done; the tile region is overwritten, not max-blended')
    args = ap.parse_args()

    tiles = json.load(open(args.tile_plan))['tiles']
    tz, ty, tx = map(int, args.input_size.split(',')); pz, py, px = map(int, args.pad_size.split(','))
    in_z = tz + 2 * pz
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m_neu = load_model(args.neuron_cfg, args.neuron_ckpt, dev)
    m_mit = load_model(args.mito_cfg, args.mito_ckpt, dev) if args.out_mito else None
    em_cv = CloudVolume(f'file://{args.em_precomputed}', mip=0, progress=False, fill_missing=True)
    vx, vy, vz = map(int, em_cv.volume_size); off = tuple(map(int, em_cv.voxel_offset))
    cv_n = out_volume(args.out_neuron, em_cv, 3); cv_m = out_volume(args.out_mito, em_cv, 1) if args.out_mito else None
    done_dir = os.path.join(args.out_neuron, '_done'); os.makedirs(done_dir, exist_ok=True)
    pool = list(range(len(tiles)))
    if args.tiles_from: pool = json.load(open(args.tiles_from))['tile_indices']
    if args.tiles: pool = [int(t) for t in args.tiles.split(',')]
    mine = pool[args.task_idx::args.n_tasks]
    if args.max_tiles: mine = mine[:args.max_tiles]
    print(f'task {args.task_idx}/{args.n_tasks}: {len(mine)} tiles', flush=True)
    t0, n_blank_tiles = time.time(), 0
    for k, ti in enumerate(mine):
        flag = os.path.join(done_dir, f'{ti}')
        if os.path.exists(flag) and not args.force:
            continue
        ox0, oy0, oz0, ox1, oy1, oz1 = tiles[ti]
        vldz = oz1 - oz0
        ix0, iy0 = ox0 - px, oy0 - py
        wz0, wz1 = oz0 - pz - EXTRA_Z, oz0 + tz + pz + EXTRA_Z          # z window with extra context
        cz0, cz1 = max(0, wz0), min(vz, wz1)
        cx0, cx1 = max(0, ix0), min(vx, ix0 + tx + 2 * px); cy0, cy1 = max(0, iy0), min(vy, iy0 + ty + 2 * py)
        slab = np.asarray(em_cv[cx0 + off[0]:cx1 + off[0], cy0 + off[1]:cy1 + off[1], cz0 + off[2]:cz1 + off[2]])[..., 0]
        pb = [cx0 - ix0, cy0 - iy0]; pa = [ix0 + tx + 2 * px - cx1, iy0 + ty + 2 * py - cy1]
        if any(pb) or any(pa):
            slab = np.pad(slab, [(pb[0], pa[0]), (pb[1], pa[1]), (0, 0)], mode='reflect')
        core = slab[px:px + tx, py:py + ty]                                   # blank test on the tile's own xy
        zs = np.arange(cz0, cz1)                                              # original z of each slab section
        blank = empty_sections(core, zs)[0]                                   # black, or structureless (white / noise frames)
        nb = zs[~blank]
        if len(nb) == 0:
            open(flag, 'w').close(); continue
        n_blank_tiles += int(blank[(zs >= oz0) & (zs < oz0 + vldz)].any())
        # input = pz non-blank before the core, the core's non-blank, then fill up to in_z with non-blank after
        before = nb[nb < oz0][-pz:]
        inside = nb[(nb >= oz0) & (nb < oz0 + vldz)]
        after = nb[nb >= oz0 + vldz][:max(0, in_z - len(before) - len(inside))]
        seq = np.concatenate([before, inside, after])
        sel = [int(np.where(zs == z)[0][0]) for z in seq]
        x = slab[:, :, sel]
        if x.shape[2] < in_z:                                                 # volume edge: reflect-pad like the original
            lo = max(0, pz - len(before)); hi = in_z - x.shape[2] - lo
            x = np.pad(x, [(0, 0), (0, 0), (lo, hi)], mode='reflect')
        else:
            lo = 0
        xt = torch.from_numpy(((x.transpose(2, 1, 0).astype(np.float32) / 255.0) - 0.5) / 0.5).to(dev)[None, None]
        with torch.no_grad():
            neu = (torch.sigmoid(m_neu(xt))[0].cpu().numpy() * 255).astype(np.uint8)    # (3, in_z, Y, X)
            mit = (torch.sigmoid(m_mit(xt))[0].cpu().numpy() * 255).astype(np.uint8) if m_mit else None
        pos = {int(z): lo + i for i, z in enumerate(seq)}                     # original non-blank z -> output index
        # source section for every core z (nearest non-blank; halves for runs), and cut flags
        src, cut = [], []
        for z in range(oz0, oz0 + vldz):
            if z in pos:
                src.append(pos[z]); cut.append(False); continue
            below = nb[nb < z]; above = nb[nb > z]
            a = z;  b = z
            while a - 1 >= cz0 and blank[a - 1 - cz0]: a -= 1
            while b + 1 < cz1 and blank[b + 1 - cz0]: b += 1
            n = b - a + 1; h = (n + 1) // 2
            use_below = len(below) and (z < a + h or not len(above))
            s = int(below[-1]) if use_below else int(above[0])
            if s not in pos:   # source fell outside the model input (very long run at the window edge)
                s = min(pos, key=lambda q: abs(q - z))
            src.append(pos[s]); cut.append(n >= args.long_gap and z == a + h and len(below) > 0)
        src = np.array(src)
        out_n = neu[:, src][:, :, py:py + (oy1 - oy0), px:px + (ox1 - ox0)].copy()
        out_m = mit[:, src][:, :, py:py + (oy1 - oy0), px:px + (ox1 - ox0)] if m_mit else None
        for i in range(vldz):
            prev_src = src[i - 1] if i > 0 else None
            if i == 0 and (oz0 - 1) in pos:
                prev_src = pos[oz0 - 1]
            if prev_src is not None and src[i] == prev_src:
                out_n[0, i] = 255                                          # identical copy of the section below
            if cut[i]:
                out_n[0, i] = 0                                            # long gap: do not bridge
        wx0, wx1, wy0, wy1 = ox0 + off[0], ox1 + off[0], oy0 + off[1], oy1 + off[1]
        wz0_, wz1_ = oz0 + off[2], oz1 + off[2]
        for cv, arr in ((cv_n, out_n), (cv_m, out_m)):
            if cv is None: continue
            new = arr.transpose(3, 2, 1, 0)                                 # (X, Y, Z, C)
            prev = np.asarray(cv[wx0:wx1, wy0:wy1, wz0_:wz1_])
            cv[wx0:wx1, wy0:wy1, wz0_:wz1_] = new if args.force else np.maximum(prev, new)
        open(flag, 'w').close()
        if (k + 1) % 50 == 0:
            dt = time.time() - t0
            print(f'  {k + 1}/{len(mine)}  {dt:.0f}s ({dt / (k + 1):.2f} s/tile)  tiles with blank sections so far: {n_blank_tiles}', flush=True)
    print(f'done in {time.time() - t0:.0f}s; tiles touching blank sections: {n_blank_tiles}', flush=True)


if __name__ == '__main__':
    main()
