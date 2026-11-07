"""Step 1: fetch the box's EM from BossDB at 8x8x30.

em.tif       = as fetched (Z,Y,X uint8), blank sections included, for viewing
em_infer.tif = blank sections removed; this is what the models see
box.json     = geometry + blank_z, src (original z -> index in em_infer), cuts (long gaps that are not bridged)
"""
import argparse
import json
import os
os.environ['CLOUDFILES_MONITORING'] = 'false'
import numpy as np
import tifffile
from cloudvolume import CloudVolume
import config as C

ap = argparse.ArgumentParser()
ap.add_argument('--box', required=True)
args = ap.parse_args()

row = C.load_boxes()[args.box]
g = C.box_geometry(row)

cv = CloudVolume(C.EM_URL, mip=0, progress=False, fill_missing=True, use_https=True)
for i, s in enumerate(cv.info['scales']):
    if tuple(s['resolution']) == C.INF_RES:
        cv.mip = i
        break
else:
    raise RuntimeError(f'no {C.INF_RES} mip in {C.EM_URL}')

(x0, y0, z0), (sx, sy, sz) = g['inf_origin'], g['inf_size']
em = np.squeeze(np.asarray(cv[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz])).transpose(2, 1, 0)
out = C.box_path(args.box, 'em.tif')
tifffile.imwrite(out, em.astype(np.uint8), compression='zlib')
Z = em.shape[0]
blank = ((em == 0).mean(axis=(1, 2)) > C.BLANK_ZERO_FRAC) | (em.mean(axis=(1, 2)) < C.BLANK_MEAN)
kept = np.where(~blank)[0]
assert len(kept), 'every section is blank'
pos = {int(z): i for i, z in enumerate(kept)}
src = np.array([pos.get(z, -1) for z in range(Z)])
cuts, z = [], 0
while z < Z:
    if not blank[z]:
        z += 1
        continue
    a = z
    while z < Z and blank[z]:
        z += 1
    b, n = z - 1, z - a
    if a == 0:
        src[a:b + 1] = pos[b + 1]
    elif b == Z - 1:
        src[a:b + 1] = pos[a - 1]
    else:   # first half copies the section below, second half the section above
        h = (n + 1) // 2
        src[a:a + h] = pos[a - 1]
        src[a + h:b + 1] = pos[b + 1]
        if n >= C.LONG_GAP:
            cuts.append(dict(z_below=a + h - 1, z_above=a + h, compact_z=pos[b + 1]))
tifffile.imwrite(C.box_path(args.box, 'em_infer.tif'), em[kept].astype(np.uint8), compression='zlib')
if C.SKELETON_NPY:   # nodes inside the box (+margin) and a coarse mask of what lies within SKEL_MARGIN_NM of them
    from scipy.spatial import cKDTree
    nodes = np.load(C.SKELETON_NPY)                                         # wk x,y,z
    loc = np.stack([nodes[:, 2] - z0, nodes[:, 1] / 2 - y0, nodes[:, 0] / 2 - x0], axis=1)   # box z,y,x (8 nm)
    nm = np.array([30., 8., 8.]); f = C.KEEP_XY_FACTOR
    near = np.all((loc * nm > -C.SKEL_MARGIN_NM - 200) & (loc * nm < np.array(em.shape) * nm + C.SKEL_MARGIN_NM + 200), axis=1)
    inside = np.all((loc >= 0) & (loc < np.array(em.shape)), axis=1)
    np.save(C.box_path(args.box, 'skel_local.npy'), np.round(loc[inside]).astype(np.int32))
    gz, gy, gx = np.meshgrid(np.arange(Z), np.arange(0, em.shape[1], f) + f / 2, np.arange(0, em.shape[2], f) + f / 2, indexing='ij')
    keep = np.zeros(gz.shape, bool)
    if near.any():
        d, _ = cKDTree(loc[near] * nm).query(np.stack([gz, gy, gx], -1).reshape(-1, 3) * nm)
        keep = (d <= C.SKEL_MARGIN_NM + f * 8).reshape(gz.shape)             # slack = one coarse cell
    np.save(C.box_path(args.box, 'keep.npy'), keep[kept])                     # same z frame as em_infer
    print(f'{args.box}: skeleton nodes in box {int(inside.sum())}; volume within {C.SKEL_MARGIN_NM} nm of skeleton: {100 * keep.mean():.0f}%')
with open(C.box_path(args.box, 'box.json'), 'w') as f:
    json.dump({**row, **g, 'blank_z': np.where(blank)[0].tolist(), 'cuts': cuts, 'src': src.tolist()}, f)
print(f'{args.box}: {int(blank.sum())} blank sections removed, {len(cuts)} long gap(s) cut: {cuts}')
print(f'{args.box}: wk origin {g["wk_origin"]} -> em {em.shape} (Z,Y,X) black_frac={(em == 0).mean():.3f}')
