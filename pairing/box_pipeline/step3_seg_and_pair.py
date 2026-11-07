"""Step 3 (CPU): neuron seg (waterz) -> synapse instances -> pre/post pairing -> bouton catalog.

Inputs  (<RUN>/<name>/): aff.tif, syn_prob_infer.tif, mito_prob_infer.tif (all with blank sections removed), box.json
Alternative to aff.tif: seg_in.tif (a ready segmentation in the original frame, compact ids) + optional seg_gid.npy
(local id -> id in a global segmentation; adds pre_gid / post_gid columns). Used by cellB_connectivity.
Blank sections are put back after segmentation by copying the neighbouring section (box.json 'src'), so every
output below is in the original box frame. Long blank gaps (box.json 'cuts') are not bridged.
Outputs (boxes/<name>/): seg.tif, syn_inst.tif, synapses.csv, synapses_wk.nml,
                         bouton_labels.tif, boutons.csv, summary.json
All local coords are 8x8x30 voxels inside the box; *_wk columns are full-volume wk (4x4x30).
"""
import argparse
import csv
import importlib.util
import json
import os
from collections import Counter

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

import config as C

ap = argparse.ArgumentParser()
ap.add_argument('--box', required=True)
args = ap.parse_args()
P = lambda f: C.box_path(args.box, f)
box = json.load(open(P('box.json')))
WK_O = box['wk_origin']
SRC = np.array(box['src'])                # original z -> index in the blank-removed stack
CUTS = box.get('cuts', [])
CUT_Z = np.array(sorted({c['z_below'] for c in CUTS} | {c['z_above'] for c in CUTS}), dtype=int)
RES = np.array([30., 8., 8.])            # Z,Y,X nm/vox
STRUCT = ndi.generate_binary_structure(3, 3)


def to_wk(z, y, x):
    return WK_O[0] + 2 * int(x), WK_O[1] + 2 * int(y), WK_O[2] + int(z)


def az_size(vox):
    """Active-zone size of one synapse instance (voxels z,y,x): fit a plane, measure the footprint in that plane.
    Returns (area_um2 from a 30 nm raster with holes closed, convex-hull area_um2, thickness_nm = 5-95 % along the normal)."""
    p = vox * RES
    p = p - p.mean(0)
    if len(p) < 4:
        return 0.0, 0.0, 0.0
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    uv = p @ vt[:2].T
    ij = np.floor((uv - uv.min(0)) / 30.0).astype(int)
    img = np.zeros(ij.max(0) + 3, bool)
    img[ij[:, 0] + 1, ij[:, 1] + 1] = True
    area = ndi.binary_closing(img, structure=np.ones((3, 3), bool)).sum() * 900 / 1e6
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(uv).volume / 1e6
    except Exception:
        hull = 0.0
    t = p @ vt[2]
    return round(float(area), 4), round(float(hull), 4), round(float(np.percentile(t, 95) - np.percentile(t, 5)), 1)


# ---------- neuron segmentation ----------
GID = None                               # optional: local seg id -> id in a global segmentation (seg_gid.npy)
if os.path.exists(P('seg_in.tif')):      # segmentation supplied, already in the original frame: no waterz
    seg = tifffile.imread(P('seg_in.tif')).astype(np.uint32)
    if os.path.exists(P('seg_gid.npy')):
        GID = np.load(P('seg_gid.npy'))
else:
    import waterz
    waterz.__compile(scoring_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>',
                     discretize_queue=256)
    spec = importlib.util.spec_from_file_location('waterz_block', C.WATERZ_BLOCK_PY)
    wb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wb)

    aff = tifffile.imread(P('aff.tif'))
    for c in CUTS:   # aff[0, z] links z-1 and z: zero it so nothing merges across a long gap
        aff[0, c['compact_z']] = 0
    seg = wb.run_waterz_block(aff, seg_thresholds=[C.WATERZ_THRESHOLD], aff_thresholds=[1e-5, 0.99999],
                              sv_type='2d', sv_2d='maxima_distance',
                              merge_function='aff50_his256', return_fragments=False).astype(np.uint32)
    del aff
    seg = seg[SRC]
    tifffile.imwrite(P('seg.tif'), seg, compression='zlib')
Z, Y, X = seg.shape
seg_sizes = np.bincount(seg.ravel())
allowed = seg_sizes >= C.MIN_PARTNER_SEG_VOX
allowed[0] = False
print(f'seg: {int((seg_sizes > 0).sum())} ids, {int(allowed.sum())} >= {C.MIN_PARTNER_SEG_VOX} vox', flush=True)

# ---------- synapse instances (binary watershed) ----------
prob8 = tifffile.imread(P('syn_prob_infer.tif'))
if GID is None:
    tifffile.imwrite(P('syn_prob.tif'), prob8[SRC], compression='zlib')
if os.path.exists(P('mito_prob_infer.tif')):
    tifffile.imwrite(P('mito_prob.tif'), tifffile.imread(P('mito_prob_infer.tif'))[SRC], compression='zlib')
prob = prob8.astype(np.float32) / 255.0
del prob8
fg = prob > C.SYN_FG_THR
seeds, _ = ndi.label(remove_small_objects(prob > C.SYN_SEED_THR, C.SYN_SEED_MIN))
inst = watershed(-ndi.distance_transform_edt(fg), markers=seeds, mask=fg)
inst = remove_small_objects(inst, C.SYN_MIN_SEG)
if C.SYN_KEEP_REGIONS:
    # keep the watershed regions (one per seed) as instances; regions that touch are fused below only if they turn out to
    # have the same pre and post. (Relabelling by connectivity, as before, fused neighbouring synapses onto one target,
    # e.g. two boutons on one thin dendrite -> one instance with one partner; found 2026-10-22, wk 78112, 78722, 4371.)
    from skimage.segmentation import relabel_sequential
    inst, _, _ = relabel_sequential(inst); n_syn = int(inst.max())
else:
    inst, n_syn = ndi.label(inst > 0, structure=STRUCT)
inst = inst.astype(np.uint32)[SRC]
del prob, fg, seeds
print(f'synapse instances: {n_syn}', flush=True)

# ---------- optional skeleton: which segments does the traced axon run through ----------
SKEL = np.load(P('skel_local.npy')) if os.path.exists(P('skel_local.npy')) else None
skel_segs, skel_tree = set(), None
if SKEL is not None and len(SKEL):
    from scipy.spatial import cKDTree
    SKEL = np.clip(SKEL, 0, np.array(seg.shape) - 1)
    skel_tree = cKDTree(SKEL * RES)
    ids, cnt = np.unique(seg[SKEL[:, 0], SKEL[:, 1], SKEL[:, 2]], return_counts=True)
    with open(P('skel_segs.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['seg_id', 'n_skeleton_nodes', 'seg_vox', 'seg_gid'])
        for i, c in zip(ids, cnt):
            w.writerow([int(i), int(c), int(seg_sizes[i]), int(GID[i]) if GID is not None else int(i)])
    skel_segs = {int(i) for i in ids if i != 0}
    print(f'skeleton: {len(SKEL)} nodes in box, on {len(skel_segs)} segments ({int(cnt[ids == 0].sum()) if (ids == 0).any() else 0} nodes on background)', flush=True)

# ---------- pre/post attribution ----------
def pair_instances(inst):
    rows = []
    for sid, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        off = np.array([s.start for s in sl])
        vox = np.argwhere(inst[sl] == sid) + off
        cen = vox.mean(axis=0)
        cz, cy, cx = np.round(cen).astype(int)
        ids = seg[vox[:, 0], vox[:, 1], vox[:, 2]]
        ids = ids[allowed[ids]]
        post = 0
        if len(ids):
            best = None   # dominant-slice rule, then majority
            for zv in np.unique(vox[:, 0]):
                v = vox[vox[:, 0] == zv]
                iz = seg[v[:, 0], v[:, 1], v[:, 2]]
                iz = iz[allowed[iz]]
                if len(iz) < 5:
                    continue
                u, c = np.unique(iz, return_counts=True)
                if best is None or c.max() / len(iz) > best[0]:
                    best = (c.max() / len(iz), int(u[c.argmax()]))
            if best and best[0] >= 0.8:
                post = best[1]
            else:
                u, c = np.unique(ids, return_counts=True)
                post = int(u[c.argmax()])

        z_span = int(np.ptp(vox[:, 0])) + 1
        cross = z_span == 1
        hits = {}
        if len(vox) >= 3:
            _, _, vt = np.linalg.svd(vox * RES - (vox * RES).mean(0), full_matrices=False)
            normal = vt[2]
            for sign in (1, -1):
                for t in range(30, 800, 30):
                    pz, py, px = np.round(cen + sign * normal * t / RES).astype(int)
                    if not (0 <= pz < Z and 0 <= py < Y and 0 <= px < X):
                        break
                    s = int(seg[pz, py, px])
                    if s and allowed[s] and s != post:
                        hits[sign] = (s, t, (pz, py, px))
                        break
        pre, ray, land = 0, 0, None
        if hits:
            pre, ray, land = min(hits.values(), key=lambda h: h[1])
        if (not cross) and (pre == 0 or ray > C.FALLBACK_RAY_NM) and len(ids):
            u, c = np.unique(ids, return_counts=True)
            for i in np.argsort(c)[::-1]:
                if int(u[i]) != post and c[i] / len(ids) >= 0.05:
                    pre, ray, land = int(u[i]), 0, (cz, cy, cx)
                    break
        truncated = ((vox.min(0) == 0).any() or (vox.max(0) == np.array([Z - 1, Y - 1, X - 1])).any()
                     or np.isin(vox[:, 0], CUT_Z).any())
        flags = [f for f, ok in [('long_ray', ray > C.LONG_RAY_NM), ('single_slice', cross),
                                  ('truncated_at_box', truncated), ('no_pre', pre == 0)] if ok]
        wx, wy, wz = to_wk(cz, cy, cx)
        az_area, az_hull, az_thick = az_size(vox)
        rows.append(dict(syn_id=sid, n_vox=len(vox), z_span=z_span, az_area_um2=az_area, az_hull_um2=az_hull,
                         az_thick_nm=az_thick, cz=cz, cy=cy, cx=cx,
                         wk_x=wx, wk_y=wy, wk_z=wz, post_id=post, pre_id=pre, pre_ray_nm=ray,
                         pre_lz=land[0] if land else 0, pre_ly=land[1] if land else 0,
                         pre_lx=land[2] if land else 0, quality=';'.join(flags) or 'ok',
                         pre_on_skeleton=int(pre in skel_segs), post_on_skeleton=int(post in skel_segs),
                         skel_dist_nm=round(float(skel_tree.query(cen * RES)[0]), 0) if skel_tree is not None else -1))
        if GID is not None:
            rows[-1].update(pre_gid=int(GID[pre]), post_gid=int(GID[post]), pre_seg_vox=int(seg_sizes[pre]) if pre else 0,
                            post_seg_vox=int(seg_sizes[post]) if post else 0)
    return rows


def touching_pairs(lab):
    """pairs of different instance ids that touch (26-connectivity)"""
    pairs = set(); n = lab.shape
    for d in [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (0, 1, -1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0), (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]:
        s0 = tuple(slice(max(0, -k), n[i] - max(0, k)) for i, k in enumerate(d)); s1 = tuple(slice(max(0, k), n[i] - max(0, -k)) for i, k in enumerate(d))
        a, b = lab[s0], lab[s1]; m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            pairs |= set(map(tuple, np.unique(np.sort(np.stack([a[m], b[m]], 1), 1), axis=0).tolist()))
    return pairs


rows = pair_instances(inst)
if C.SYN_KEEP_REGIONS and rows:
    pp = {r['syn_id']: (r['pre_id'], r['post_id']) for r in rows}
    parent = list(range(n_syn + 1))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    nf = 0
    for a, b in touching_pairs(inst):
        pa, pb = pp.get(a), pp.get(b)
        # same two segments in either order = the same contact (a small piece can come out with pre and post swapped)
        if pa and pb and set(pa) == set(pb) and pa[0] and pa[1] and find(a) != find(b): parent[max(find(a), find(b))] = min(find(a), find(b)); nf += 1
    if nf:
        lut = np.array([find(i) for i in range(n_syn + 1)], np.uint32)
        from skimage.segmentation import relabel_sequential
        inst, _, _ = relabel_sequential(lut[inst]); inst = inst.astype(np.uint32); n_syn = int(inst.max())
        rows = pair_instances(inst)
    print(f'synapse instances after fusing touching regions with the same two partners: {n_syn} ({nf} fusions)', flush=True)
tifffile.imwrite(P('syn_inst.tif'), inst, compression='zlib')

# out-ratio hint: pre-heavy segs look RGC-like, post-heavy look TC-like (weak inside one box)
n_pre = Counter(r['pre_id'] for r in rows if r['pre_id'])
n_post = Counter(r['post_id'] for r in rows if r['post_id'])

with open(P('synapses.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['syn_id'])
    w.writeheader()
    w.writerows(rows)

with open(P('synapses_wk.nml'), 'w') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<things>\n<parameters>'
            f'<experiment name="box_{args.box}"/><scale x="4" y="4" z="30"/></parameters>\n')
    nid = 1
    for r in rows:
        if not r['pre_id']:
            continue
        px, py, pz = to_wk(r['pre_lz'], r['pre_ly'], r['pre_lx'])
        f.write(f'<thing id="{r["syn_id"]}" color.r="1" color.g="0.4" color.b="0" color.a="1" '
                f'name="syn{r["syn_id"]}_pre{r["pre_id"]}_post{r["post_id"]}_{r["quality"]}"><nodes>\n'
                f'<node id="{nid}" x="{r["wk_x"]}" y="{r["wk_y"]}" z="{r["wk_z"]}" radius="60"/>\n'
                f'<node id="{nid + 1}" x="{px}" y="{py}" z="{pz}" radius="60"/>\n'
                f'</nodes><edges><edge source="{nid}" target="{nid + 1}"/></edges></thing>\n')
        nid += 2
    f.write('</things>\n')

# ---------- QC crops for synapses made by the skeleton's axon (EM, synapse instance, segments) ----------
if skel_segs and os.path.exists(P('em.tif')):
    em = tifffile.imread(P('em.tif'))
    os.makedirs(P('qc'), exist_ok=True)
    for r in rows:
        if not r['pre_on_skeleton']:
            continue
        z0, y0, x0 = max(0, r['cz'] - 12), max(0, r['cy'] - 48), max(0, r['cx'] - 48)
        sl = (slice(z0, z0 + 24), slice(y0, y0 + 96), slice(x0, x0 + 96))
        np.savez_compressed(P(f"qc/syn{r['syn_id']}.npz"), em=em[sl], syn=(inst[sl] == r['syn_id']), seg=seg[sl],
                            origin_zyx=np.array([z0, y0, x0]), pre_id=r['pre_id'], post_id=r['post_id'])
    del em

# ---------- bouton catalog ----------
by_pre = {}
for r in rows:
    if r['pre_id']:
        by_pre.setdefault(r['pre_id'], []).append(r)
seg_slices = ndi.find_objects(seg)
blabels = np.zeros_like(seg)
brows, gid = [], 0
for pre_id, syns in by_pre.items():
    sl = seg_slices[pre_id - 1]
    off = np.array([s.start for s in sl])
    sub = seg[sl] == pre_id
    edt = ndi.distance_transform_edt(sub, sampling=tuple(RES))
    core, k = ndi.label(edt > C.BOUTON_NECK_NM, structure=STRUCT)
    if k:
        core = remove_small_objects(core, 100)
        k = int((np.unique(core) > 0).sum())
    bl = watershed(-edt, markers=core, mask=sub) if k else sub.astype(np.int32)
    per_b = {}
    for r in syns:
        p = np.array([r['pre_lz'], r['pre_ly'], r['pre_lx']]) - off
        if (p >= 0).all() and (p < sub.shape).all() and bl[tuple(p)]:
            b = int(bl[tuple(p)])
        else:
            pts = np.argwhere(bl > 0)
            if not len(pts):
                continue
            b = int(bl[tuple(pts[np.linalg.norm((pts - p) * RES, axis=1).argmin()])])
        per_b.setdefault(b, []).append(r['syn_id'])
    for b, sids in per_b.items():
        m = bl == b
        gid += 1
        blabels[sl][m] = gid
        vz, vy, vx = (np.argwhere(m).mean(0) + off).astype(int)
        wx, wy, wz = to_wk(vz, vy, vx)
        touches = (np.argwhere(m) + off)
        truncated = ((touches.min(0) == 0).any() or (touches.max(0) == np.array([Z - 1, Y - 1, X - 1])).any()
                     or np.isin(touches[:, 0], CUT_Z).any())
        brows.append(dict(bouton_id=gid, pre_seg_id=pre_id, vol_vox=int(m.sum()),
                          vol_um3=round(m.sum() * 8 * 8 * 30 / 1e9, 4), syn_count=len(sids),
                          syn_ids=';'.join(map(str, sorted(sids))), cz=vz, cy=vy, cx=vx,
                          wk_x=wx, wk_y=wy, wk_z=wz, truncated_at_box=int(truncated),
                          pre_seg_n_pre=n_pre[pre_id], pre_seg_n_post=n_post.get(pre_id, 0),
                          pre_seg_gid=int(GID[pre_id]) if GID is not None else pre_id))
tifffile.imwrite(P('bouton_labels.tif'), blabels, compression='zlib')
with open(P('boutons.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(brows[0].keys()) if brows else ['bouton_id'])
    w.writeheader()
    w.writerows(brows)

complete = [b for b in brows if not b['truncated_at_box']]
summary = dict(box=args.box, n_blank_sections=len(box.get('blank_z', [])), n_long_gaps_cut=len(CUTS), depth_um=int(float(box.get('depth_um') or -1)), n_synapses=len(rows),
               n_synapses_pre_on_skeleton=sum(r['pre_on_skeleton'] for r in rows),
               n_synapses_ok=sum(r['quality'] == 'ok' for r in rows),
               n_boutons=len(brows), n_boutons_complete=len(complete),
               box_volume_um3=round(np.prod(box['wk_size']) * 4 * 4 * 30 / 1e9, 1),
               median_bouton_um3_complete=float(np.median([b['vol_um3'] for b in complete])) if complete else None,
               multisyn_frac_complete=(sum(b['syn_count'] >= 2 for b in complete) / len(complete)) if complete else None)
json.dump(summary, open(P('summary.json'), 'w'), indent=2)
print(json.dumps(summary, indent=2))
