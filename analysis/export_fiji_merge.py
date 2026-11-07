"""Fiji TIFF stacks around a v4 merge (Nghia, 2026-11-06: 'huge merge error' at wk 76458, 81740, 7285): 600 x 600 px at 8 nm x 41 sections.
em; aff_mean / aff_z / aff_xy = affinity as the graph saw it (v4 tile-plan prediction with the blank-section patch pasted, 0 outside tiles), 0-255;
mask_r6 = 1 where the r6 rule made background (myelin); v3 / v4 = segmentation renumbered per crop (float32, ImageJ has no uint32; ids in *_ids.csv);
v4_centre_object = 1 on the v4 object at the centre. Stacks are z, y, x; slice k = wk z (z0 + k); pixel (x, y) = wk (2 * (x0 + x), 2 * (y0 + y)).
  python export_fiji_merge.py <wk x> <wk y> <wk z> [half xy px = 300] [half z = 20]"""
import csv, json, os, sys, numpy as np, tifffile
from cloudvolume import CloudVolume
from scipy.ndimage import binary_opening
R = '/gpfs/radev/project/kuan/ntl5/cell_A_full'; M = '/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full'; OFF = np.array([25114, 33058, 134])
wx, wy, wz = (int(v) for v in sys.argv[1:4]); H = int(sys.argv[4]) if len(sys.argv) > 4 else 300; HZ = int(sys.argv[5]) if len(sys.argv) > 5 else 20
OUT = f'{R}/cellB_connectivity/pairing_graph_v4/analysis/compare_v3/view_merge_{wx}_{wy}_{wz}_fiji'; os.makedirs(OUT, exist_ok=True)
a0 = np.array([wx // 2 - H, wy // 2 - H, wz - HZ]); a1 = np.array([wx // 2 + H, wy // 2 + H, wz + HZ + 1]); l0, l1 = a0 - OFF, a1 - OFF
zyx = lambda a: np.ascontiguousarray(np.transpose(a, (2, 1, 0)))
save = lambda n, a: tifffile.imwrite(f'{OUT}/{n}.tif', a, imagej=True, resolution=(1 / 8, 1 / 8), metadata={'spacing': 30, 'unit': 'nm', 'axes': 'ZYX'})
cv = lambda p, **k: CloudVolume(p, mip=0, fill_missing=True, progress=False, bounded=False, **k)
# affinity exactly as run_cellB_graph.Overlay builds it (patch on the re-inferred tiles, 0 outside the tile plan and in the no-data start)
T = np.array(json.load(open(f'{R}/magneton_inference/cellB_tile_plan_3um.json'))['tiles'])
P = set(json.load(open(f'{R}/cellB_connectivity/blank_blocks.json'))['tile_indices']) | set(json.load(open(f'{R}/cellB_connectivity/emptyfix.json'))['tile_indices']) | set(json.load(open(f'{R}/cellB_connectivity/nodatafix.json'))['tile_indices'])
aff = np.asarray(cv(f'file://{M}/magneton_inference/cellB_neuron_v4_3um_sharded')[l0[0]:l1[0], l0[1]:l1[1], l0[2]:l1[2]]).copy(); inside = np.zeros(aff.shape[:3], bool); npatch = 0
hit = np.nonzero((T[:, 0] < l1[0]) & (T[:, 3] > l0[0]) & (T[:, 1] < l1[1]) & (T[:, 4] > l0[1]) & (T[:, 2] < l1[2]) & (T[:, 5] > l0[2]))[0]
for i in hit:
    t = T[i]; lo = np.maximum(t[:3], l0); hi = np.minimum(t[3:], l1); s = tuple(slice(int(a - b), int(c - b)) for a, c, b in zip(lo, hi, l0)); inside[s] = True
    if i in P: aff[s] = np.asarray(cv(f'file://{M}/magneton_inference/cellB_neuron_v4_3um_noblank_patch')[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); npatch += 1
if l0[2] < 41: inside[:, :, :max(0, 41 - l0[2])] = False
aff[~inside] = 0
low = aff.astype(np.float32).mean(-1) / 255 <= 0.4; yy, xx = np.mgrid[-6:7, -6:7]; disk = yy ** 2 + xx ** 2 <= 36
mask = np.stack([binary_opening(low[:, :, z], structure=disk) for z in range(low.shape[2])], -1)
save('aff_mean', zyx(aff.astype(np.float32).mean(-1).round().astype(np.uint8))); save('aff_z', zyx(aff[..., 0])); save('aff_xy', zyx(((aff[..., 1].astype(np.uint16) + aff[..., 2]) // 2).astype(np.uint8)))
save('mask_r6', zyx(mask.astype(np.uint8))); del aff, low
save('em', zyx(np.asarray(cv(f'file://{R}/magneton_inference/cellB_em_sparse_3um_256_halo64')[l0[0]:l1[0], l0[1]:l1[1], l0[2]:l1[2]])[..., 0]))
for k, p in (('v4', 'graph_cellB_v4_seg_sharded'), ('v3', 'graph_cellB_v3_seg_sharded')):
    f = np.asarray(cv(f'file://{M}/{p}')[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]])[..., 0]; u = np.unique(f); u = u[u > 0]; r = np.zeros(f.shape[::-1], np.float32)
    for z in range(f.shape[2]):
        sl = f[:, :, z].T; m = sl > 0; r[z][m] = np.searchsorted(u, sl[m]) + 1
    save(k, r)
    with open(f'{OUT}/{k}_ids.csv', 'w', newline='') as fh: w = csv.writer(fh); w.writerow(['fiji_value', 'segment_id']); w.writerows((i + 1, int(v)) for i, v in enumerate(u))
    c = int(f[H, H, HZ]); print(f'{k}: object at centre {c}, {len(u)} objects in crop')
    if k == 'v4':
        save('v4_centre_object', zyx((f == c).astype(np.uint8))); C = f == c
    else:
        uu, cc = np.unique(f[C & (f > 0)], return_counts=True); o = np.argsort(-cc)[:6]
        print('v3 objects under the v4 centre object (voxels):', {int(uu[i]): int(cc[i]) for i in o})
print(f'tiles {len(hit)} (patch {npatch}); r6 background {mask.mean():.2f} of crop; slice 0 = wk z {a0[2]}; pixel (x, y) -> wk (2 * ({a0[0]} + x), 2 * ({a0[1]} + y))')
print('written:', OUT, sorted(os.listdir(OUT)))
