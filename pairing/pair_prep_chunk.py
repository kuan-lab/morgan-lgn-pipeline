"""Prepare one chunk of cell B for box_pipeline/step3_seg_and_pair.py from volumes we already have (no inference).

Writes into the box folder (config.box_path): seg_in.tif (segmentation stitched from this chunk's blocks, ids made
compact) + seg_gid.npy (compact id -> object in the global stitch), syn_prob_infer.tif (synapse probability; blank sections are filled with the minimum of
the two neighbouring real sections, long blank gaps with 0), skel_local.npy (cell B skeleton nodes), box.json.
  BOX_BOXES_CSV=pairing/chunks.csv BOX_WORK_DIR=/tmp/x python pair_prep_chunk.py --box c052_049_024 [--tag noblank]
"""
import argparse, json, os, sys
os.environ['CLOUDFILES_MONITORING'] = 'false'
import numpy as np, tifffile
from cloudvolume import CloudVolume

ROOT = '/gpfs/radev/project/kuan/ntl5/cell_A_full'
MAR = '/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full'
sys.path.insert(0, f'{ROOT}/box_pipeline'); sys.path.insert(0, f'{ROOT}/cellB_connectivity')
import config as C

ap = argparse.ArgumentParser()
ap.add_argument('--box', required=True); ap.add_argument('--tag', default='local', help="'local' = stitch this chunk's blocks; else read waterz_cellB_v4_3um/output_<tag>")
args = ap.parse_args()
row = C.load_boxes()[args.box]; g = C.box_geometry(row)
(x0, y0, z0), (sx, sy, sz) = g['inf_origin'], g['inf_size']
OZ = 134                                                    # cell B frame: local z = absolute z - OZ


def read(path):
    v = CloudVolume(f'file://{path}', mip=0, bounded=False, fill_missing=True, progress=False)
    return np.asarray(v[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz])[..., 0].transpose(2, 1, 0)      # Z,Y,X


if args.tag == 'local':      # stitched from this chunk's own blocks: a wrong join far away cannot leak in (see stitch_local.py)
    import stitch_local
    seg = stitch_local.stitch(x0 - 25114, y0 - 33058, z0 - OZ, sx, sy, sz).transpose(2, 1, 0)
else:
    seg = read(f'{MAR}/waterz_cellB_v4_3um/output_{args.tag}')
gid, inv = np.unique(seg, return_inverse=True)
if gid[0] != 0:
    gid = np.concatenate([[0], gid]); inv = inv + 1
tifffile.imwrite(C.box_path(args.box, 'seg_in.tif'), inv.reshape(seg.shape).astype(np.uint32), compression='zlib')
if args.tag == 'local' and os.environ.get('SEG_SOURCE') != 'graph':   # legacy: seg_gid = object in the legacy global stitch (may be over-merged)
    gid = np.concatenate([[0], stitch_local.global_object(gid[1:])])
np.save(C.box_path(args.box, 'seg_gid.npy'), gid.astype(np.uint64))    # graph ids exceed 32 bits
del seg, inv

prob = read(f'{MAR}/magneton_inference/cellB_pred_v6ft_3um_sharded').copy()
sys.path.insert(0, f'{ROOT}/cellB_connectivity')
from empty_frames import FORCE_EMPTY_LOCAL_Z                 # white / noise frames (not black): wk 2355-2357, 7617
blank = np.array(sorted(set(json.load(open(f'{ROOT}/cellB_connectivity/blank_blocks.json'))['blank_z_local']) | set(FORCE_EMPTY_LOCAL_Z))) + OZ
cuts, k = [], 0
while k < len(blank):
    a = k
    while k + 1 < len(blank) and blank[k + 1] == blank[k] + 1: k += 1
    lo, hi = blank[a] - 1 - z0, blank[k] + 1 - z0            # real sections around this blank run (chunk z index)
    k += 1
    if hi < 0 or lo >= sz: continue
    zs = slice(max(lo + 1, 0), min(hi, sz))
    if hi - lo - 1 >= C.LONG_GAP:
        prob[zs] = 0; cuts.append(dict(z_below=int(lo), z_above=int(hi)))
    elif lo >= 0 and hi < sz:
        prob[zs] = np.minimum(prob[lo], prob[hi])
    else:
        prob[zs] = 0
tifffile.imwrite(C.box_path(args.box, 'syn_prob_infer.tif'), prob, compression='zlib')
blank_local = [int(z - z0) for z in blank if 0 <= z - z0 < sz]

n_skel = 0
if C.SKELETON_NPY:
    s = np.load(C.SKELETON_NPY)                             # wk x,y,z
    loc = np.stack([s[:, 2] - z0, s[:, 1] / 2 - y0, s[:, 0] / 2 - x0], 1)
    inside = np.all((loc >= 0) & (loc < [sz, sy, sx]), axis=1)
    np.save(C.box_path(args.box, 'skel_local.npy'), np.round(loc[inside]).astype(np.int32)); n_skel = int(inside.sum())
with open(C.box_path(args.box, 'box.json'), 'w') as f:
    json.dump({**row, **{k_: list(v) for k_, v in g.items()}, 'blank_z': blank_local, 'cuts': cuts, 'src': list(range(sz)), 'seg_tag': args.tag}, f)
print(f'{args.box}: {len(gid) - 1} segments, synapse-prob voxels > 0.5: {int((prob > 127).sum())}, blank sections {len(blank_local)}, '
      f'long gaps {len(cuts)}, skeleton nodes {n_skel}', flush=True)
