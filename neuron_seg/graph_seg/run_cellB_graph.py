"""Tim's graph segmentation (yf354 magneton, read-only) for the whole of cell B, on our tile-plan affinity.

  python run_cellB_graph.py cores                          # list the cores that touch cell B's inferred tube -> cores.json
  python run_cellB_graph.py fragments|edges|relabel W [--region x0,x1,y0,y1,z0,z1] [--lut path]
  python run_cellB_graph.py merge [threshold]
Needs PYTHONPATH=<this dir>/pylib (link to Tim's repo; see GRAPH_SEGMENTATION_GUIDE.md).

Three things added around his stages, without editing his code (module-level CloudVolume / watershed are wrapped;
ProcessPoolExecutor forks, so workers inherit the wrappers):
  1. affinity = our v4 tile-plan volume with the blank-section patch pasted over the tiles it re-inferred
     (blank_blocks.json + emptyfix.json; the long gap 2349-2357 stays cut);
  2. supervoxels outside the inferred tiles are set to 0 (his mask option does nothing in 2D mode, and on a sparse
     affinity each empty area otherwise becomes one huge supervoxel that glues cells together);
  3. fragments and output are written gzip-compressed (his code writes uncompressed uint64: ~2 GB per core).
Only cores that contain an inferred tile are run (cores.json); core numbering stays that of the full grid.
"""
import argparse, json, os, sys, time
os.environ['CLOUDFILES_MONITORING'] = 'false'
import numpy as np
from cloudvolume import CloudVolume as _CV
import yaml
ROOT = '/gpfs/radev/project/kuan/ntl5/cell_A_full'; HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.environ.get('GRAPH_CFG', f'{HERE}/config_cellB.yaml')      # v2 (myelin as background): config_cellB_v2.yaml
CFG = yaml.safe_load(open(CFG_PATH))
INPUT = CFG['paths']['input'].rstrip('/'); FRAG = CFG['paths']['fragments'].rstrip('/'); OUT = CFG['paths']['output'].rstrip('/')
PATCH = 'file:///gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/magneton_inference/cellB_neuron_v4_3um_noblank_patch'
TILES = np.array(json.load(open(f'{ROOT}/magneton_inference/cellB_tile_plan_3um.json'))['tiles'])        # x0,y0,z0,x1,y1,z1 (local)
PIDX = sorted(set(json.load(open(f'{ROOT}/cellB_connectivity/blank_blocks.json'))['tile_indices']) | set(json.load(open(f'{ROOT}/cellB_connectivity/emptyfix.json'))['tile_indices'])
              | set(json.load(open(f'{ROOT}/cellB_connectivity/nodatafix.json'))['tile_indices']))   # v3: tiles next to the no-data start / wk 193, 8959
IS_PATCH = np.zeros(len(TILES), bool); IS_PATCH[PIDX] = True
if os.environ.get('NO_PATCH'): IS_PATCH[:] = False   # tests only: the patch volume was deleted 2026-10-28; read the plain affinity everywhere
# The EM volume starts at wk z 175; the 3 um plan reaches down to wk 134, so local z 0-40 hold no data (zeros, inferred as
# noise and stitched into one object, e.g. wk 99104, 79698, 182). Treated as outside the data. Scan: edge_empty_scan.py.
NO_DATA_Z = (0, 41)
_STATE = {'mask': None}


def _hits(x1, x2, y1, y2, z1, z2):
    t = TILES; return np.nonzero((t[:, 0] < x2) & (t[:, 3] > x1) & (t[:, 1] < y2) & (t[:, 4] > y1) & (t[:, 2] < z2) & (t[:, 5] > z1))[0]


class Overlay:
    """The affinity as the pipeline should see it: tile-plan prediction, blank-section patch on top, 0 outside tiles."""
    def __init__(self, path, *a, **kw):
        self._b = _CV(path, *a, **kw); self._p = None

    def __getattr__(self, k): return getattr(self._b, k)

    def __getitem__(self, s):
        sx, sy, sz = s[:3]; x1, x2, y1, y2, z1, z2 = sx.start, sx.stop, sy.start, sy.stop, sz.start, sz.stop
        a = np.array(self._b[x1:x2, y1:y2, z1:z2]); m = np.zeros(a.shape[:3], bool)
        for i in _hits(x1, x2, y1, y2, z1, z2):
            tx0, ty0, tz0, tx1, ty1, tz1 = (int(v) for v in TILES[i])
            ax, bx, ay, by, az, bz = max(tx0, x1), min(tx1, x2), max(ty0, y1), min(ty1, y2), max(tz0, z1), min(tz1, z2)
            m[ax - x1:bx - x1, ay - y1:by - y1, az - z1:bz - z1] = True
            if IS_PATCH[i]:
                if self._p is None: self._p = _CV(PATCH, mip=0, bounded=False, fill_missing=True, progress=False)
                a[ax - x1:bx - x1, ay - y1:by - y1, az - z1:bz - z1] = np.asarray(self._p[ax:bx, ay:by, az:bz])
        nz1, nz2 = max(NO_DATA_Z[0], z1), min(NO_DATA_Z[1], z2)
        if nz2 > nz1: m[:, :, nz1 - z1:nz2 - z1] = False
        a[~m] = 0; _STATE['mask'] = m
        return a


class CV:
    """Stand-in for cloudvolume.CloudVolume inside Tim's stage modules."""
    create_new_info = staticmethod(_CV.create_new_info)

    def __new__(cls, path, *a, **kw):
        p = str(path).rstrip('/')
        if p == INPUT: return Overlay(path, *a, **kw)
        if p in (FRAG, OUT) and kw.get('compress') is False: kw['compress'] = 'gzip'
        return _CV(path, *a, **kw)


import magneton.instance_segmentation.stages.fragments_stage as FS
import magneton.instance_segmentation.stages.edges_stage as ES
import magneton.instance_segmentation.stages.relabel_stage as RS
import magneton.instance_segmentation.stages.global_merge as GM
from magneton.instance_segmentation import main as MAIN
for mod in (FS, ES, RS): mod.CloudVolume = CV
_ws = FS.run_watershed_fragments


def _ws_masked(aff, *a, **kw):
    out = _ws(aff, *a, **kw); m = _STATE['mask']
    if m is not None and m.shape[::-1] == out.shape: out[~np.transpose(m, (2, 1, 0))] = 0
    elif m is not None: raise RuntimeError(f'mask {m.shape} does not match fragments {out.shape}')
    return out


FS.run_watershed_fragments = _ws_masked

# 4. (v2) myelin / low-affinity voxels as background, exactly like our legacy waterz (magneton_local waterz_block.watershed_2d
#    with interior_thr): voxels whose affinity, averaged over the 3 channels, is <= threshold stay 0; no seeds there.
#    The v4 model predicts ~0.35 inside myelin; Tim's 2D supervoxels label every voxel, so without this myelin became objects.
#    Optional myelin_open_radius r (pixels): only low-affinity regions that hold a disk of radius r (a morphological opening of the
#    low mask, per section) become background, so thin membranes / processes are kept and thick myelin bands are removed.
_THR2D = CFG.get('fragments_stage', {}).get('interior_threshold_2d')
_OPEN_R = CFG.get('fragments_stage', {}).get('myelin_open_radius')
if _THR2D is not None:
    import magneton.instance_segmentation.waterz_block as WB
    from skimage.segmentation import watershed as _skw
    from scipy.ndimage import binary_opening as _open

    def _interior(affs):
        low = affs.mean(axis=0) <= float(_THR2D)
        if _OPEN_R:
            r = int(_OPEN_R); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; disk = (yy ** 2 + xx ** 2) <= r * r
            low = np.stack([_open(low[z], structure=disk) for z in range(low.shape[0])])
        return ~low

    def _watershed_2d_masked(affs, seed_method, use_mahotas_watershed=True):
        affs_xy = 1.0 - 0.5 * (affs[1] + affs[2]); interior = _interior(affs)
        frags = np.zeros_like(affs[0]).astype(np.uint64); nid = 1
        for z in range(affs_xy.shape[0]):
            seeds, n = WB.get_seeds_2d(affs_xy[z], next_id=nid, method=seed_method)
            seeds = seeds.astype(np.int32); seeds[~interior[z]] = 0
            frags[z] = _skw(affs_xy[z], markers=seeds, mask=interior[z]); nid += n
        return frags
    WB.watershed_2d = _watershed_2d_masked
cfg = MAIN.load_config(CFG_PATH)


def cores(region=None):
    blocks, _ = FS.build_grid(cfg, CV(INPUT, mip=0, bounded=False, progress=False))
    sel = []
    for b in blocks:
        z1, z2, y1, y2, x1, x2 = b['core']
        if region is not None:
            rx1, rx2, ry1, ry2, rz1, rz2 = region
            if not (x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1 and z1 < rz2 and z2 > rz1): continue
        if len(_hits(x1, x2, y1, y2, z1, z2)): sel.append(int(b['index']))
    return blocks, sel


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('arg', nargs='?')
    ap.add_argument('--region', default=None, help='local x0,x1,y0,y1,z0,z1: only cores touching this box (tests)'); ap.add_argument('--lut', default=None)
    A = ap.parse_args(); t = time.time()
    region = [int(v) for v in A.region.split(',')] if A.region else None
    if A.stage == 'cores':
        blocks, sel = cores(region); json.dump(dict(n_grid=len(blocks), cores=sel), open(f'{HERE}/cores.json', 'w'))
        print(f'grid {len(blocks)} cores; touching the inferred tube: {len(sel)}')
    else:
        sel = json.load(open(f'{HERE}/cores.json'))['cores'] if region is None else cores(region)[1]
        W = int(A.arg) if A.arg and A.stage != 'merge' else 1
        if A.stage == 'fragments': FS.fragments_blocks(cfg, MAIN.get_stage_config(cfg, 'fragments'), indices=sel, workers=W)
        elif A.stage == 'edges': ES.edges_blocks(cfg, MAIN.get_stage_config(cfg, 'edges'), indices=sel, workers=W)
        elif A.stage == 'merge': print(GM.global_merge(cfg, MAIN.get_stage_config(cfg, 'global_merge'), threshold=float(A.arg) if A.arg else None))
        elif A.stage == 'relabel':
            lut = A.lut or os.path.join(CFG['global_merge_stage']['lut_dir'], 'lut_t02_c0.npz')
            RS.relabel_blocks(cfg, MAIN.get_stage_config(cfg, 'relabel'), lut, indices=sel, workers=W)
        print(f'[{A.stage}] {len(sel)} cores, done in {time.time() - t:.0f}s', flush=True)
