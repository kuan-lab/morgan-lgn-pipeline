"""Pass 4 (relabel) of Tim's graph pipeline for cell B, scaled: each core loads only its slice of the lookup table
(memory-mapped), and the segmentation is written gzip-compressed directly in the frame webKnossos uses
(voxel_offset = cell B origin 25114, 33058, 134 at 8 nm), so no rewrite is needed before sharding.

  python relabel_fast.py <table prefix> <output volume> [workers] [--region x0,x1,y0,y1,z0,z1]
"""
import argparse, json, os, time
os.environ['CLOUDFILES_MONITORING'] = 'false'
from multiprocessing import Pool
import numpy as np
from cloudvolume import CloudVolume
import run_cellB_graph as R
CAP = 256 * 1024 * 1024; O = np.array([25114, 33058, 134]); SIZE = [34971, 19460, 8858]
ap = argparse.ArgumentParser(); ap.add_argument('table'); ap.add_argument('out'); ap.add_argument('workers', nargs='?', type=int, default=8); ap.add_argument('--region', default=None)
A = ap.parse_args()
DONE = os.path.join(R.HERE, 'ckpt_relabel_fast_' + os.path.basename(A.out.rstrip('/'))); os.makedirs(DONE, exist_ok=True)
BLOCKS, SEL = R.cores([int(v) for v in A.region.split(',')] if A.region else None) if A.region else (R.cores()[0], json.load(open(f'{R.HERE}/cores.json'))['cores'])
NODES = ROOTS = None


def one(i):
    global NODES, ROOTS
    if NODES is None: NODES, ROOTS = np.load(f'{A.table}_nodes.npy', mmap_mode='r'), np.load(f'{A.table}_roots.npy', mmap_mode='r')
    z1, z2, y1, y2, x1, x2 = BLOCKS[i]['core']
    F = np.asarray(CloudVolume(R.FRAG, mip=0, bounded=False, fill_missing=True, progress=False)[x1:x2, y1:y2, z1:z2])[..., 0].astype(np.uint64)
    a0, a1 = np.searchsorted(NODES, [i * CAP + 1, (i + 1) * CAP + 1]); nn, rr = np.asarray(NODES[a0:a1]), np.asarray(ROOTS[a0:a1])
    nz = F > 0; v = F[nz]
    if len(nn):
        k = np.clip(np.searchsorted(nn, v), 0, len(nn) - 1); hit = nn[k] == v; v[hit] = rr[k[hit]]; F[nz] = v
    out = CloudVolume(A.out, mip=0, bounded=True, progress=False, compress='gzip', non_aligned_writes=True)
    out[x1 + O[0]:x2 + O[0], y1 + O[1]:y2 + O[1], z1 + O[2]:z2 + O[2]] = F[..., None]
    open(f'{DONE}/{i}', 'w').close(); return i, int(nz.sum())


if __name__ == '__main__':
    if not os.path.exists(A.out.replace('file://', '') + '/info'):
        info = CloudVolume.create_new_info(num_channels=1, layer_type='segmentation', data_type='uint64', encoding='raw', resolution=[8, 8, 30],
                                           voxel_offset=O.tolist(), volume_size=SIZE, chunk_size=[128, 128, 32])
        CloudVolume(A.out, info=info, progress=False).commit_info()
    todo = [i for i in SEL if not os.path.exists(f'{DONE}/{i}')]; t0 = time.time()
    print(f'relabel: {len(todo)} of {len(SEL)} cores to do, {A.workers} workers -> {A.out}', flush=True)
    with Pool(A.workers) as p:
        for k, (i, n) in enumerate(p.imap_unordered(one, todo)):
            if k % 100 == 0: print(f'  {k}/{len(todo)} cores, {time.time() - t0:.0f}s', flush=True)
    print(f'relabel done: {len(todo)} cores in {time.time() - t0:.0f}s', flush=True)
