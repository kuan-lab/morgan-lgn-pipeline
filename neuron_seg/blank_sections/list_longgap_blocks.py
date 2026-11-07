"""Blocks whose z-range meets the long blank gap (local z 2214-2224): clear their done flags and write a manifest."""
import glob, json, os
from multiprocessing import Pool
M = '/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/waterz_cellB_v4_3um'
def load(p):
    b = json.load(open(p)); return b['index'], b['coords'][0], b['coords'][1]
if __name__ == '__main__':
    with Pool(8) as p: meta = p.map(load, glob.glob(f'{M}/metadata_noblank/block_*.json'), chunksize=1000)
    ix = sorted(i for i, z1, z2 in meta if z1 < 2225 and z2 > 2214)
    n = 0
    for i in ix:
        f = f'{M}/ckpt_seg_noblank/block_{i:04d}.done'
        if os.path.exists(f):
            os.rename(f, f + '.cutversion'); n += 1
    os.makedirs(f'{M}/hpc_jobs_seg_longgap/logs', exist_ok=True)
    g = [ix[k:k + 500] for k in range(0, len(ix), 500)]
    open(f'{M}/hpc_jobs_seg_longgap/manifest.txt', 'w').write(''.join(','.join(map(str, x)) + '\n' for x in g))
    json.dump(ix, open('/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/longgap_blocks.json', 'w'))
    print(f'{len(ix)} blocks at the long gap, {n} done flags set aside, {len(g)} tasks')
