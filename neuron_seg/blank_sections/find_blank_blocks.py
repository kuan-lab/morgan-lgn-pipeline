"""Which cell B waterz blocks and inference tiles touch a blank EM section.

Blank list: blank_sections_cellB.csv (wk z; blank at >= 1 sampled spot). Local frame (aff / EM cache / tile plan /
block coords) has z offset 134. A block is affected if its z-range (coords include the overlap) contains a blank
section or the section right after one. Tiles to re-infer = plan tiles whose z-range meets an affected block's z-range.
Writes blank_blocks.json: {blank_z_local, block_indices, block_z_ranges, tile_indices}.
"""
import glob, json, os
from multiprocessing import Pool
import numpy as np, pandas as pd

D = '/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity'
META = '/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/waterz_cellB_v4_3um/metadata'
PLAN = '/gpfs/radev/project/kuan/ntl5/cell_A_full/magneton_inference/cellB_tile_plan_3um.json'
Z_OFF = 134

bl = pd.read_csv(f'{D}/blank_sections_cellB.csv')
blank = np.sort(bl[bl.frac_spots_blank > 0].z.to_numpy() - Z_OFF)
hit = np.union1d(blank, blank + 1)
print(f'blank sections (local z): {len(blank)} -> sections whose z-link is affected: {len(hit)}', flush=True)


def load(p):
    b = json.load(open(p))
    return b['index'], b['coords'][0], b['coords'][1]


if __name__ == '__main__':
    files = glob.glob(f'{META}/block_*.json')
    with Pool(8) as pool:
        meta = np.array(pool.map(load, files, chunksize=2000))
    idx, z1, z2 = meta[:, 0], meta[:, 1], meta[:, 2]
    aff = np.zeros(len(idx), bool)
    for z in hit:
        aff |= (z1 <= z) & (z < z2)
    zr = sorted({(int(a), int(b)) for a, b in zip(z1[aff], z2[aff])})
    print(f'blocks: {len(idx)}; affected: {int(aff.sum())} ({100 * aff.mean():.1f}%) in {len(zr)} z-layers', flush=True)
    tiles = json.load(open(PLAN))['tiles']
    tz0 = np.array([t[2] for t in tiles]); tz1 = np.array([t[5] for t in tiles])
    need = np.zeros(len(tiles), bool)
    for a, b in zr:
        need |= (tz0 < b) & (tz1 > a)
    print(f'plan tiles: {len(tiles)}; to re-infer: {int(need.sum())} ({100 * need.mean():.1f}%)', flush=True)
    json.dump(dict(blank_z_local=blank.tolist(), block_indices=sorted(idx[aff].tolist()), block_z_ranges=zr,
                   tile_indices=np.nonzero(need)[0].tolist()), open(f'{D}/blank_blocks.json', 'w'))
    print('wrote blank_blocks.json')
