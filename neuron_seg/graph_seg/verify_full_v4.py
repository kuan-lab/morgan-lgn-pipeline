"""Sharded vs unsharded cell B graph segmentation (exact, 5 slabs), every level decodes, and which object each soma is in."""
import os; os.environ['CLOUDFILES_MONITORING'] = 'false'
import numpy as np, json
from cloudvolume import CloudVolume
a = CloudVolume('file:///gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/graph_cellB_v4/seg_wk_v4', mip=0, fill_missing=True, bounded=False, progress=False)
b = CloudVolume('file:///gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/graph_cellB_v4_seg_sharded', mip=0, fill_missing=True, bounded=False, progress=False)
ok = True
for nm, (x, y, z) in {'myelin spot e3': (87415, 83065, 3822), 'cell B soma': (89798, 87673, 3829), 'cell 108 soma': (83843, 85338, 3829), 'cell 224 soma': (93622, 90040, 2340), 'glomerulus 27': (79285, 90694, 4561), 'far dendrite': (68755, 86833, 4413)}.items():
    s = np.s_[x // 2 - 256:x // 2 + 256, y // 2 - 256:y // 2 + 256, z - 4:z + 4]; A = np.asarray(a[s]); B = np.asarray(b[s]); e = np.array_equal(A, B); ok &= e
    print(f'{nm} wk {x}, {y}, {z}: id {int(A[256, 256, 4, 0])}; sharded identical {e}; labelled {100 * (A > 0).mean():.0f}%')
for mip, sc in enumerate(json.load(open('/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/graph_cellB_v4_seg_sharded/info'))['scales']):
    c = CloudVolume('file:///gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/graph_cellB_v4_seg_sharded', mip=mip, fill_missing=True, progress=False); f = 2 ** mip
    v = np.asarray(c[89798 // 2 // f - 64:89798 // 2 // f + 64, 87673 // 2 // f - 64:87673 // 2 // f + 64, 3829]); ok &= bool((v > 0).any())
    print(f'level {sc["key"]}: offset {sc["voxel_offset"]}, decodes at cell B soma: {bool((v > 0).any())}')
print('PASS' if ok else 'FAIL')
