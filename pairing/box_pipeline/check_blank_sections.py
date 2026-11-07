"""List blank/dark EM sections in every box of the current RUN and show what the models produced there."""
import os, numpy as np, tifffile
import config as C
for n in C.load_boxes():
    d = os.path.join(C.BOX_DIR, n)
    with tifffile.TiffFile(f'{d}/em.tif') as t:
        st = [(float((p.asarray()[::4, ::4] == 0).mean()), float(p.asarray()[::4, ::4].mean())) for p in t.pages]
    bf, mu = np.array(st).T
    bad = np.where((bf > 0.5) | (mu < 40))[0]
    runs = []
    for z in bad:
        if runs and z == runs[-1][1] + 1: runs[-1][1] = int(z)
        else: runs.append([int(z), int(z)])
    print(f'{n}: blank sections {len(bad)}: {runs}   partly black (2-50%): {int(((bf > 0.02) & (bf <= 0.5)).sum())}', flush=True)
    if not runs: continue
    with tifffile.TiffFile(f'{d}/seg.tif') as ts, tifffile.TiffFile(f'{d}/syn_prob.tif') as tsy, tifffile.TiffFile(f'{d}/mito_prob.tif') as tm:
        aff = tifffile.imread(f'{d}/aff.tif')
        for a0, b0 in runs:
            for z in range(max(a0 - 1, 0), min(b0 + 2, len(ts.pages))):
                sg = ts.pages[z].asarray()
                print(f'   z={z:3d} {"BLANK" if a0 <= z <= b0 else "     "} EM mean {mu[z]:5.1f} | aff z/y/x '
                      f'{aff[0, z].mean() / 255:.2f}/{aff[1, z].mean() / 255:.2f}/{aff[2, z].mean() / 255:.2f} | '
                      f'segs {len(np.unique(sg)):5d}, biggest {np.bincount(sg.ravel()).max() / sg.size:.2f} | '
                      f'syn>0.5 {int((tsy.pages[z].asarray() > 127).sum()):6d}  mito>0.5 {int((tm.pages[z].asarray() > 127).sum()):6d}', flush=True)
        del aff
