"""Pass 3 (global merge) of Tim's graph pipeline, rewritten to scale to billions of edges.

Same result as his stages/global_merge.py: pairs scored by two cores keep the WORSE score (dedupe 'max'); keep
score <= threshold and contact >= min_contact; connected components; object id = smallest fragment id in the
component; the table lists every fragment that ends up in a multi-fragment object.
Why rewritten: his dedupe runs np.unique(axis=0) over all edges at once (9.3 billion for cell B: did not finish in
16 h). Here only cross-core edges (the only ones scored twice) are deduped, and fragment ids are mapped to compact
indices arithmetically (id = core * CORE_VOXELS + local, local = 1..n_fragments of that core).

  python merge_graph_fast.py <edges dir> <fragments ckpt dir> <out prefix> [threshold] [min_contact] [workers]
Writes <out prefix>_nodes.npy, _roots.npy (sorted by node; uncompressed, so relabel can memory-map slices).
"""
import glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
CAP = 256 * 1024 * 1024
EDGES, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
THR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2; MINC = int(sys.argv[5]) if len(sys.argv) > 5 else 0; NW = int(sys.argv[6]) if len(sys.argv) > 6 else 8
meta = [json.load(open(f)) for f in glob.glob(f'{CKPT}/block_*.json')]
cores = np.array(sorted(m['index'] for m in meta), np.int64); nf = {m['index']: int(m['n_fragments']) for m in meta}
for m in meta: assert int(m['bump']) == m['index'] * CAP, m['index']
off = np.concatenate([[0], np.cumsum([nf[c] for c in cores])]).astype(np.int64); N = int(off[-1])
pos = np.full(int(cores.max()) + 1, -1, np.int64); pos[cores] = np.arange(len(cores))
IDX = np.int32 if N < 2 ** 31 - 1 else np.int64


def dense(ids):
    ids = ids.astype(np.int64); c = (ids - 1) // CAP; p = pos[c]
    assert (p >= 0).all(), 'edge to a fragment of a core without metadata'
    return (off[p] + (ids - c * CAP) - 1).astype(IDX)


def load(f):
    d = np.load(f); u, v, s, c = d['u'].astype(np.int64), d['v'].astype(np.int64), d['score'], d['contact']
    cross = (u - 1) // CAP != (v - 1) // CAP
    k = ~cross & (s <= THR) & (c >= MINC)
    return dense(u[k]), dense(v[k]), u[cross], v[cross], s[cross], c[cross], len(u)


if __name__ == '__main__':
    t0 = time.time(); files = sorted(glob.glob(f'{EDGES}/edges_*.npz'))
    print(f'{len(files)} edge files; {len(cores)} cores with fragments, {N:,} fragments; threshold {THR}, min contact {MINC}', flush=True)
    R, C, XU, XV, XS, XC = [], [], [], [], [], []; tot = 0
    with Pool(NW) as p:
        for k, (a, b, xu, xv, xs, xc, n) in enumerate(p.imap(load, files, chunksize=4)):
            R.append(a); C.append(b); XU.append(xu); XV.append(xv); XS.append(xs); XC.append(xc); tot += n
            if k % 500 == 0: print(f'  read {k}/{len(files)} files, {tot:,} edges so far, {time.time() - t0:.0f}s', flush=True)
    xu, xv, xs, xc = (np.concatenate(x) for x in (XU, XV, XS, XC)); del XU, XV, XS, XC
    a, b = np.minimum(xu, xv), np.maximum(xu, xv); o = np.lexsort((xs, b, a)); a, b, xs, xc = a[o], b[o], xs[o], xc[o]
    last = np.ones(len(a), bool); last[:-1] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])          # last row of each pair = its worst score
    a, b, xs, xc = a[last], b[last], xs[last], xc[last]; k = (xs <= THR) & (xc >= MINC)
    ncross = len(a); R.append(dense(a[k])); C.append(dense(b[k]))
    rows, cols = np.concatenate(R), np.concatenate(C); del R, C
    print(f'{tot:,} edges read; within-core kept {len(rows) - int(k.sum()):,}; cross-core unique pairs {ncross:,} (kept {int(k.sum()):,}); {time.time() - t0:.0f}s', flush=True)
    # connected components in batches (scipy's csgraph uses 32-bit indices): contract with the labels found so far
    BATCH = int(os.environ.get('MERGE_BATCH', 800_000_000)); lab = np.arange(N, dtype=IDX); nb = 0
    for i in range(0, len(rows), BATCH):
        lr, lc = lab[rows[i:i + BATCH]], lab[cols[i:i + BATCH]]; m = lr != lc
        if m.any():
            _, nl = connected_components(coo_matrix((np.ones(int(m.sum()), np.int8), (lr[m], lc[m])), shape=(N, N)), directed=False)
            lab = nl.astype(IDX)[lab]
        nb += 1
    del rows, cols
    print(f'components found in {nb} batch(es); {time.time() - t0:.0f}s', flush=True)
    size = np.bincount(lab); _, first = np.unique(lab, return_index=True)
    sel = np.nonzero(size[lab] >= 2)[0]; rootd = first[lab[sel]]
    tid = lambda d: cores[np.searchsorted(off, d, 'right') - 1] * CAP + (d - off[np.searchsorted(off, d, 'right') - 1]) + 1
    nodes, roots = tid(sel).astype(np.uint64), tid(rootd).astype(np.uint64)
    np.save(f'{OUT}_nodes.npy', nodes); np.save(f'{OUT}_roots.npy', roots)
    print(f'{len(nodes):,} fragments merged into {int((size >= 2).sum()):,} multi-fragment objects; largest {int(size.max()):,} fragments; table -> {OUT}_*.npy; {time.time() - t0:.0f}s', flush=True)
