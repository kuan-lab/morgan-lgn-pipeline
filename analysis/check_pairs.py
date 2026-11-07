"""Pair-level ground truth check for cell B: does the pipeline find the same presynaptic bouton -> cell B pair as the annotation?

Annotation convention (Nghia, 2026-10-08): a 'post rgc' / 'post ffi' comment sits on a branch tip of the cell B skeleton;
extending the last edge a little past the tip lands in the presynaptic bouton. A 'post ffi' tip shares its parent node
with a 'post rgc' tip: those two partners form the triad.
Per tip: stitch a 3 x 3 x 1.4 um box (stitch_local), march along the last edge from the tip until leaving cell B's
segment(s) -> the annotated bouton segment. Captured = the pipeline has a synapse onto cell B whose presynaptic point
lies in that segment (one bouton = one pair, whatever the number of active zones).
Writes pairing/analysis/pair_check.csv, pair_check_missed_wk.nml, and cellB_rgc_boutons.csv (pipeline output per bouton).
"""
import os, re, sys
os.environ['CLOUDFILES_MONITORING'] = 'false'
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = '/gpfs/radev/project/kuan/ntl5/cell_A_full'; D = f'{ROOT}/cellB_connectivity'; RUN = os.environ.get('PAIR_RUN', 'pairing'); A = f'{D}/{RUN}/analysis'
O = np.array([25114, 33058, 134]); RES = np.array([8., 8., 30.]); HALF = np.array([192, 192, 24]); MAXEXT = 450      # collect every segment the extension passes through up to here (2026-10-22)
FILES = [('Nghia', f'{D}/CellB_skeleton.nml'), ('JJ', f'{ROOT}/cellB_skel_JJ.nml')]
loc = lambda p: np.array([p[0] / 2, p[1] / 2, p[2]]) - O                      # wk -> local 8 nm voxel (float)

S = pd.read_csv(f'{D}/{RUN}/cellB_all_synapses.csv', low_memory=False, usecols=['uid', 'wk_x', 'wk_y', 'wk_z', 'pre_wk_x', 'pre_wk_y', 'pre_wk_z', 'quality', 'pre_id', 'pre_gid', 'az_area_um2'])
S = S[S.pre_id > 0].reset_index(drop=True)
SC = np.stack([S.wk_x / 2, S.wk_y / 2, S.wk_z], 1) - O; SP = np.stack([S.pre_wk_x / 2, S.pre_wk_y / 2, S.pre_wk_z], 1) - O
_I = pd.read_csv(f'{A}/cellB_inputs.csv', usecols=['uid', 'pre_gid']); ONB = set(_I.uid)
# with the graph segmentation, box labels ARE the global ids, so compare the presynaptic segment by id; this also uses the
# analysis' corrected pre side (cell B swapped to post when the pairing had it reversed)
GRAPH = os.environ.get('SEG_SOURCE') == 'graph'
PRE_GID = S.pre_gid.to_numpy().astype(np.int64).copy(); _m = pd.Series(np.arange(len(S)), index=S.uid)
_k = _I.uid.isin(_m.index); PRE_GID[_m[_I.uid[_k]].to_numpy()] = _I.pre_gid[_k].to_numpy().astype(np.int64); RB = pd.read_csv(f'{A}/cellB_rgc_synapses.csv').set_index('uid')
SKEL = np.load(f'{D}/{RUN}/skeleton_nodes.npy'); SKL = np.stack([SKEL[:, 0] / 2, SKEL[:, 1] / 2, SKEL[:, 2]], 1) - O


def sites():
    out = []
    for who, f in FILES:
        t = open(f).read(); pos = {}; adj = defaultdict(list); C = {}
        xyz = lambda s: tuple(float(re.search(r' %s="([-\d.]+)"' % a, s).group(1)) for a in 'xyz')
        for m in re.finditer(r'<node [^>]*>', t): pos[int(re.search(r' id="(\d+)"', m.group(0)).group(1))] = xyz(m.group(0))
        for a, b in re.findall(r'<edge source="(\d+)" target="(\d+)"', t): adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
        for s in re.findall(r'<comment [^>]*>', t):
            n = int(re.search(r'node="(\d+)"', s).group(1))
            if n in pos: C[n] = re.search(r'content="([^"]*)"', s).group(1)
        for n, c in C.items():
            if not c.startswith('post') or len(adj[n]) != 1: continue
            p = adj[n][0]; kind = ('post rgc' if 'rgc' in c else 'post ffi' if 'ffi' in c else 'post') + (' ?' if '?' in c else '')
            sib = [m for m in adj[p] if m != n and 'post rgc' in C.get(m, '')] if 'ffi' in c else []
            out.append(dict(who=who, kind=kind, text=c, tip=pos[n], parent=pos[p], sib_tip=pos[sib[0]] if sib else None, sib_parent=pos[p] if sib else None))
    return out


def march(seg, lo, tip, par, Bl, cnt, all_segments=False):
    """segments that are not cell B along the last edge, from the tip outwards. Returns (first segment of >= 2000 voxels
    (or the first one met), its extension nm, its contact voxel); with all_segments=True also the set of every segment
    of >= 300 voxels met up to MAXEXT. (Earlier versions took only the first one: that overshot or stopped in slivers.)"""
    a, b = loc(tip) * RES, loc(par) * RES; u = a - b; n = np.linalg.norm(u)
    if n == 0: return (0, -1, None, set()) if all_segments else (0, -1, None)
    u /= n; met = []; seen = set()
    for t in range(0, MAXEXT + 1, 15):
        v = np.round((a + u * t) / RES).astype(int) - lo
        if (v < 0).any() or (v >= seg.shape).any(): break
        l = int(seg[tuple(v)])
        if l and l not in Bl and cnt.get(l, 0) >= 300 and l not in seen: seen.add(l); met.append((l, t, v + lo))
    first = next((m for m in met if cnt.get(m[0], 0) >= 2000), met[0] if met else (0, -1, None))
    return (first[0], first[1], first[2], seen) if all_segments else first


def one(s):
    import stitch_local
    T = loc(s['tip']); lo = np.round(T).astype(int) - HALF; size = 2 * HALF
    try:
        seg = stitch_local.stitch(int(lo[0]), int(lo[1]), int(lo[2]), int(size[0]), int(size[1]), int(size[2]))
    except Exception as e:
        return dict(who=s['who'], kind=s['kind'], status=f'error {str(e)[:50]}')
    u, c = np.unique(seg, return_counts=True); cnt = dict(zip(u.tolist(), c.tolist()))
    inb = lambda Q: np.all((Q >= lo) & (Q < lo + size), 1)
    at = lambda Q: seg[tuple((np.round(Q).astype(int) - lo).T)]
    k = SKL[inb(np.round(SKL))]; ul, uc = np.unique(at(k), return_counts=True)
    Bl = set(ul[(uc >= 2) & (ul > 0)].tolist())
    tp = np.round(np.stack([T, loc(s['parent'])]))
    if inb(tp).all(): Bl |= {int(x) for x in at(tp) if x}
    L, ext, contact, MET = march(seg, lo, s['tip'], s['parent'], Bl, cnt, all_segments=True)
    m = inb(np.round(SC)) & inb(np.round(SP)); ix = np.nonzero(m)[0]
    pre = PRE_GID[ix] if GRAPH else at(SP[ix]); onb = np.array([u_ in ONB for u_ in S.uid.to_numpy()[ix]], dtype=bool); uid = S.uid.to_numpy()[ix]
    wk = lambda v: f'{int(2 * (v[0] + O[0]))}, {int(2 * (v[1] + O[1]))}, {int(v[2] + O[2])}'
    r = dict(who=s['who'], kind=s['kind'], text=s['text'], tip_wk=f"{int(s['tip'][0])}, {int(s['tip'][1])}, {int(s['tip'][2])}", status='ok', cellB_segments_in_box=len(Bl),
             bouton_found=bool(L), extension_nm=ext, bouton_wk=wk(contact) if L else '', bouton_vox_in_box=cnt.get(L, 0) if L else 0)
    hit = ix[np.isin(pre, list(MET)) & onb] if MET else np.array([], int)      # captured: a synapse onto cell B from any segment the extension passes
    r['segments_past_tip'] = ';'.join(str(m) for m in MET)
    r['pair_captured'] = bool(len(hit)); r['n_active_zones'] = int(len(hit)); r['az_sum_um2'] = round(float(S.az_area_um2.to_numpy()[hit].sum()), 4) if len(hit) else 0.0
    hu = S.uid.to_numpy()[hit]; rg = [RB.rgc_num[x] for x in hu if x in RB.index]
    r['called_rgc'] = bool(rg); r['rgc_num'] = float(pd.Series(rg).mode()[0]) if rg else np.nan; r['pipeline_triad'] = bool(any(RB.triad[x] for x in hu if x in RB.index))
    if L and not len(hit):
        other = ix[np.isin(pre, list(MET)) & ~onb]
        d = np.linalg.norm((SC[ix] - contact) * RES, axis=1) if len(ix) else np.array([])
        nearB = ix[onb & (d <= 400)] if len(ix) else []
        r['why_missed'] = ('synapse from this bouton detected, but its target was not called cell B' if len(other) else
                           'a synapse onto cell B is detected within 400 nm, but assigned to a different presynaptic segment' if len(nearB) else 'no synapse detected at this contact')
    elif not L:
        r['why_missed'] = 'no separate segment found past the tip (bouton merged with cell B here, or tip points elsewhere)'
    else: r['why_missed'] = ''
    if s['sib_tip'] is not None and L:                 # triad: sibling RGC bouton in the same box; R -> I synapse
        Lr, _, _ = march(seg, lo, s['sib_tip'], s['sib_parent'], Bl, cnt)
        r['sibling_rgc_bouton_found'] = bool(Lr)
        if Lr and len(ix):
            v = SC[ix] - SP[ix]; nv = np.linalg.norm(v * RES, axis=1); post = np.zeros(len(ix), np.int64)
            for t in (0, 60, 120, 180, 240, 300):
                q = np.round(SC[ix] + (v * RES / np.maximum(nv, 1)[:, None]) * t / RES); ok = inb(q) & (post == 0)
                if ok.any():
                    l = at(q[ok]); l = np.where(l == pre[ok], 0, l); post[np.nonzero(ok)[0]] = l
            r['rgc_to_lin_synapse_found'] = bool(((pre == Lr) & (post == L)).any())
            r['rgc_pair_captured'] = bool(((pre == Lr) & onb).any())
            r['pipeline_triad_on_sibling'] = bool(any(RB.triad[x] for x in uid[(pre == Lr) & onb] if x in RB.index))
    return r


if __name__ == '__main__':
    ss = sites(); print('tips with an edge to follow:', pd.Series([(s['who'], s['kind']) for s in ss]).value_counts().sort_index().to_dict(), flush=True)
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 4))) as p: res = p.map(one, ss, chunksize=2)
    R = pd.DataFrame(res)
    # --- capture rule (2026-10-22): the pipeline synapse onto cell B nearest the tip, within NEAR_NM. The walk along the last
    # skeleton edge (columns above) often passes beside the bouton, so it no longer decides captured / missed.
    from scipy.spatial import cKDTree
    from cloudvolume import CloudVolume
    import re as _re
    NEAR_NM = 300; NM = np.array([4., 4., 30.])
    In = pd.read_csv(f'{A}/cellB_inputs.csv', low_memory=False); Rg = pd.read_csv(f'{A}/cellB_rgc_synapses.csv').set_index('uid')
    allS = pd.read_csv(f'{D}/{RUN}/cellB_all_synapses.csv', low_memory=False, usecols=['uid', 'wk_x', 'wk_y', 'wk_z'])
    tI = cKDTree(In[['wk_x', 'wk_y', 'wk_z']].to_numpy() * NM); tA = cKDTree(allS[['wk_x', 'wk_y', 'wk_z']].to_numpy() * NM)
    man = CloudVolume(f'file://{ROOT}/precomputed_full_sharded', mip=0, fill_missing=True, bounded=False, progress=False)
    NAME = pd.read_csv(f'{ROOT}/segment_groups.csv').set_index('seg_id')['name'].to_dict(); PART = _re.compile(r'^\*?\d+[A-Za-z]?\s+part\s+(\d+)')
    def vast_rgc(w):      # RGC number from a "<target> part <rgc>" piece within ~130 nm of the tip (VAST is 64 nm in x,y)
        x, y, z = int(w[0]) // 16, int(w[1]) // 16, int(w[2]) // 2
        ids = np.unique(np.asarray(man[x - 2:x + 3, y - 2:y + 3, z - 1:z + 2]))
        for i in ids:
            m = PART.match(str(NAME.get(int(i), '')))
            if m: return int(m.group(1))
        return np.nan
    cols = dict(nearest_input_nm=[], nearest_input_wk=[], nearest_pre_gid=[], nearest_swapped=[], nearest_az_um2=[], nearest_rgc_call=[], vast_rgc_near_tip=[], captured=[], rgc_match=[], why_missed2=[])
    for r in R.itertuples():
        w = np.array([float(v) for v in r.tip_wk.split(',')]); d, i = tI.query(w * NM); s = In.iloc[i]
        ok = d <= NEAR_NM; vr = vast_rgc(w) if r.kind.startswith('post rgc') else np.nan
        rc = Rg.rgc_num.get(s.uid, np.nan) if ok else np.nan
        cols['nearest_input_nm'].append(round(float(d))); cols['nearest_input_wk'].append(f'{int(s.wk_x)}, {int(s.wk_y)}, {int(s.wk_z)}')
        cols['nearest_pre_gid'].append(int(s.pre_gid) if ok else 0); cols['nearest_swapped'].append(bool(s.swapped_cellB_to_post) if ok else False)
        cols['nearest_az_um2'].append(round(float(s.az_area_um2), 4) if ok else 0.0); cols['nearest_rgc_call'].append(rc); cols['vast_rgc_near_tip'].append(vr)
        cols['captured'].append(bool(ok)); cols['rgc_match'].append(bool(ok and vr == vr and rc == rc and int(vr) == int(rc)) if r.kind.startswith('post rgc') else np.nan)
        cols['why_missed2'].append('' if ok else ('a synapse is detected within 300 nm, but not as an input onto cell B' if tA.query(w * NM)[0] <= NEAR_NM else 'no synapse detected within 300 nm'))
    for k, v in cols.items(): R[k] = v
    R.to_csv(f'{A}/pair_check.csv', index=False)
    rk = R[R.status == 'ok']
    print(f'\n=== capture = pipeline input onto cell B within {NEAR_NM} nm of the tip (random skeleton nodes: about 17 % by chance) ===')
    g2 = rk.groupby(['who', 'kind']).agg(tips=('kind', 'size'), captured=('captured', 'sum'), median_dist_nm=('nearest_input_nm', 'median'))
    g2['captured_%'] = (100 * g2.captured / g2.tips).round(0); print(g2.to_string())
    rr = rk[rk.kind.str.startswith('post rgc') & rk.captured]
    print(f'post rgc tips captured: {len(rr)}; VAST names an RGC near the tip: {int(rr.vast_rgc_near_tip.notna().sum())}; same RGC number as the pipeline call: {int(rr.rgc_match.sum())}')
    print('why missed:'); print(rk[~rk.captured].groupby(['who', 'kind', 'why_missed2']).size().to_string())
    print('errors:', int((R.status != 'ok').sum())); R = R[R.status == 'ok']
    g = R.groupby(['who', 'kind']).agg(tips=('kind', 'size'), bouton_found=('bouton_found', 'sum'), pair_captured=('pair_captured', 'sum'), called_rgc=('called_rgc', 'sum'),
                                       median_extension_nm=('extension_nm', lambda x: int(x[x >= 0].median()) if (x >= 0).any() else -1), median_AZ_per_pair=('n_active_zones', lambda x: float(x[x > 0].median()) if (x > 0).any() else 0))
    g['captured_%'] = (100 * g.pair_captured / g.tips).round(0); print(g.to_string())
    print('\nwhy pairs were missed:'); print(R[~R.pair_captured].groupby(['who', 'kind', 'why_missed']).size().to_string())
    if 'sibling_rgc_bouton_found' in R:
        t = R[R.sibling_rgc_bouton_found == True]
        print('\ntriads (post ffi tip with a sibling post rgc tip, both boutons found):'); print(t.groupby('who').agg(triads=('kind', 'size'), lin_to_cellB=('pair_captured', 'sum'), rgc_to_cellB=('rgc_pair_captured', 'sum'),
              rgc_to_lin=('rgc_to_lin_synapse_found', 'sum'), all_three=('kind', lambda x: int((t.loc[x.index, 'pair_captured'] & t.loc[x.index, 'rgc_pair_captured'] & t.loc[x.index, 'rgc_to_lin_synapse_found']).sum())),
              pipeline_called_triad=('pipeline_triad_on_sibling', 'sum')).to_string())
    miss = R[~R.captured]
    with open(f'{A}/pair_check_missed_wk.nml', 'w') as fh:
        fh.write('<things>\n<parameters>\n<experiment name="Morgan" />\n<scale x="4" y="4" z="30" />\n</parameters>\n')
        for k, m in enumerate(miss.itertuples(), 1):
            x, y, z = [int(v) for v in m.tip_wk.split(',')]
            fh.write(f'<thing id="{k}" color.r="1" color.g="0.85" color.b="0" color.a="1" name="MISSED PAIR {m.who} {m.text}: {m.why_missed2}"><nodes><node id="{k}" x="{x}" y="{y}" z="{z}" radius="120"/></nodes><edges/></thing>\n')
        fh.write('</things>\n')
    b = pd.read_csv(f'{A}/cellB_rgc_synapses.csv')
    bt = b.groupby('pre_eid').agg(rgc_num=('rgc_num', lambda x: x.mode()[0]), n_active_zones=('uid', 'size'), az_sum_um2=('az_area_um2', 'sum'), wk_x=('wk_x', 'mean'), wk_y=('wk_y', 'mean'), wk_z=('wk_z', 'mean'),
                                  triad=('triad', 'any'), glom=('glom', lambda x: x.mode()[0]), in_glom=('in_glom', 'any')).reset_index().round({'az_sum_um2': 4, 'wk_x': 0, 'wk_y': 0, 'wk_z': 0})
    bt.to_csv(f'{A}/cellB_rgc_boutons.csv', index=False)
    print(f'\npipeline output per bouton: {len(bt)} RGC bouton -> cell B pairs (from {len(b)} active zones); active zones per pair: median {bt.n_active_zones.median():.0f}, max {bt.n_active_zones.max()}; pairs in a triad: {int(bt.triad.sum())}')
