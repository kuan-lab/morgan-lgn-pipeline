"""For every annotated triad on cell B: is the RGC -> interneuron synapse seen, and if not, at which level is it lost?

Triad = a 'post ffi' tip and a 'post rgc' tip sharing a parent node (check_pairs.py finds the two partner segments).
At the place where the RGC bouton and the interneuron piece touch:
  - raw synapse probability of the model (v6ft) in the cleft zone (within 60 nm of both segments)
  - pipeline synapse instances there and how they were paired
For scale, the same numbers at the RGC bouton / cell B contact of the same triad.
Writes pairing/analysis/triad_rgc_lin_check.csv and triad_rgc_lin_wk.nml (one point per triad at the RGC-LIN contact).
"""
import os
os.environ['CLOUDFILES_MONITORING'] = 'false'
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from cloudvolume import CloudVolume
import check_pairs as cp
PRED = '/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/magneton_inference/cellB_pred_v6ft_3um_sharded'
O, RES, HALF = cp.O, cp.RES, cp.HALF
_pv = None


def zone_stats(prob, zone):
    if not zone.any(): return dict(vox=0, max=0.0, n_fg=0, n_half=0, n_seed=0)
    p = prob[zone]
    return dict(vox=int(zone.sum()), max=round(float(p.max()) / 255, 2), n_fg=int((p > 51).sum()), n_half=int((p > 127).sum()), n_seed=int((p > 229).sum()))


def one(s):
    global _pv
    import stitch_local
    T = cp.loc(s['tip']); lo = np.round(T).astype(int) - HALF; size = 2 * HALF
    seg = stitch_local.stitch(int(lo[0]), int(lo[1]), int(lo[2]), int(size[0]), int(size[1]), int(size[2]))
    u, c = np.unique(seg, return_counts=True); cnt = dict(zip(u.tolist(), c.tolist()))
    inb = lambda Q: np.all((Q >= lo) & (Q < lo + size), 1); at = lambda Q: seg[tuple((np.round(Q).astype(int) - lo).T)]
    k = cp.SKL[inb(np.round(cp.SKL))]; ul, uc = np.unique(at(k), return_counts=True); Bl = set(ul[(uc >= 2) & (ul > 0)].tolist())
    tp = np.round(np.stack([T, cp.loc(s['parent'])])); Bl |= {int(x) for x in at(tp[inb(tp)]) if x}
    Ll, _, _ = cp.march(seg, lo, s['tip'], s['parent'], Bl, cnt); Lr, _, _ = cp.march(seg, lo, s['sib_tip'], s['sib_parent'], Bl, cnt)
    r = dict(who=s['who'], ffi_text=s['text'], ffi_tip_wk='%d, %d, %d' % tuple(s['tip']), rgc_tip_wk='%d, %d, %d' % tuple(s['sib_tip']), lin_found=bool(Ll), rgc_found=bool(Lr), same_segment=bool(Ll and Ll == Lr))
    if not Ll or not Lr or Ll == Lr:
        r['diagnosis'] = 'partner segment not found past a tip' if not (Ll and Lr) else 'RGC tip and ffi tip lead into the SAME segment (RGC bouton and interneuron piece merged, or tips point at one object)'
        return r
    if _pv is None: _pv = CloudVolume(f'file://{PRED}', mip=0, fill_missing=True, bounded=False, progress=False)
    a = lo + O; prob = np.asarray(_pv[a[0]:a[0] + size[0], a[1]:a[1] + size[1], a[2]:a[2] + size[2]])[..., 0]
    samp = tuple(RES); dR = ndi.distance_transform_edt(seg != Lr, sampling=samp); dL = ndi.distance_transform_edt(seg != Ll, sampling=samp)
    dB = ndi.distance_transform_edt(~np.isin(seg, list(Bl)), sampling=samp)
    zRL = (dR <= 60) & (dL <= 60); zRB = (dR <= 60) & (dB <= 60); zLB = (dL <= 60) & (dB <= 60)
    r['rgc_lin_gap_nm'] = int(dR[seg == Ll].min()); r['rgc_lin_contact_um2'] = round(float(((seg == Ll) & (dR <= 40)).sum()) * 64e-6, 3)
    for nm, z in (('rgc_lin', zRL), ('rgc_cellB', zRB), ('lin_cellB', zLB)):
        for k_, v in zone_stats(prob, z).items(): r[f'{nm}_{k_}'] = v
    if not zRL.any():
        r['diagnosis'] = f'the two partner segments do not touch in the box (gap {r["rgc_lin_gap_nm"]} nm)'; return r
    cen = np.argwhere(zRL).mean(0) + lo; r['contact_wk'] = f'{int(2 * (cen[0] + O[0]))}, {int(2 * (cen[1] + O[1]))}, {int(cen[2] + O[2])}'
    # pipeline instances near the contact zone
    m = inb(np.round(cp.SC)) & inb(np.round(cp.SP)); ix = np.nonzero(m)[0]
    dz = ndi.distance_transform_edt(~zRL, sampling=samp)
    near = ix[dz[tuple((np.round(cp.SC[ix]).astype(int) - lo).T)] <= 150] if len(ix) else np.array([], int)
    cls = lambda l: 'RGC' if l == Lr else 'LIN' if l == Ll else 'cellB' if l in Bl else 'none' if l == 0 else 'other'
    desc = []
    for j in near:
        pre = int(at(cp.SP[j][None])[0]); v = (cp.SC[j] - cp.SP[j]) * RES; n = np.linalg.norm(v); post = 0
        for t in (0, 60, 120, 180, 240, 300):
            q = np.round(cp.SC[j] + (v / max(n, 1)) * t / RES)[None]
            if inb(q).all():
                l = int(at(q)[0])
                if l and l != pre: post = l; break
        desc.append((cls(pre), cls(post), str(cp.S.quality[j]), round(float(cp.S.az_area_um2[j]), 3)))
    r['instances_at_contact'] = len(near); r['instance_pairs'] = '; '.join(f'{a_}->{b_} ({q_}, {z_} um2)' for a_, b_, q_, z_ in desc)
    pairs = {(a_, b_) for a_, b_, _, _ in desc}
    if ('RGC', 'LIN') in pairs: r['diagnosis'] = 'captured: instance paired RGC -> LIN'
    elif ('LIN', 'RGC') in pairs: r['diagnosis'] = 'detected, direction reversed (paired LIN -> RGC)'
    elif len(near): r['diagnosis'] = 'instance at the contact, but paired to other segments'
    elif r['rgc_lin_n_seed'] >= 50 and r['rgc_lin_n_fg'] >= 300: r['diagnosis'] = 'model sees it (enough voxels for an instance) but no instance at the contact'
    elif r['rgc_lin_max'] >= 0.9: r['diagnosis'] = 'model sees it, but too small for the instance thresholds (needs >= 50 voxels > 0.9 and >= 300 voxels > 0.2)'
    elif r['rgc_lin_max'] >= 0.5: r['diagnosis'] = 'weak model signal (max probability 0.5-0.9)'
    else: r['diagnosis'] = 'no model signal at the contact (max probability < 0.5)'
    return r


if __name__ == '__main__':
    ss = [s for s in cp.sites() if s['sib_tip'] is not None]
    print('annotated triads (ffi tip + sibling rgc tip):', pd.Series([s['who'] for s in ss]).value_counts().to_dict(), flush=True)
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 4))) as p: R = pd.DataFrame(p.map(one, ss, chunksize=1))
    R.to_csv(f'{cp.A}/triad_rgc_lin_check.csv', index=False)
    print(R.groupby(['diagnosis', 'who']).size().unstack(fill_value=0).assign(total=lambda d: d.sum(1)).sort_values('total', ascending=False).to_string())
    t = R[R.get('rgc_lin_vox', pd.Series(dtype=float)).fillna(0) > 0]
    print(f'\ntriads where the two partner segments touch: {len(t)}')
    print('model signal in the cleft zone, median (max probability | voxels > 0.2 | voxels > 0.9):')
    for nm in ('rgc_lin', 'rgc_cellB', 'lin_cellB'):
        d = t[t[f'{nm}_vox'] > 0]; print(f'  {nm:10s} n={len(d):2d}   max {d[f"{nm}_max"].median():.2f} | {d[f"{nm}_n_fg"].median():.0f} | {d[f"{nm}_n_seed"].median():.0f}   ; with max >= 0.9: {int((d[f"{nm}_max"] >= 0.9).sum())}, max < 0.5: {int((d[f"{nm}_max"] < 0.5).sum())}')
    show = ['who', 'contact_wk', 'rgc_lin_contact_um2', 'rgc_lin_max', 'rgc_lin_n_fg', 'rgc_lin_n_seed', 'instances_at_contact', 'instance_pairs', 'diagnosis']
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
    print(t[show].sort_values('rgc_lin_max').to_string(index=False))
    with open(f'{cp.A}/triad_rgc_lin_wk.nml', 'w') as fh:
        fh.write('<things>\n<parameters>\n<experiment name="Morgan" />\n<scale x="4" y="4" z="30" />\n</parameters>\n')
        for k, m in enumerate(R.itertuples(), 1):
            ok = str(m.diagnosis).startswith('captured'); p = m.contact_wk if isinstance(getattr(m, 'contact_wk', None), str) else m.ffi_tip_wk
            x, y, z = [int(v) for v in p.split(',')]; col = (0, 0.8, 0.3) if ok else (1, 0.2, 0.2)
            fh.write(f'<thing id="{k}" color.r="{col[0]}" color.g="{col[1]}" color.b="{col[2]}" color.a="1" name="RGC-LIN ({m.who}): {m.diagnosis}"><nodes><node id="{k}" x="{x}" y="{y}" z="{z}" radius="120"/></nodes><edges/></thing>\n')
        fh.write('</things>\n')
