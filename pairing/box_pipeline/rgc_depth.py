"""Depth comparison of RGC candidates (current RUN).  Usage: python rgc_depth.py _mito045
Reads <RUN>/<box>/rgc_candidates_v2<TAG>.csv, synapses.csv, em.tif. Writes analysis/<RUN>/rgc_depth<TAG>_{per_box,per_depth}.csv + .png"""
import json, os, sys
import numpy as np, pandas as pd, tifffile
import config as C
T = sys.argv[1] if len(sys.argv) > 1 else ''
rows, cands, bouts = [], [], []
for name, r in C.load_boxes().items():
    d = os.path.join(C.BOX_DIR, name)
    c = pd.read_csv(f'{d}/rgc_candidates_v2{T}.csv'); s = pd.read_csv(f'{d}/synapses.csv')
    with tifffile.TiffFile(f'{d}/em.tif') as t:
        tissue = np.mean([(p.asarray()[::4, ::4] > 0).mean() for p in t.pages])
    vol = np.prod(C.BOX_WK) * 4 * 4 * 30 / 1e9 * tissue
    cc = c[c.truncated_at_box == 0]
    s_ok = s[s.quality == 'ok']
    rgc_syn = s_ok[s_ok.pre_id.isin(c.seg_id)]
    c['box'] = name; c['depth_um'] = int(r['depth_um']); cands.append(c)
    b = pd.read_csv(f'{d}/boutons_mito.csv')
    b = b[b.pre_seg_id.isin(c.seg_id) & (b.truncated_at_box == 0)].copy()   # complete boutons of candidate segments
    b['box'] = name; b['depth_um'] = int(r['depth_um']); bouts.append(b)
    rows.append(dict(box=name, depth_um=int(r['depth_um']), tissue_um3=round(vol, 1),
                     n_rgc=len(c), rgc_per_1000um3=round(1000 * len(c) / vol, 1),
                     rgc_volume_frac=round(c.vol_um3.sum() / vol, 4),
                     n_complete=len(cc), median_um3_complete=round(cc.vol_um3.median(), 3) if len(cc) else None,
                     median_mito_frac=round(c.mito_frac.median(), 3),
                     mean_pre_syn=round(c.n_syn_as_pre.mean(), 2), mean_pre_syn_complete=round(cc.n_syn_as_pre.mean(), 2) if len(cc) else None,
                     frac_rgc_multisyn=round((c.n_syn_as_pre >= 2).mean(), 3),
                     frac_all_ok_syn_from_rgc=round(len(rgc_syn) / max(len(s_ok), 1), 3),
                     cv_um3_complete=round(cc.vol_um3.std() / cc.vol_um3.mean(), 3) if len(cc) > 1 else None,
                     p90_um3_complete=round(cc.vol_um3.quantile(.9), 3) if len(cc) else None,
                     max_um3_complete=round(cc.vol_um3.max(), 3) if len(cc) else None,
                     n_rgc_postsyn_partners=int(rgc_syn.post_id.nunique())))
pb = pd.DataFrame(rows).sort_values(['depth_um', 'box'])
ac = pd.concat(cands)
os.makedirs(C.ANALYSIS_DIR, exist_ok=True)
pb.to_csv(f'{C.ANALYSIS_DIR}/rgc_depth{T}_per_box.csv', index=False)
ac.to_csv(f'{C.ANALYSIS_DIR}/rgc_candidates_all{T}.csv', index=False)
m = ['rgc_per_1000um3', 'rgc_volume_frac', 'median_um3_complete', 'median_mito_frac', 'mean_pre_syn', 'frac_rgc_multisyn', 'frac_all_ok_syn_from_rgc']
pdp = pb.groupby('depth_um')[m].agg(['mean', 'min', 'max']).round(3)
pdp.to_csv(f'{C.ANALYSIS_DIR}/rgc_depth{T}_per_depth.csv')
comp = ac[ac.truncated_at_box == 0]
pooled = comp.groupby('depth_um').agg(n_complete=('seg_id', 'size'), median_um3=('vol_um3', 'median'),
                                     p90_um3=('vol_um3', lambda x: x.quantile(.9)), mean_pre_syn=('n_syn_as_pre', 'mean'),
                                     multisyn=('n_syn_as_pre', lambda x: (x >= 2).mean())).round(3)
pooled.to_csv(f'{C.ANALYSIS_DIR}/rgc_depth{T}_pooled_complete.csv')

# ---- size spread: candidate segments and their boutons (complete only), pooled by depth + permutation test
ab = pd.concat(bouts)
ab.to_csv(f'{C.ANALYSIS_DIR}/rgc_boutons_complete{T}.csv', index=False)
def spread(v):
    v = np.asarray(v, float)
    return dict(n=len(v), mean=v.mean(), sd=v.std(ddof=1), cv=v.std(ddof=1) / v.mean(), median=np.median(v),
                p90=np.quantile(v, .9), max=v.max())
def perm_p(vals, labels, stat, a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.isin(labels, [a, b]); v, l = vals[m], labels[m]
    obs = stat(v[l == b]) - stat(v[l == a])
    null = np.empty(n)
    for i in range(n):
        q = rng.permutation(l); null[i] = stat(v[q == b]) - stat(v[q == a])
    return obs, float((np.abs(null) >= abs(obs)).mean())
STATS = {'cv': lambda v: v.std(ddof=1) / v.mean(), 'p90': lambda v: np.quantile(v, .9), 'mean': np.mean}
srows = []
for level, df in (('candidate segment', comp), ('bouton', ab)):
    for dd, g in df.groupby('depth_um'):
        srows.append(dict(level=level, depth_um=dd, **{k: round(v, 3) for k, v in spread(g.vol_um3).items()}))
    vals, labs = df.vol_um3.to_numpy(float), df.depth_um.to_numpy()
    for a_, b_ in ((50, 100), (50, 200), (100, 200)):
        for sn, f in STATS.items():
            obs, pv = perm_p(vals, labs, f, a_, b_)
            srows.append(dict(level=level, depth_um=f'{b_} vs {a_}', stat=sn, diff=round(obs, 3), perm_p=round(pv, 4)))
sp = pd.DataFrame(srows)
sp.to_csv(f'{C.ANALYSIS_DIR}/rgc_depth{T}_spread.csv', index=False)
print(sp.to_string(index=False))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
INK, MUTED, MARK = '#1f2328', '#6e7781', '#2f6fb3'
titles = {'rgc_per_1000um3': 'RGC candidates per 1000 µm³', 'rgc_volume_frac': 'Tissue volume in RGC candidates',
          'median_um3_complete': 'Median candidate volume (µm³, complete)', 'mean_pre_syn': 'Synapses per candidate (as pre)',
          'frac_rgc_multisyn': 'Candidates with ≥2 synapses', 'frac_all_ok_syn_from_rgc': 'Share of all synapses made by candidates',
          'cv_um3_complete': 'Size spread (CV, complete)', 'max_um3_complete': 'Largest complete candidate (µm³)'}
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
dep = sorted(pb.depth_um.unique())
for ax, k in zip(axes.flat, titles):
    for dd in dep:
        v = pb[pb.depth_um == dd][k].values
        ax.scatter(np.full(len(v), dd), v, s=40, color=MARK, edgecolor='white', linewidth=1.5, zorder=3)
        ax.hlines(np.nanmean(v), dd - 12, dd + 12, color=INK, linewidth=2, zorder=4)
    ax.set_title(titles[k], fontsize=10, color=INK, loc='left'); ax.set_xticks(dep)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.8); ax.set_axisbelow(True); ax.set_ylim(bottom=0)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
for ax in axes[1]: ax.set_xlabel('Depth below myelin (µm)', color=MUTED, fontsize=9)
fig.suptitle(f'RGC candidates by depth ({T or "10%"} mito rule; dots = boxes, bar = mean)', fontsize=10, color=MUTED, x=0.01, ha='left')
fig.tight_layout(); fig.savefig(f'{C.ANALYSIS_DIR}/rgc_depth{T}.png', dpi=150)
print(pb.to_string(index=False)); print(); print(pooled.to_string())
