"""How clustered are RGC boutons on cell B? (inputs: bouton_level/rgc_boutons.csv from bouton_connectivity.py, CellB_skeleton.nml)
1. RGC boutons vs random points on the cell B skeleton, matched in distance-from-soma (1 um bins), so proximal bias alone is not called clustering.
2. Triadic boutons among all RGC boutons: shuffle the triad label.
3. Same-RGC boutons: shuffle RGC identity.
Writes bouton_level/clustering_{summary.json,pair_ratio.csv,per_rgc.csv}, clustering.png"""
import json, xml.etree.ElementTree as ET, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
O = os.environ.get('BC_OUT', 'pairing_graph_v2/analysis/bouton_level'); NM = np.array([4., 4., 30.]); SOMA = np.array([89752, 87832, 3839]) * NM / 1000
GLOM_UM, NREP, rng = 2.5, 1000, np.random.default_rng(0)
B = pd.read_csv(f'{O}/rgc_boutons.csv'); P = B[['wk_x', 'wk_y', 'wk_z']].to_numpy(float) * NM / 1000; tri = B.triad.to_numpy(bool); n = len(P)
# skeleton -> points every ~0.2 um along edges (uniform in cable length)
segs = []
for t in ET.parse('CellB_skeleton.nml').getroot().findall('.//thing'):
    nd = {x.get('id'): np.array([float(x.get(k)) for k in 'xyz']) * NM / 1000 for x in t.findall('./nodes/node')}
    segs += [(nd[e.get('source')], nd[e.get('target')]) for e in t.findall('./edges/edge') if e.get('source') in nd and e.get('target') in nd]
segs = np.array(segs); L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1); k = np.maximum(1, np.round(L / 0.2)).astype(int)
f = np.concatenate([(np.arange(m) + 0.5) / m for m in k]); s0 = np.repeat(segs[:, 0], k, 0); s1 = np.repeat(segs[:, 1], k, 0)
C = s0 + (s1 - s0) * f[:, None]; C = C[np.linalg.norm(C - cKDTree(P).data[cKDTree(P).query(C)[1]], axis=1) < 1e9]
dS = lambda X: np.linalg.norm(X - SOMA, axis=1)
bins = np.arange(0, dS(np.vstack([P, C])).max() + 1, 1.0)
hb = np.histogram(dS(P), bins)[0]; ib = np.digitize(dS(C), bins) - 1; pools = [np.where(ib == j)[0] for j in range(len(bins) - 1)]
def null_draw():
    return C[np.concatenate([rng.choice(pools[j], hb[j]) for j in range(len(hb)) if hb[j]])]
def nn(X): return cKDTree(X).query(X, 2)[0][:, 1]
def nglom(X):
    if len(X) < 3: return 0
    lab = fcluster(linkage(X, 'single'), GLOM_UM, 'distance'); c = np.bincount(lab); return int((c >= 3).sum())
R = np.array([0.5, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20])
def pairs(X): t = cKDTree(X); return np.array([t.count_neighbors(t, r) - len(X) for r in R]) / 2
obs = dict(nn_median=np.median(nn(P)), frac_nb_2p5=(nn(P) <= GLOM_UM).mean(), n_glom=nglom(P), pairs=pairs(P))
nul = [null_draw() for _ in range(NREP)]
nst = dict(nn_median=np.array([np.median(nn(X)) for X in nul]), frac_nb_2p5=np.array([(nn(X) <= GLOM_UM).mean() for X in nul]),
           n_glom=np.array([nglom(X) for X in nul]), pairs=np.array([pairs(X) for X in nul]))
pr = pd.DataFrame(dict(r_um=R, obs_pairs=obs['pairs'], null_mean=nst['pairs'].mean(0), null_lo=np.percentile(nst['pairs'], 2.5, 0), null_hi=np.percentile(nst['pairs'], 97.5, 0)))
pr['ratio'] = (pr.obs_pairs / pr.null_mean).round(2); pr.to_csv(f'{O}/clustering_pair_ratio.csv', index=False)
p_hi = lambda o, a: float(((a >= o).sum() + 1) / (len(a) + 1)); p_lo = lambda o, a: float(((a <= o).sum() + 1) / (len(a) + 1))
# 2. triad label shuffle: mean distance of each triadic bouton to the nearest other triadic bouton; fraction whose nearest bouton is triadic
D_nn_idx = cKDTree(P).query(P, 2)[1][:, 1]
def tri_stats(lab): return np.median(nn(P[lab])), (lab[D_nn_idx][lab]).mean()
t_obs = tri_stats(tri); t_nul = np.array([tri_stats(rng.permutation(tri)) for _ in range(10000)])
# 3. same-RGC: fraction of boutons whose nearest bouton is from the same RGC
rg = B.rgc.to_numpy(); has = ~np.isnan(rg)
same = lambda r: (r[D_nn_idx] == r)[has].mean()
s_obs = same(rg); s_nul = np.array([same(np.where(has, rng.permutation(rg), np.nan)) for _ in range(10000)])
# per-RGC spread: median pairwise distance among its boutons vs same number drawn from all boutons
rows = []
for r_, d in B[has].groupby('rgc'):
    if len(d) < 3: continue
    X = d[['wk_x', 'wk_y', 'wk_z']].to_numpy(float) * NM / 1000; m = lambda Y: np.median(np.linalg.norm(Y[:, None] - Y[None], axis=-1)[np.triu_indices(len(Y), 1)])
    nm_ = np.array([m(P[rng.choice(n, len(d), replace=False)]) for _ in range(2000)])
    rows.append(dict(rgc=int(r_), boutons=len(d), triadic=int(d.triad.sum()), median_pair_um=round(m(X), 1), random_median_pair_um=round(float(np.median(nm_)), 1), p_tighter=round(p_lo(m(X), nm_), 4),
                     soma_dist_um=round(float(np.median(dS(X))), 1)))
PR = pd.DataFrame(rows).sort_values('boutons', ascending=False); PR.to_csv(f'{O}/clustering_per_rgc.csv', index=False)
summ = dict(boutons=n, triadic=int(tri.sum()),
  vs_skeleton_null=dict(nn_median_um=[round(obs['nn_median'], 2), round(float(np.median(nst['nn_median'])), 2), round(p_lo(obs['nn_median'], nst['nn_median']), 4)],
                        frac_with_neighbour_within_2p5um=[round(obs['frac_nb_2p5'], 3), round(float(np.median(nst['frac_nb_2p5'])), 3), round(p_hi(obs['frac_nb_2p5'], nst['frac_nb_2p5']), 4)],
                        clusters_ge3_at_2p5um=[obs['n_glom'], float(np.median(nst['n_glom'])), round(p_hi(obs['n_glom'], nst['n_glom']), 4)],
                        note='[observed, null median, one-sided p]'),
  triad_label_shuffle=dict(nn_median_between_triadic_um=[round(t_obs[0], 2), round(float(np.median(t_nul[:, 0])), 2), round(p_lo(t_obs[0], t_nul[:, 0]), 4)],
                           frac_triadic_whose_nearest_bouton_is_triadic=[round(t_obs[1], 3), round(float(np.median(t_nul[:, 1])), 3), round(p_hi(t_obs[1], t_nul[:, 1]), 4)]),
  same_rgc_shuffle=dict(frac_nearest_bouton_same_rgc=[round(s_obs, 3), round(float(np.median(s_nul)), 3), round(p_hi(s_obs, s_nul), 4)]),
  soma_dist_um_median=dict(triadic=round(float(np.median(dS(P[tri]))), 1), non_triadic=round(float(np.median(dS(P[~tri]))), 1)))
json.dump(summ, open(f'{O}/clustering_summary.json', 'w'), indent=1); print(json.dumps(summ, indent=1)); print(pr.to_string(index=False)); print(PR.head(12).to_string(index=False))
# chart
INK, MUTED, ACC = '#1f1f1f', '#8a8a8a', '#2a6fdb'
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
a = ax[0]; a.fill_between(R, pr.null_lo / pr.null_mean, pr.null_hi / pr.null_mean, color=MUTED, alpha=0.25, lw=0, label='random on cell B (95%)')
a.axhline(1, color=MUTED, lw=1); a.plot(R, pr.ratio, color=ACC, lw=2, marker='o', ms=6, label='RGC boutons'); a.set_xscale('log')
a.set_xlabel('distance between boutons (µm)'); a.set_ylabel('bouton pairs, observed / random'); a.set_title('RGC boutons vs random spots on cell B\n(same distance-from-soma profile)', fontsize=10)
a.legend(frameon=False, fontsize=9)
for a, o, nl, ttl, xl in ((ax[1], t_obs[0], t_nul[:, 0], 'Triadic boutons: shuffle triad label', 'median distance to nearest other triadic bouton (µm)'),
                          (ax[2], s_obs, s_nul, 'Same RGC: shuffle RGC identity', 'fraction of boutons whose nearest bouton is the same RGC')):
    a.hist(nl, 40, color=MUTED, alpha=0.6, label='shuffled'); a.axvline(o, color=ACC, lw=2, label=f'observed {o:.2f}'); a.set_xlabel(xl, fontsize=9); a.set_title(ttl, fontsize=10); a.legend(frameon=False, fontsize=9)
for a in ax:
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    a.tick_params(colors=INK, labelsize=8)
plt.tight_layout(); plt.savefig(f'{O}/clustering.png', dpi=140)
