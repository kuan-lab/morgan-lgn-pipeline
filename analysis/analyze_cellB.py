"""Cell B (VAST cell 201) input connectivity from the automatic synapses + merged segmentation.

  python analyze_cellB.py [pairing_dir]
Input: <pairing_dir>/cellB_all_synapses.csv, cellB_skel_segs.csv (collect_cellB.py); the manual sheet of RGC
synapses onto cell 201 (cell201_rgc_synapse_assignments.csv) is used as ground truth for recall.

Definitions
- segments are the chunk-local ones (stitch_local.py). A segment's identity across chunks is its object in the global
  stitch, except for the global objects that hold cell B (over-merged with its inputs): those stay chunk-local.
- onto cell B: the chunk-local postsynaptic segment holds >= MIN_NODES cell B skeleton nodes, unless the manual label
  at the postsynaptic side names another cell; or the manual label says cell 201.
- RGC synapse: manual label names an RGC ("201 part <rgc>" at the synapse, or an RGC cell on the presynaptic side),
  or the presynaptic automatic segment carries such a label on another of its synapses (label spreads over the segment).
- triad: RGC bouton R -> cell B, and within TRIAD_NM of that synapse some other segment I (not cell B, not an RGC)
  that both receives a synapse from R and makes a synapse onto cell B.
- glomerulus: single-linkage cluster of RGC synapses onto cell B (GLOM_NM), with >= GLOM_MIN synapses.
Writes <pairing_dir>/analysis/: cellB_inputs.csv, cellB_rgc_synapses.csv, cellB_glomeruli.csv, cellB_rgc_synapses_wk.nml,
cellB_gt_recall.csv, cellB_summary.json, cellB_glomeruli.png
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = '/gpfs/radev/project/kuan/ntl5/cell_A_full'
R = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else f'{ROOT}/cellB_connectivity/pairing'
A = f'{R}/analysis'; os.makedirs(A, exist_ok=True)
NM = np.array([4., 4., 30.])
MIN_NODES, TRIAD_NM, GLOM_NM, GLOM_MIN, MATCH_NM = 3, 2000, 2500, 3, 1000
SOMA = np.array([89752, 87832, 3839])
BAD = ('long_ray', 'no_pre')

S = pd.read_csv(f'{R}/cellB_all_synapses.csv', low_memory=False)
K = pd.read_csv(f'{R}/cellB_skel_segs.csv')
g = pd.read_csv(f'{ROOT}/segment_groups.csv')
TYPE = g[g.role == 'cell'].dropna(subset=['cell_num']).drop_duplicates('cell_num').set_index('cell_num').cell_type.to_dict()
S['xyz'] = list(S[['wk_x', 'wk_y', 'wk_z']].to_numpy(float) * NM)
S['flagged'] = S.quality.fillna('').apply(lambda q: any(b in q for b in BAD))

# ---- cell B segments and synapses onto it (decided inside each chunk: the global stitch over-merges cell B with its inputs)
# direction prior (2026-10-22): cell B is a TC cell and makes no output synapses, so when the pairing put cell B on the
# presynaptic side and another segment on the postsynaptic side (the instance lay mostly inside the bouton), swap them.
SW = (S.pre_skel_nodes >= MIN_NODES) & (S.post_skel_nodes < MIN_NODES) & (S.post_id > 0)
for a_, b_ in (('pre_id', 'post_id'), ('pre_gid', 'post_gid'), ('pre_skel_nodes', 'post_skel_nodes'), ('pre_cell', 'post_cell'), ('pre_type', 'post_type')):
    S.loc[SW, [a_, b_]] = S.loc[SW, [b_, a_]].to_numpy()
S['swapped_cellB_to_post'] = SW
S['post_isB'] = S.post_skel_nodes >= MIN_NODES; S['pre_isB'] = S.pre_skel_nodes >= MIN_NODES
BG = set(K[K.n_nodes_max >= MIN_NODES].seg_gid.astype(int))            # global objects that hold cell B: not usable as an identity
other = set()
def eid(gid, chunk, local):      # identity of a segment: the global object, unless that object is one of cell B's (then chunk-local)
    return np.where(gid.isin(BG) | (gid == 0), 'c:' + chunk + ':' + local.astype(str), 'g:' + gid.astype(str))
S['pre_eid'] = eid(S.pre_gid, S.chunk, S.pre_id); S['post_eid'] = eid(S.post_gid, S.chunk, S.post_id)
S['onB_auto'] = S.post_isB; S['onB_manual'] = S.post_cell == 201
# a manual label on the post side only vetoes the skeleton when it names a cell that can be postsynaptic (not an RGC:
# a lookup that lands in an RGC hit the bouton, i.e. the presynaptic side)
S['onB'] = (S.onB_auto & (S.post_cell.isna() | S.onB_manual | (S.post_type == 'rgc'))) | S.onB_manual

# ---- RGC identity
S['rgc_at_syn'] = np.where((S.ann_target == 201) | S.ann_target.isna(), S.ann_pre, np.nan)
S['rgc_at_syn'] = np.where(S.rgc_at_syn.map(TYPE) == 'rgc', S.rgc_at_syn, np.where(S.pre_type == 'rgc', S.pre_cell, np.nan))
lab = S[(S.pre_id > 0) & ~S.pre_isB]
seg_rgc = lab.dropna(subset=['rgc_at_syn']).groupby('pre_eid').rgc_at_syn.agg(lambda x: x.value_counts().index[0])
non = lab[lab.pre_type.notna() & (lab.pre_type != 'rgc')].groupby('pre_eid').size()
yes = lab.dropna(subset=['rgc_at_syn']).groupby('pre_eid').size()
seg_rgc = seg_rgc[[i for i in seg_rgc.index if yes[i] >= non.get(i, 0)]]
RG = set(seg_rgc.index)
S['rgc_num'] = np.where(S.rgc_at_syn.notna(), S.rgc_at_syn, np.where(S.pre_isB, np.nan, S.pre_eid.map(seg_rgc)))
S['rgc_source'] = np.where(S.rgc_at_syn.notna(), 'manual label at synapse', np.where(S.rgc_num.notna(), 'same segment as a labelled synapse', ''))
B = S[S.onB].copy()
RB = B[B.rgc_num.notna() & ~B.flagged].copy().reset_index(drop=True)

# ---- triads
ok = S[~S.flagged & (S.pre_id > 0) & (S.post_id > 0)].reset_index(drop=True)
tree = cKDTree(np.stack(ok.xyz)); okB = ok.onB.to_numpy(); opre = ok.pre_eid.to_numpy(); opost = ok.post_eid.to_numpy()
opreB = ok.pre_isB.to_numpy(); opostB = ok.post_isB.to_numpy(); opc = ok.pre_cell.to_numpy(); oqc = ok.post_cell.to_numpy()
tri, inter, inter_cells = [], [], []
for r in RB.itertuples():
    nb = np.array(tree.query_ball_point(r.xyz, TRIAD_NM), int)
    out = set(opost[nb[(opre[nb] == r.pre_eid) & ~okB[nb] & ~opostB[nb]]]) - RG                 # R -> I
    back = set(opre[nb[okB[nb] & (opre[nb] != r.pre_eid) & ~opreB[nb]]]) - RG                    # I -> cell B
    I = sorted(out & back)
    tri.append(bool(I)); inter.append(';'.join(I))
    cells = set()
    for i in I:
        c = opc[nb[(opre[nb] == i) & okB[nb]]]; cells |= {int(x) for x in c[~np.isnan(c)]}
        c = oqc[nb[(opost[nb] == i) & (opre[nb] == r.pre_eid)]]; cells |= {int(x) for x in c[~np.isnan(c)]}
    inter_cells.append(';'.join(map(str, sorted(cells - {201}))))
RB['triad'] = tri; RB['interneuron_segs'] = inter; RB['interneuron_manual_cells'] = inter_cells

# ---- glomeruli
X = np.stack(RB.xyz)
RB['glom'] = fcluster(linkage(X, 'single'), GLOM_NM, 'distance') if len(RB) > 1 else 1
sz = RB.groupby('glom').size(); RB['in_glom'] = RB.glom.map(sz) >= GLOM_MIN
RB['soma_dist_um'] = np.linalg.norm(X - SOMA * NM, axis=1) / 1000
rows = []
for gid, d in RB[RB.in_glom].groupby('glom'):
    c = np.stack(d.xyz).mean(0) / NM
    bt = d.groupby('pre_eid').triad.any()
    segs = sorted({s for x in d.interneuron_segs for s in x.split(';') if s}); cells = sorted({s for x in d.interneuron_manual_cells for s in x.split(';') if s})
    rows.append(dict(glom=int(gid), wk_x=int(c[0]), wk_y=int(c[1]), wk_z=int(c[2]), n_rgc_syn=len(d), n_rgc_boutons=int(d.pre_eid.nunique()),
                     n_rgc_cells=int(d.rgc_num.nunique()), rgc_cells=';'.join(str(int(x)) for x in sorted(d.rgc_num.unique())),
                     n_triad_syn=int(d.triad.sum()), triad_frac_syn=round(d.triad.mean(), 3), n_triad_boutons=int(bt.sum()), triad_frac_boutons=round(bt.mean(), 3),
                     n_interneuron_segs=len(segs), n_interneuron_manual_cells=len(cells), interneuron_manual_cells=';'.join(cells),
                     az_sum_um2=round(d.az_area_um2.sum(), 3), az_median_um2=round(d.az_area_um2.median(), 4),
                     extent_um=round(float(np.ptp(np.stack(d.xyz), axis=0).max() / 1000), 2), soma_dist_um=round(d.soma_dist_um.mean(), 1)))
G = pd.DataFrame(rows).sort_values('n_rgc_syn', ascending=False)

# ---- recall against the manual sheet
gt = pd.read_csv(f'{ROOT}/cell201_rgc_synapse_assignments.csv')
gt = gt[gt.target_cell_num == 201].reset_index(drop=True)      # the sheet lists every synapse of the RGCs that contact cell 201; keep those onto 201
gx = gt.anchor.str.split(',', expand=True).astype(float).to_numpy() * NM
tB = cKDTree(np.stack(B.xyz)) if len(B) else None; tA = cKDTree(np.stack(S.xyz))
dB, iB = tB.query(gx); dA, _ = tA.query(gx)
gt['nearest_any_synapse_nm'] = dA.round(0); gt['nearest_onB_nm'] = dB.round(0); gt['detected'] = dB <= MATCH_NM
m = B.iloc[iB]
gt['matched_uid'] = np.where(gt.detected, m.uid.to_numpy(), -1); gt['matched_called_rgc'] = gt.detected & m.rgc_num.notna().to_numpy()
gt['matched_same_rgc'] = gt.detected & (m.rgc_num.to_numpy() == gt.rgc_num.to_numpy())
gt.to_csv(f'{A}/cellB_gt_recall.csv', index=False)

# ---- statistics: does the triad rate differ between glomeruli? (bouton level; permutation of triad labels)
bb = RB[RB.in_glom].groupby(['glom', 'pre_eid']).triad.any().reset_index()
big = bb.groupby('glom').filter(lambda d: len(d) >= 4)
stat = lambda d: d.groupby('glom').triad.mean().var()
p_perm = np.nan
if big.glom.nunique() >= 3:
    obs = stat(big); rng = np.random.default_rng(0); sh = big.copy(); null = []
    for _ in range(5000):
        sh['triad'] = rng.permutation(big.triad.to_numpy()); null.append(stat(sh))
    p_perm = float((np.array(null) >= obs).mean())
from scipy.stats import spearmanr
rho, p_rho = spearmanr(G.n_rgc_syn, G.triad_frac_syn) if len(G) >= 5 else (np.nan, np.nan)
rho_d, p_d = spearmanr(G.soma_dist_um, G.triad_frac_syn) if len(G) >= 5 else (np.nan, np.nan)

keep = ['uid', 'wk_x', 'wk_y', 'wk_z', 'az_area_um2', 'az_hull_um2', 'n_vox', 'quality', 'pre_gid', 'post_gid', 'pre_eid', 'post_eid', 'pre_ray_nm', 'pre_cell', 'pre_type', 'post_cell',
        'vast_name_at_syn', 'rgc_num', 'rgc_source', 'onB_auto', 'onB_manual', 'swapped_cellB_to_post', 'chunk']
B = S[S.onB]; B[keep].to_csv(f'{A}/cellB_inputs.csv', index=False)
RB[keep + ['triad', 'interneuron_segs', 'interneuron_manual_cells', 'glom', 'in_glom', 'soma_dist_um']].to_csv(f'{A}/cellB_rgc_synapses.csv', index=False)
G.to_csv(f'{A}/cellB_glomeruli.csv', index=False)

# ---- NML (wk 4x4x30): triad = magenta, not triad = cyan, RGC synapse outside a glomerulus keeps the same colours but is named 'solo';
#      manual RGC synapses that were not detected = yellow
with open(f'{A}/cellB_rgc_synapses_wk.nml', 'w') as f:
    f.write('<things>\n<parameters>\n<experiment name="Morgan" />\n<scale x="4" y="4" z="30" />\n</parameters>\n')
    n = 0
    for r in RB.itertuples():
        n += 1; col = (1, 0, 1) if r.triad else (0, 0.8, 1)
        name = f"{'TRIAD' if r.triad else 'no_triad'} {'glom' + str(r.glom) if r.in_glom else 'solo'} rgc{int(r.rgc_num)} az{r.az_area_um2:.3f}um2 syn{r.uid}"
        f.write(f'<thing id="{n}" color.r="{col[0]}" color.g="{col[1]}" color.b="{col[2]}" color.a="1" name="{name}"><nodes>'
                f'<node id="{n}" x="{r.wk_x}" y="{r.wk_y}" z="{r.wk_z}" radius="120"/></nodes><edges/></thing>\n')
    for r in gt[~gt.detected].itertuples():
        n += 1; x, y, z = [int(float(v)) for v in r.anchor.split(',')]
        f.write(f'<thing id="{n}" color.r="1" color.g="0.85" color.b="0" color.a="1" name="MISSED manual {r.syn_name}"><nodes>'
                f'<node id="{n}" x="{x}" y="{y}" z="{z}" radius="120"/></nodes><edges/></thing>\n')
    f.write('</things>\n')

summ = dict(synapses_in_tube=len(S), swapped_cellB_to_post=int(SW.sum()), global_objects_holding_cellB=len(BG),
            inputs_onto_cellB=int(S.onB.sum()), inputs_auto_only=int((S.onB & ~S.onB_manual).sum()), inputs_manual_only=int((S.onB & ~S.onB_auto).sum()),
            inputs_flagged=int((S.onB & S.flagged).sum()),
            rgc_synapses_onto_cellB=len(RB), rgc_cells=int(RB.rgc_num.nunique()), rgc_boutons=int(RB.pre_eid.nunique()),
            rgc_by_source=RB.rgc_source.value_counts().to_dict(),
            manual_rgc_synapses=len(gt), manual_detected=int(gt.detected.sum()), manual_recall=round(gt.detected.mean(), 3),
            manual_detected_called_rgc=int(gt.matched_called_rgc.sum()), manual_detected_same_rgc=int(gt.matched_same_rgc.sum()),
            triad_synapses=int(RB.triad.sum()), triad_frac_syn=round(RB.triad.mean(), 3),
            triad_frac_boutons=round(RB.groupby('pre_eid').triad.any().mean(), 3),
            glomeruli=len(G), rgc_syn_in_glomeruli=int(RB.in_glom.sum()), median_syn_per_glom=float(G.n_rgc_syn.median()) if len(G) else None,
            triad_frac_in_glom=round(RB[RB.in_glom].triad.mean(), 3), triad_frac_solo=round(RB[~RB.in_glom].triad.mean(), 3) if (~RB.in_glom).any() else None,
            glomeruli_with_any_triad=int((G.n_triad_syn > 0).sum()) if len(G) else 0,
            median_interneuron_segs_per_triad_glom=float(G[G.n_triad_syn > 0].n_interneuron_segs.median()) if len(G) and (G.n_triad_syn > 0).any() else None,
            triad_rate_differs_between_glomeruli_p=p_perm, glomeruli_in_test=int(big.glom.nunique()),
            spearman_triadfrac_vs_size=[round(float(rho), 3), round(float(p_rho), 4)], spearman_triadfrac_vs_soma_dist=[round(float(rho_d), 3), round(float(p_d), 4)],
            params=dict(MIN_NODES=MIN_NODES, TRIAD_NM=TRIAD_NM, GLOM_NM=GLOM_NM, GLOM_MIN=GLOM_MIN, MATCH_NM=MATCH_NM))
json.dump(summ, open(f'{A}/cellB_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
print(G.head(25).drop(columns=['rgc_cells', 'interneuron_manual_cells']).to_string(index=False))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, MUTED, MARK = '#111827', '#6b7280', '#2563eb'
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
ax[0].hist(G.n_rgc_syn, bins=range(GLOM_MIN, int(G.n_rgc_syn.max()) + 2), color=MARK, edgecolor='white'); ax[0].set_title('RGC synapses per glomerulus', loc='left', fontsize=10, color=INK); ax[0].set_xlabel('synapses', color=MUTED); ax[0].set_ylabel('glomeruli', color=MUTED)
ax[1].hist(G.triad_frac_syn, bins=np.linspace(0, 1, 11), color=MARK, edgecolor='white'); ax[1].set_title('Share of RGC synapses in a triad, per glomerulus', loc='left', fontsize=10, color=INK); ax[1].set_xlabel('triad fraction', color=MUTED)
ax[2].scatter(G.n_rgc_syn, G.triad_frac_syn, s=18, color=MARK, alpha=.7); ax[2].set_title(f'Triad fraction vs glomerulus size (rho={rho:.2f}, p={p_rho:.3f})', loc='left', fontsize=10, color=INK); ax[2].set_xlabel('RGC synapses in glomerulus', color=MUTED)
t = G[G.n_triad_syn > 0]
ax[3].hist(t.n_interneuron_segs, bins=range(1, int(t.n_interneuron_segs.max()) + 2) if len(t) else 5, color=MARK, edgecolor='white'); ax[3].set_title('Interneuron pieces in triads, per glomerulus', loc='left', fontsize=10, color=INK); ax[3].set_xlabel('distinct automatic segments', color=MUTED)
for a in ax:
    a.spines[['top', 'right']].set_visible(False); a.tick_params(colors=MUTED, labelsize=8); a.grid(axis='y', color='#e5e7eb', linewidth=0.8); a.set_axisbelow(True)
plt.tight_layout(); plt.savefig(f'{A}/cellB_glomeruli.png', dpi=160)
