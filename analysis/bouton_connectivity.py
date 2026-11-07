"""Cell B bouton-level connectivity (pairing_graph_v2: graph segmentation v2, partner-fused synapses, cell B always post).
RGC bouton = RGC synapses onto cell B from one presynaptic segment, single-linkage within BOUTON_NM.
Triad = for an RGC -> cell B synapse s: the same RGC (same RGC number; same segment if unnumbered) synapses onto a segment X
        (not cell B, not an RGC segment), and X synapses onto cell B within TRIAD_NM of s.
Also: share of annotated 'post ffi' (interneuron -> cell B) sites caught.
Writes pairing_graph_v2/analysis/bouton_level/{rgc_boutons.csv, triads.csv, ffi_estimate.csv, summary.json, triads_wk.nml}"""
import json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fcluster, linkage
import os
R = os.environ.get('PAIR_RUN', 'pairing_graph_v2'); A = f'{R}/analysis'; OUT = os.environ.get('BC_OUT', f'{A}/bouton_level'); os.makedirs(OUT, exist_ok=True)
EXTRA = os.environ.get('BC_RGC_EXTRA')      # optional csv with pre_gid: extra RGC segments (e.g. mito-called), added with no RGC number
NM = np.array([4., 4., 30.]); BOUTON_NM, TRIAD_NM = 1500, 5000
BAD = ('long_ray', 'no_pre')
S = pd.read_csv(f'{R}/cellB_all_synapses.csv', low_memory=False, usecols=['uid', 'wk_x', 'wk_y', 'wk_z', 'pre_gid', 'post_gid', 'quality', 'az_area_um2', 'pre_type', 'post_type', 'pre_cell', 'post_cell'])
I = pd.read_csv(f'{A}/cellB_inputs.csv', low_memory=False); RB = pd.read_csv(f'{A}/cellB_rgc_synapses.csv')
K = pd.read_csv(f'{R}/cellB_skel_segs.csv'); CB = set(K[K.n_nodes_sum >= 3].seg_gid.astype(np.int64))
xyz = lambda d: d[['wk_x', 'wk_y', 'wk_z']].to_numpy(float) * NM
good = lambda d: d[~d.quality.fillna('').str.contains('|'.join(BAD))]
onB = set(I.uid)
# ---- RGC boutons on cell B
if EXTRA:
    eg = set(pd.read_csv(EXTRA).pre_gid.astype(np.int64))
    add = I[I.pre_gid.isin(eg) & I.rgc_num.isna() & ~I.uid.isin(RB.uid)][[c for c in RB.columns if c in I.columns]].assign(rgc_source='mito')
    print('extra RGC segments', len(eg), '-> extra RGC synapses onto cell B', len(add)); RB = pd.concat([RB, add], ignore_index=True)
RB = good(RB).copy(); RB['bouton'] = -1; nb = 0
for g, d in RB.groupby('pre_gid'):
    lab = fcluster(linkage(xyz(d), 'single'), BOUTON_NM, 'distance') if len(d) > 1 else np.array([1])
    RB.loc[d.index, 'bouton'] = lab + nb; nb += int(lab.max())
seg_rgc = RB.dropna(subset=['rgc_num']).groupby('pre_gid').rgc_num.agg(lambda x: x.mode()[0])           # RGC number per RGC segment
RGCSEG = set(RB.pre_gid.astype(np.int64)) | set(I[I.rgc_num.notna()].pre_gid.astype(np.int64))
num2segs = {}
for sgid, n in seg_rgc.items(): num2segs.setdefault(int(n), set()).add(int(sgid))
# ---- candidate interneuron pieces: X -> cell B synapses (pre not RGC)
# VAST identity of each segment, from the labels at the synapses it takes part in. Pieces that VAST calls TC or RGC (majority of labels),
# or any piece labelled cell 201 (cell B fragments without skeleton nodes), are not interneurons.
V = pd.concat([S[['pre_gid', 'pre_type', 'pre_cell']].set_axis(['gid', 'type', 'cell'], axis=1), S[['post_gid', 'post_type', 'post_cell']].set_axis(['gid', 'type', 'cell'], axis=1)])
V = V[V.type.notna()]
vtype = V.groupby('gid').type.agg(lambda x: x.mode()[0]); vB = set(V[V.cell == 201].gid)
NOTLIN = set(vtype[vtype.isin(['tcr', 'rgc'])].index) | vB
XB = good(I[(I.rgc_num.isna()) & ~I.pre_gid.isin(RGCSEG) & ~I.pre_gid.isin(NOTLIN) & (I.pre_gid > 0)]).copy()
xb_by = {g: d for g, d in XB.groupby('pre_gid')}
# ---- RGC -> X synapses (anything not onto cell B)
RX = good(S[~S.uid.isin(onB) & S.pre_gid.isin(RGCSEG) & ~S.post_gid.isin(CB) & (S.post_gid > 0)]).copy()
RX = RX[~RX.post_gid.isin(RGCSEG)]
rx_by = {g: d for g, d in RX.groupby('pre_gid')}
tri = []
for r in RB.itertuples():
    segs = num2segs.get(int(r.rgc_num), {int(r.pre_gid)}) if r.rgc_num == r.rgc_num else {int(r.pre_gid)}
    s0 = np.array([r.wk_x, r.wk_y, r.wk_z]) * NM
    for sg in segs:
        for q in (rx_by.get(sg, pd.DataFrame()).itertuples() if sg in rx_by else []):
            X = int(q.post_gid)
            if X not in xb_by: continue
            xb = xb_by[X]; d = np.linalg.norm(xyz(xb) - s0, axis=1)
            if (d <= TRIAD_NM).any():
                j = int(d.argmin()); b = xb.iloc[j]
                tri.append(dict(rgc_cellB_uid=r.uid, rgc=r.rgc_num, bouton=r.bouton, rgc_seg=sg, lin_seg=X, rgc_lin_uid=q.uid,
                                rgc_cellB_wk=f'{r.wk_x}, {r.wk_y}, {r.wk_z}', rgc_lin_wk=f'{q.wk_x}, {q.wk_y}, {q.wk_z}', lin_cellB_wk=f'{int(b.wk_x)}, {int(b.wk_y)}, {int(b.wk_z)}',
                                lin_cellB_to_rgc_cellB_nm=round(float(d[j])), rgc_lin_to_rgc_cellB_nm=round(float(np.linalg.norm(np.array([q.wk_x, q.wk_y, q.wk_z]) * NM - s0))), lin_vast=vtype.get(X, 'untraced')))
T = pd.DataFrame(tri).drop_duplicates(['rgc_cellB_uid', 'lin_seg']) if tri else pd.DataFrame(columns=['rgc_cellB_uid', 'bouton', 'lin_seg', 'rgc'])
T['same_bouton'] = T.rgc_lin_to_rgc_cellB_nm <= BOUTON_NM
RB['triad'] = RB.uid.isin(T.rgc_cellB_uid); RB['triad_same_bouton'] = RB.uid.isin(T[T.same_bouton].rgc_cellB_uid); RB['triad_vast_lin'] = RB.uid.isin(T[T.lin_vast == 'lin'].rgc_cellB_uid)
B = RB.groupby('bouton').agg(rgc=('rgc_num', lambda x: x.mode()[0] if x.notna().any() else np.nan), pre_gid=('pre_gid', 'first'), n_active_zones=('uid', 'size'),
                             az_sum_um2=('az_area_um2', 'sum'), wk_x=('wk_x', 'mean'), wk_y=('wk_y', 'mean'), wk_z=('wk_z', 'mean'), triad=('triad', 'any'), triad_same_bouton=('triad_same_bouton', 'any'), triad_vast_lin=('triad_vast_lin', 'any')).round({'az_sum_um2': 4, 'wk_x': 0, 'wk_y': 0, 'wk_z': 0})
B['lin_pieces'] = B.index.map(T.groupby('bouton').lin_seg.nunique() if len(T) else {}).fillna(0).astype(int)
B.to_csv(f'{OUT}/rgc_boutons.csv'); T.to_csv(f'{OUT}/triads.csv', index=False)
# ---- post ffi estimate
P = pd.read_csv(f'{A}/pair_check.csv'); F = P[(P.status == 'ok') & P.kind.str.startswith('post ffi')].copy()
tI = cKDTree(xyz(I)); rows = []
for r in F.itertuples():
    w = np.array([float(v) for v in r.tip_wk.split(',')]) * NM; idx = tI.query_ball_point(w, 300)
    near = I.iloc[idx]; nonrgc = near[near.rgc_num.isna() & ~near.pre_gid.isin(RGCSEG)]
    segs = {int(s) for s in str(r.segments_past_tip).split(';') if s and s != 'nan'}
    rows.append(dict(who=r.who, tip_wk=r.tip_wk, any_input_300nm=len(near) > 0, non_rgc_input_300nm=len(nonrgc) > 0,
                     input_from_pointed_segment=bool(I.pre_gid.isin(segs).any()) if segs else False))
FE = pd.DataFrame(rows); FE.to_csv(f'{OUT}/ffi_estimate.csv', index=False)
summ = dict(rgc_synapses_onto_cellB=len(RB), rgc_boutons_onto_cellB=int(B.shape[0]), rgc_cells=int(RB.rgc_num.nunique()), rgc_boutons_vast=int(B.rgc.notna().sum()), rgc_boutons_mito_only=int(B.rgc.isna().sum()),
            active_zones_per_bouton_median=float(B.n_active_zones.median()), boutons_with_triad=int(B.triad.sum()), boutons_triad_frac=round(B.triad.mean(), 3),
            boutons_triad_same_bouton=int(B.triad_same_bouton.sum()), boutons_triad_vast_lin_only=int(B.triad_vast_lin.sum()),
            rgc_synapses_in_triad=int(RB.triad.sum()), rgc_synapses_triad_frac=round(RB.triad.mean(), 3), distinct_lin_pieces_in_triads=int(T.lin_seg.nunique()) if len(T) else 0, lin_pieces_by_vast=T.drop_duplicates('lin_seg').lin_vast.value_counts().to_dict() if len(T) else {},
            excluded_notlin_pieces=len(NOTLIN),
            candidate_lin_pieces_onto_cellB=len(xb_by), rgc_to_nonB_synapses=len(RX),
            ffi_tips=len(FE), ffi_any_input_300nm=int(FE.any_input_300nm.sum()), ffi_non_rgc_input_300nm=int(FE.non_rgc_input_300nm.sum()), ffi_pointed_segment_onto_cellB=int(FE.input_from_pointed_segment.sum()),
            params=dict(BOUTON_NM=BOUTON_NM, TRIAD_NM=TRIAD_NM))
json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1); print(json.dumps(summ, indent=1))
pr = RB.groupby('rgc_num').agg(boutons=('bouton', 'nunique'), synapses=('uid', 'size'), triad_boutons=('bouton', lambda b: B.loc[b.unique()].triad.sum())).sort_values('boutons', ascending=False)
print('\nper RGC (top 15 by boutons):'); print(pr.head(15).to_string())
with open(f'{OUT}/triads_wk.nml', 'w') as fh:
    fh.write('<things>\n<parameters>\n<experiment name="Morgan" />\n<scale x="4" y="4" z="30" />\n</parameters>\n'); nid = 0
    for k, t in enumerate(T.itertuples(), 1):
        pts = [t.rgc_cellB_wk, t.rgc_lin_wk, t.lin_cellB_wk]
        fh.write(f'<thing id="{k}" color.r="1" color.g="0" color.b="1" color.a="1" name="triad rgc{t.rgc} bouton{t.bouton}: RGC->B, RGC->LIN {t.lin_seg}, LIN->B ({t.lin_cellB_to_rgc_cellB_nm} nm)"><nodes>')
        for p in pts:
            x, y, z = (int(v) for v in p.split(',')); nid += 1; fh.write(f'<node id="{nid}" x="{x}" y="{y}" z="{z}" radius="80"/>')
        fh.write(f'</nodes><edges><edge source="{nid - 2}" target="{nid - 1}"/><edge source="{nid - 1}" target="{nid}"/></edges></thing>\n')
    fh.write('</things>\n')
