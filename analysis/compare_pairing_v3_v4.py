"""Graph v4 vs v3 inputs onto cell B: which inputs are new / lost / kept, how far each set is from the cell B skeleton (far = likely a
merge into a cell B object, not a real input), and which objects bring them. Writes pairing_graph_v4/analysis/compare_v3/."""
import os, numpy as np, pandas as pd
from scipy.spatial import cKDTree
A = 'pairing_graph_v3/analysis'; B = 'pairing_graph_v4/analysis'; O = f'{B}/compare_v3'; os.makedirs(O, exist_ok=True)
S = np.load('pairing_graph_v4/skeleton_nodes.npy') * [4, 4, 30] / 1000; T = cKDTree(S)
key = ['wk_x', 'wk_y', 'wk_z']
a = pd.read_csv(f'{A}/cellB_inputs.csv', low_memory=False); b = pd.read_csv(f'{B}/cellB_inputs.csv', low_memory=False)
for d in (a, b): d['skel_um'] = T.query(d[key].to_numpy() * [4, 4, 30] / 1000)[0]
ka = set(map(tuple, a[key].to_numpy())); kb = set(map(tuple, b[key].to_numpy()))
b['set'] = ['kept' if k in ka else 'new' for k in map(tuple, b[key].to_numpy())]; a['set'] = ['kept' if k in kb else 'lost' for k in map(tuple, a[key].to_numpy())]
rows = []
for nm, d in (('v3 all', a), ('v4 all', b), ('kept (v4 side)', b[b.set == 'kept']), ('new in v4', b[b.set == 'new']), ('lost from v3', a[a.set == 'lost'])):
    q = d.skel_um; rows.append(dict(set=nm, n=len(d), median_um=round(q.median(), 3), p90_um=round(q.quantile(.9), 3), p99_um=round(q.quantile(.99), 3), over_1um=int((q > 1).sum()), over_3um=int((q > 3).sum()),
                                    flagged=int((d.quality != 'ok').sum()), median_az_um2=round(d.az_area_um2.median(), 4), auto_only=int((d.onB_auto & ~d.onB_manual).sum())))
R = pd.DataFrame(rows); R.to_csv(f'{O}/input_sets.csv', index=False); print(R.to_string(index=False))
n = b[b.set == 'new']
by = n.groupby('post_gid').agg(n=('uid', 'size'), median_um=('skel_um', 'median'), max_um=('skel_um', 'max'), wk_x=('wk_x', 'median'), wk_y=('wk_y', 'median'), wk_z=('wk_z', 'median')).sort_values('n', ascending=False)
by.to_csv(f'{O}/new_inputs_by_post_object.csv'); print('\nnew inputs by post object (top 12 of', len(by), '):'); print(by.head(12).to_string())
for nm, d in (('v3', a), ('v4', b)):
    c = d.groupby('post_gid').size().sort_values(ascending=False); print(f'{nm}: {len(c)} post objects; largest {c.iloc[:5].tolist()}')
b.to_csv(f'{O}/v4_inputs_with_set.csv', index=False); a[a.set == 'lost'].to_csv(f'{O}/v3_lost_inputs.csv', index=False)

# Where were the new inputs in v3, and does VAST name another cell at them? A manual label != 201 at the post side = certainly not cell B.
c = ['uid', 'wk_x', 'wk_y', 'wk_z', 'post_gid']
v3 = pd.read_csv('pairing_graph_v3/cellB_all_synapses.csv', usecols=c, low_memory=False).rename(columns={'post_gid': 'v3_post_gid'})[key + ['v3_post_gid']].drop_duplicates(key)
isB3 = set(a.post_gid); m = b.merge(v3, on=key, how='left'); m['v3_post_was_cellB'] = m.v3_post_gid.isin(isB3)
m['label'] = np.where(m.post_cell.isna(), 'no VAST cell', np.where(m.post_cell.astype(str).str.replace('.0', '', regex=False) == '201', 'cell 201', 'OTHER cell'))
L = m.groupby(['set', 'label']).size().unstack(fill_value=0); L['total'] = L.sum(1); L['other_%_of_labelled'] = (100 * L.get('OTHER cell', 0) / (L.get('OTHER cell', 0) + L.get('cell 201', 0)).clip(lower=1)).round(1)
L.to_csv(f'{O}/post_label_by_set.csv'); print('\nVAST post label by set:'); print(L.to_string())
n = m[m.set == 'new']; print('\nnew inputs: v3 post object present', int(n.v3_post_gid.notna().sum()), '; was a cell B object in v3', int(n.v3_post_was_cellB.sum()))
g = n[~n.v3_post_was_cellB].groupby('v3_post_gid').size().sort_values(ascending=False); print('top v3 objects they came from:', g.head(8).to_dict(), 'of', len(g))
print('new OTHER-labelled cells:', n[n.label == 'OTHER cell'].post_cell.value_counts().head(10).to_dict())

# Same test on the inputs lost from v3, the RGC calls, and the sites to look at (v3 object that became cell B in v4, per chunk).
l = a[a.set == 'lost']; print('\nlost from v3: VAST cell 201', int((l.post_cell == 201).sum()), '; other cell', int(((l.post_cell != 201) & l.post_cell.notna()).sum()), '; none', int(l.post_cell.isna().sum()), 'of', len(l))
ra = pd.read_csv(f'{A}/cellB_rgc_synapses.csv', low_memory=False); rb = pd.read_csv(f'{B}/cellB_rgc_synapses.csv', low_memory=False)
kra = set(map(tuple, ra[key].to_numpy())); rb['new'] = [k not in kra for k in map(tuple, rb[key].to_numpy())]; krb = set(map(tuple, rb[key].to_numpy())); ra['lost'] = [k not in krb for k in map(tuple, ra[key].to_numpy())]
print('RGC synapses: v3', len(ra), 'v4', len(rb), '; new in v4', int(rb.new.sum()), '; lost', int(ra.lost.sum())); print(rb[rb.new].groupby('rgc_source').size().to_dict(), '| lost:', ra[ra.lost].groupby('rgc_source').size().to_dict())
rb[rb.new].to_csv(f'{O}/rgc_new_in_v4.csv', index=False); ra[ra.lost].to_csv(f'{O}/rgc_lost_from_v3.csv', index=False)
f = m[(m.set == 'new') & ~m.v3_post_was_cellB & m.v3_post_gid.notna()]
s = f.groupby('v3_post_gid').agg(n=('uid', 'size'), v4_post=('post_gid', lambda x: x.value_counts().index[0]), skel_um=('skel_um', 'median'), wk_x=('wk_x', 'median'), wk_y=('wk_y', 'median'), wk_z=('wk_z', 'median'), vast201=('label', lambda x: int((x == 'cell 201').sum()))).sort_values('n', ascending=False)
s.to_csv(f'{O}/merge_sites_v3_object_into_cellB.csv'); print('\ntop v3 objects whose synapses became cell B inputs in v4:'); print(s.head(10).round(2).to_string())
print('share of new inputs in the top 10 objects:', round(s.n.head(10).sum() / len(n), 2))

# One real synapse per top site (nearest the site's median) to open in webKnossos, and the headline numbers side by side.
top = []
for gid, r in s.head(6).iterrows():
    g = f[f.v3_post_gid == gid]; i = ((g[key] - g[key].median()) ** 2 * [1, 1, 56]).sum(1).idxmin(); q = g.loc[i]
    top.append(dict(v3_object=int(gid), v4_cellB_object=int(q.post_gid), new_inputs=int(r.n), wk=f'{int(q.wk_x)}, {int(q.wk_y)}, {int(q.wk_z)}', skel_um=round(q.skel_um, 2)))
Tp = pd.DataFrame(top); Tp.to_csv(f'{O}/sites_to_view.csv', index=False); print(Tp.to_string(index=False))
import json
ja, jb = json.load(open(f'{A}/cellB_summary.json')), json.load(open(f'{B}/cellB_summary.json'))
H = pd.DataFrame([dict(metric=k, v3=ja[k], v4=jb[k]) for k in ja if k in jb and not isinstance(ja[k], (dict,))]); H.to_csv(f'{O}/summary_v3_v4.csv', index=False)
