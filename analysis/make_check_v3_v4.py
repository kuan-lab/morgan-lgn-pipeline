"""Files to check by eye whether inputs that differ between graph v3 and v4 are really on cell B: top merge sites (20 biggest objects) + a random
sample (40 new in v4, 20 lost from v3, 20 kept in both as control, seed 0). Writes compare_v3/check_inputs_wk.nml + check_inputs.xlsx (same ids)."""
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
O = 'pairing_graph_v4/analysis/compare_v3'; key = ['wk_x', 'wk_y', 'wk_z']; PER_SITE, N_SITES = 5, 20
b = pd.read_csv(f'{O}/v4_inputs_with_set.csv', low_memory=False); l = pd.read_csv(f'{O}/v3_lost_inputs.csv', low_memory=False); s = pd.read_csv(f'{O}/merge_sites_v3_object_into_cellB.csv')
rd = lambda v, nm: pd.read_csv(f'pairing_graph_{v}/cellB_all_synapses.csv', usecols=key + ['post_gid'], low_memory=False).rename(columns={'post_gid': nm}).drop_duplicates(key)
b = b.merge(rd('v3', 'v3_object'), on=key, how='left').rename(columns={'post_gid': 'v4_object'}); l = l.merge(rd('v4', 'v4_object'), on=key, how='left').rename(columns={'post_gid': 'v3_object'})
rng = np.random.default_rng(0); pick = lambda d, n: d.iloc[rng.choice(len(d), n, replace=False)]
ROWS, TREES = [], []; ids = lambda v: '' if pd.isna(v) else str(int(v))


def row(i, group, q, look, n_site=''):
    return dict(id=i, group=group, wk=f'{int(q.wk_x)}, {int(q.wk_y)}, {int(q.wk_z)}', dist_to_skeleton_um=round(float(q.skel_um), 2), look_in=look, v3_object=ids(q.v3_object), v4_object=ids(q.v4_object),
                presyn_object=ids(q.pre_gid), inputs_at_site=n_site, vast_cell=ids(q.post_cell), quality=q.quality, verdict='', notes='')


new = b[b.set == 'new']
for k, r in enumerate(s.head(N_SITES).itertuples(), 1):
    g = new[new.v3_object == r.v3_post_gid]; d = ((g[key] - g[key].median()) ** 2 * [1, 1, 56]).sum(1); g = g.loc[d.sort_values().index[:PER_SITE]]
    i = f'S{k:02d}'; ROWS.append(row(i, 'top site', g.iloc[0], 'v4', int(r.n)))
    TREES.append((1, f'{i} v3 {int(r.v3_post_gid)} into v4 {int(r.v4_post)} ({int(r.n)} new inputs)', g, (1, .15, .15)))
for grp, pre, gid, d, n, look, col in ((2, 'N', 'random new in v4', new, 40, 'v4', (1, .55, 0)), (3, 'L', 'random lost from v3', l, 20, 'v3', (.2, .45, 1)), (4, 'K', 'random kept (control)', b[b.set == 'kept'], 20, 'v4', (0, .75, .3))):
    for k, (_, q) in enumerate(pick(d, n).iterrows(), 1):
        i = f'{pre}{k:02d}'; ROWS.append(row(i, gid, q, look)); TREES.append((grp, f'{i} {gid} ({q.skel_um:.1f} um from skeleton)', q.to_frame().T, col))
T = pd.DataFrame(ROWS)
esc = lambda t: t.replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;')
c0 = T.wk[0].split(', ')
with open(f'{O}/check_inputs_wk.nml', 'w') as f:
    f.write(f'<things>\n  <meta name="writer" content="make_check_v3_v4.py" />\n  <parameters>\n    <experiment name="Morgan" description="graph v3 vs v4 inputs onto cell B" />\n    <scale x="4" y="4" z="30" unit="nanometer" />\n'
            f'    <offset x="0" y="0" z="0" />\n    <editPosition x="{c0[0]}" y="{c0[1]}" z="{c0[2]}" />\n    <editRotation xRot="0" yRot="0" zRot="0" />\n    <zoomLevel zoom="1" />\n  </parameters>\n')
    nid = 0
    for t, (grp, name, g, col) in enumerate(TREES, 1):
        f.write(f'  <thing id="{t}" groupId="{grp}" color.r="{col[0]}" color.g="{col[1]}" color.b="{col[2]}" color.a="1.0" name="{esc(name)}">\n    <nodes>\n'); a = nid + 1
        for q in g.itertuples(): nid += 1; f.write(f'      <node id="{nid}" x="{int(q.wk_x)}" y="{int(q.wk_y)}" z="{int(q.wk_z)}" radius="120" />\n')
        f.write('    </nodes>\n    <edges>\n' + ''.join(f'      <edge source="{e}" target="{e + 1}" />\n' for e in range(a, nid)) + '    </edges>\n  </thing>\n')
    f.write('  <branchpoints></branchpoints>\n  <comments></comments>\n  <groups>\n    <group id="1" name="top sites: v3 object now counted as cell B" />\n    <group id="2" name="random new in v4" />\n'
            '    <group id="3" name="random lost from v3 (view with v3 seg)" />\n    <group id="4" name="random kept in both (control)" />\n  </groups>\n</things>\n')
W = Workbook(); w = W.active; w.title = 'check'; cols = list(T.columns); w.append(cols)
for r in T.itertuples(index=False): w.append(list(r))
fill = dict(zip(('top site', 'random new in v4', 'random lost from v3', 'random kept (control)'), ('F8CBAD', 'FFE699', 'BDD7EE', 'C6E0B4')))
for c in w[1]: c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical='top')
for i, r in enumerate(w.iter_rows(min_row=2), 2): r[1].fill = PatternFill('solid', fgColor=fill[r[1].value])
for c, wd in zip('ABCDEFGHIJKLM', (6, 22, 21, 11, 7, 16, 16, 16, 9, 9, 9, 12, 30)): w.column_dimensions[c].width = wd
w.freeze_panes = 'D2'; w.auto_filter.ref = w.dimensions
v = DataValidation(type='list', formula1='"on cell B,other cell,merge at this site,not sure"', allow_blank=True); w.add_data_validation(v); v.add(f'L2:L{len(T) + 1}')
h = W.create_sheet('how to')
for t in ('Is this synapse really on cell B? Fill the verdict column (dropdown) on the check sheet; ids match the tree names in check_inputs_wk.nml.',
          'Paste wk into webKnossos with the segmentation in look_in: v4 = graph_cellB_v4_seg_sharded, v3 = graph_cellB_v3_seg_sharded.',
          'Top sites: 5 synapses each in the nml, one row here (the first). inputs_at_site = new inputs from that v3 object. These are the worst cases, not a fair sample.',
          'Random rows: a fair sample of each set, seed 0. Kept is the control, it should come back mostly on cell B.',
          'vast_cell = manual cell at the synapse if any. Coordinates are the synapse centre, 4 x 4 x 30 nm.'): h.append([t])
h.column_dimensions['A'].width = 130
W.save(f'{O}/check_inputs.xlsx'); print(T.groupby('group', sort=False).size().to_dict(), 'trees', len(TREES), 'nodes', nid)
