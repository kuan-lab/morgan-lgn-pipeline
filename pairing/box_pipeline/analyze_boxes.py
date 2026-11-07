"""Aggregate all boxes into per-box and per-depth tables + a small-multiples chart.

Outputs (box_pipeline/analysis/): per_box.csv, per_depth.csv, all_boutons.csv, all_synapses.csv, by_depth.png
"""
import json
import os

import numpy as np
import pandas as pd
import tifffile

import config as C

OUT = C.ANALYSIS_DIR
os.makedirs(OUT, exist_ok=True)
VOX_UM3 = 8 * 8 * 30 / 1e9

per_box, all_b, all_s = [], [], []
for name, row in C.load_boxes().items():
    d = os.path.join(C.BOX_DIR, name)
    if not os.path.exists(f'{d}/summary.json'):
        continue
    s = pd.read_csv(f'{d}/synapses.csv')
    b = pd.read_csv(f'{d}/boutons.csv')
    em = tifffile.imread(f'{d}/em.tif')
    tissue_frac = float((em > 0).mean())
    vol = np.prod(C.BOX_WK) * 4 * 4 * 30 / 1e9 * tissue_frac
    s_ok = s[s.quality == 'ok']
    # mito: fraction of each bouton's voxels (and of its whole pre segment) with mito prob > 0.5
    has_mito = os.path.exists(f'{d}/mito_prob.tif') and len(b)
    if has_mito:
        mito = tifffile.imread(f'{d}/mito_prob.tif') > 127
        bl = tifffile.imread(f'{d}/bouton_labels.tif')
        nb = int(bl.max()) + 1
        mv = np.bincount(bl[mito], minlength=nb)
        b['mito_vox'] = mv[b.bouton_id.values]
        b['mito_frac'] = (b.mito_vox / b.vol_vox).round(4)
        del bl
        seg = tifffile.imread(f'{d}/seg.tif')
        ns = int(seg.max()) + 1
        sv, sm = np.bincount(seg.ravel(), minlength=ns), np.bincount(seg[mito], minlength=ns)
        b['pre_seg_vox'] = sv[b.pre_seg_id.values]
        b['pre_seg_mito_frac'] = (sm[b.pre_seg_id.values] / np.maximum(sv[b.pre_seg_id.values], 1)).round(4)
        mito_tissue_frac = float(mito.mean())
        del seg, mito
        b.to_csv(f'{d}/boutons_mito.csv', index=False)
    bc = b[b.truncated_at_box == 0]
    s['box'] = name; s['depth_um'] = int(row['depth_um'])
    b['box'] = name; b['depth_um'] = int(row['depth_um'])
    all_s.append(s); all_b.append(b)
    per_box.append(dict(
        box=name, depth_um=int(row['depth_um']), tissue_frac=round(tissue_frac, 3),
        tissue_um3=round(vol, 1),
        n_syn=len(s), n_syn_ok=len(s_ok),
        syn_density_ok_per_um3=round(len(s_ok) / vol, 3),
        frac_flagged=round(1 - len(s_ok) / max(len(s), 1), 3),
        median_syn_um3=round(float(s_ok.n_vox.median() * VOX_UM3), 4),
        n_boutons=len(b), n_boutons_complete=len(bc),
        bouton_density_per_um3=round(len(b) / vol, 3),
        median_bouton_um3=round(float(bc.vol_um3.median()), 4),
        p90_bouton_um3=round(float(bc.vol_um3.quantile(0.9)), 4),
        mean_syn_per_bouton=round(float(bc.syn_count.mean()), 3),
        multisyn_frac=round(float((bc.syn_count >= 2).mean()), 3),
        frac_boutons_ge_0p5um3=round(float((bc.vol_um3 >= 0.5).mean()), 3),
        mito_frac_of_tissue=round(mito_tissue_frac, 4) if has_mito else None,
        frac_boutons_with_mito=round(float((bc.mito_vox > 0).mean()), 3) if has_mito else None,
        median_bouton_mito_frac=round(float(bc.mito_frac.median()), 4) if has_mito else None,
    ))

pb = pd.DataFrame(per_box).sort_values(['depth_um', 'box'])
pb.to_csv(f'{OUT}/per_box.csv', index=False)
pd.concat(all_b).to_csv(f'{OUT}/all_boutons.csv', index=False)
pd.concat(all_s).to_csv(f'{OUT}/all_synapses.csv', index=False)

metrics = [c for c in ['frac_boutons_with_mito', 'median_bouton_mito_frac'] if pb[c].notna().all()] + ['syn_density_ok_per_um3', 'bouton_density_per_um3', 'median_bouton_um3',
           'p90_bouton_um3', 'multisyn_frac', 'mean_syn_per_bouton', 'median_syn_um3', 'frac_flagged']
pd_ = pb.groupby('depth_um')[metrics].agg(['mean', 'std']).round(4)
pd_.to_csv(f'{OUT}/per_depth.csv')

# pooled complete-bouton stats per depth (more boutons -> stabler than mean of box medians)
ab = pd.concat(all_b)
abc = ab[ab.truncated_at_box == 0]
pooled = abc.groupby('depth_um').agg(n_complete=('bouton_id', 'size'),
                                     median_um3=('vol_um3', 'median'),
                                     p90_um3=('vol_um3', lambda x: x.quantile(.9)),
                                     frac_ge_0p5um3=('vol_um3', lambda x: (x >= 0.5).mean()),
                                     frac_ge_1um3=('vol_um3', lambda x: (x >= 1.0).mean()),
                                     multisyn_frac=('syn_count', lambda x: (x >= 2).mean())).round(4)
if 'mito_frac' in abc:
    big = abc[abc.vol_um3 >= 0.5]
    print('\nmito by bouton size (complete boutons):')
    print(abc.assign(size=pd.cut(abc.vol_um3, [0, .1, .25, .5, 1, 100])).groupby('size', observed=True).agg(
        n=('bouton_id', 'size'), with_mito=('mito_vox', lambda x: (x > 0).mean()),
        median_mito_frac=('mito_frac', 'median'), multisyn=('syn_count', lambda x: (x >= 2).mean())).round(3).to_string())
    cand = big.sort_values(['syn_count', 'vol_um3'], ascending=False)
    cand[['box', 'depth_um', 'bouton_id', 'pre_seg_id', 'vol_um3', 'syn_count', 'mito_frac', 'pre_seg_mito_frac',
          'wk_x', 'wk_y', 'wk_z']].to_csv(f'{OUT}/large_bouton_candidates.csv', index=False)
    print('\nlarge complete boutons (>=0.5 um3) per depth:')
    print(big.groupby('depth_um').agg(n=('bouton_id', 'size'), median_um3=('vol_um3', 'median'),
          median_mito_frac=('mito_frac', 'median'), multisyn=('syn_count', lambda x: (x >= 2).mean())).round(3).to_string())
pooled.to_csv(f'{OUT}/per_depth_pooled_boutons.csv')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, MARK = '#1f2328', '#6e7781', '#2f6fb3'
titles = {'syn_density_ok_per_um3': 'Synapse density (/µm³)',
          'bouton_density_per_um3': 'Bouton density (/µm³)',
          'median_bouton_um3': 'Median bouton vol (µm³)',
          'multisyn_frac': 'Multisynaptic fraction',
          'mean_syn_per_bouton': 'Synapses per bouton',
          'frac_flagged': 'Synapses flagged'}
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
depths = sorted(pb.depth_um.unique())
for ax, m in zip(axes.flat, titles):
    for dep in depths:
        v = pb[pb.depth_um == dep][m].values
        ax.scatter(np.full(len(v), dep), v, s=40, color=MARK, edgecolor='white', linewidth=1.5, zorder=3)
        ax.hlines(v.mean(), dep - 12, dep + 12, color=INK, linewidth=2, zorder=4)
    ax.set_title(titles[m], fontsize=10, color=INK, loc='left')
    ax.set_xticks(depths)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylim(bottom=0)
for ax in axes[1]:
    ax.set_xlabel('Depth below myelin (µm)', color=MUTED, fontsize=9)
fig.suptitle('Per-box stats by depth (dots = boxes, bar = mean; complete boutons only for size/multisyn)',
             fontsize=10, color=MUTED, x=0.01, ha='left')
fig.tight_layout()
fig.savefig(f'{OUT}/by_depth.png', dpi=150)

print(pb.to_string(index=False))
print('\n', pooled.to_string())
