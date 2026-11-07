"""Single source of truth for box_pipeline settings."""
import csv
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
# Environment overrides let other drivers (rgc_pipeline) reuse these steps with their own box list / folders.
BOXES_CSV = os.environ.get('BOX_BOXES_CSV', os.path.join(ROOT, 'boxes.csv'))
# Each run gets its own output folder so changing box size never overwrites earlier results.
# 'boxes' = first run (2500x2500x100, no mito). 'boxes_z300' = 300 sections + mito, blank sections NOT handled
# (models hallucinate on black frames -> merges). 'boxes_z300_noblank' = blank sections removed before inference.
RUN = os.environ.get('BOX_RUN', 'boxes_z300_noblank')
BOX_DIR = os.environ.get('BOX_WORK_DIR', os.path.join(ROOT, RUN))
ANALYSIS_DIR = os.path.join(ROOT, 'analysis', RUN)

# Box size in wk voxels (4x4x30 nm). XY must be even (halved to 8 nm for inference).
BOX_WK = (2500, 2500, 300)   # ~10 x 10 x 9 um

# Blank (black) EM sections: removed before inference, put back afterwards by copying the neighbour's labels.
BLANK_ZERO_FRAC = 0.5    # section is blank if more than this fraction of pixels is exactly 0 ...
BLANK_MEAN = 40          # ... or its mean intensity is below this
LONG_GAP = 3             # runs of >= this many blank sections are not bridged: segmentation is cut mid-gap

# Optional skeleton (npy of wk x,y,z nodes, env BOX_SKELETON): inference is skipped farther than this from it.
SKELETON_NPY = os.environ.get('BOX_SKELETON')
SKEL_MARGIN_NM = 3000
KEEP_XY_FACTOR = 16      # keep-mask is stored at 1/16 resolution in x,y (full resolution in z)

# EM source (BossDB). wk XY // 2 == this volume's 8x8x30 mip coords.
EM_URL = 's3://bossdb-open-data/morgan2020/lgn/em'
INF_RES = (8, 8, 30)

# Neuron affinity model (v4 finetune)
NEURON_CFG = '/gpfs/radev/project/kuan/ntl5/cell_A_full/finetune_morgan_chunk1_v4_no_myelin/config.yaml'
NEURON_CKPT = '/gpfs/radev/project/kuan/ntl5/cell_A_full/finetune_morgan_chunk1_v4_no_myelin/checkpoint_08000.pth.tar'

# Synapse model (jj837 v6ft)
SYN_CFG = '/gpfs/radev/project/kuan/jj837/synapse_model/training/checkpoints_v6ft/config.yaml'
SYN_CKPT = '/gpfs/radev/project/kuan/jj837/synapse_model/training/checkpoints_v6ft/checkpoint_25000.pth.tar'

# Mito model (jj837 mito v3, checkpoint 80000). Local copy: the original moved on 2026-10-01 to
# /gpfs/radev/project/kuan/jj837/mito_model/training/checkpoints_mito_v3/ and broke a run.
MITO_CFG = '/gpfs/radev/project/kuan/ntl5/cell_A_full/models/mito_v3_jj837/config.yaml'
MITO_CKPT = '/gpfs/radev/project/kuan/ntl5/cell_A_full/models/mito_v3_jj837/checkpoint_80000.pth.tar'

# Waterz (2D supervoxels, same scorer as production)
WATERZ_THRESHOLD = 0.2
WATERZ_BLOCK_PY = '/gpfs/radev/home/yf354/magneton/instance_segmentation/waterz_block.py'

# Synapse instances: binary watershed, tuned 2026-09-28 (F1 0.973 on Jaeyee GT)
SYN_SEED_THR = 0.9
SYN_FG_THR = 0.2
SYN_SEED_MIN = int(os.environ.get('BOX_SYN_SEED_MIN', 50))      # env overrides for threshold tests
SYN_MIN_SEG = int(os.environ.get('BOX_SYN_MIN_SEG', 300))
# keep watershed regions as separate synapses (fuse only touching regions with the same pre and post); 0 = old behaviour
SYN_KEEP_REGIONS = os.environ.get('BOX_SYN_KEEP_REGIONS', '1') != '0'

# Pre/post attribution
MIN_PARTNER_SEG_VOX = 1000      # segs smaller than this can't be pre/post
LONG_RAY_NM = 180               # rays longer than this are flagged (FP / truncation)
FALLBACK_RAY_NM = 150           # ray longer than this -> use 2nd-most-overlapping seg as pre
BOUTON_NECK_NM = 100            # EDT threshold that splits a pre seg into bouton cores


def load_boxes():
    with open(BOXES_CSV) as f:
        return {r['name']: r for r in csv.DictReader(f)}


def box_geometry(row):
    """wk origin (4x4x30) and 8x8x30 inference origin/size for a box row."""
    if row.get('wk_ox') not in (None, ''):   # explicit corner + size (rgc_pipeline chunks)
        ox, oy, oz = int(row['wk_ox']), int(row['wk_oy']), int(row['wk_oz'])
        sx, sy, sz = int(row['wk_sx']), int(row['wk_sy']), int(row['wk_sz'])
    else:
        cx, cy, cz = int(row['wk_x']), int(row['wk_y']), int(row['wk_z'])
        sx, sy, sz = BOX_WK
        ox, oy, oz = cx - sx // 2, cy - sy // 2, cz - sz // 2
        ox -= ox % 2
        oy -= oy % 2
    return dict(wk_origin=(ox, oy, oz), wk_size=(sx, sy, sz),
                inf_origin=(ox // 2, oy // 2, oz), inf_size=(sx // 2, sy // 2, sz))


def box_path(name, fname=''):
    d = os.path.join(BOX_DIR, name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)
