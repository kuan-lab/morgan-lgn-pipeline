"""Tiled inference with core+halo: only keep the center of each tile, halo discarded."""
import sys, os, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '/gpfs/radev/home/yf354/magneton/pytorch_connectomics')
import numpy as np, tifffile, torch
from connectomics.utils.system import init_devices
from connectomics.config import load_cfg
from connectomics.model import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--config-base', required=True)
parser.add_argument('--config-file', default='/tmp/empty.yaml')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--em', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--keep-mask', default=None, help='npy (Z, Y/f, X/f): tiles with no kept voxel in their core are skipped')
parser.add_argument('--keep-factor', type=int, default=16)
args = parser.parse_args()
args.inference = True; args.opts = []; args.distributed = False
args.manual_seed = None; args.local_world_size = 1; args.debug = False
if not os.path.exists(args.config_file): open(args.config_file,'w').write('{}\n')

cfg = load_cfg(args)
device = init_devices(args, cfg)
model = build_model(cfg, device, rank=None)
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
(model.module if hasattr(model,'module') else model).load_state_dict(sd)
model.eval()
print(f'loaded ckpt {args.checkpoint}', flush=True)

em = tifffile.imread(args.em)
Z, Y, X = em.shape
tz, ty, tx = cfg.MODEL.INPUT_SIZE   # 32, 256, 256
# HALO — discard outer this many voxels per side of each tile
hz, hy, hx = 4, 32, 32
cz, cy, cx = tz - 2*hz, ty - 2*hy, tx - 2*hx    # 24, 192, 192 core
print(f'INPUT_SIZE={cfg.MODEL.INPUT_SIZE}  HALO=({hz},{hy},{hx})  CORE=({cz},{cy},{cx})', flush=True)

# Pad em by halo on each side (reflect) so edge tiles have full context
em_p = np.pad(em, ((hz, hz),(hy, hy),(hx, hx)), mode='reflect')
Zp, Yp, Xp = em_p.shape
# also pad up so cores tile evenly
def roundup(n, m): return ((n + m - 1) // m) * m
Zt = roundup(Z, cz); Yt = roundup(Y, cy); Xt = roundup(X, cx)
# add extra pad on high end to make Z/Y/X_padded_total = halo + cz*Nz + halo etc.
extra_z, extra_y, extra_x = Zt-Z, Yt-Y, Xt-X
em_p = np.pad(em_p, ((0, extra_z),(0, extra_y),(0, extra_x)), mode='reflect')
print(f'padded em: {em_p.shape}  (original {em.shape}, roundup to {Zt},{Yt},{Xt})', flush=True)

# Output buffer sized to padded-round-up (final crop to original later)
out = np.zeros((3, Zt, Yt, Xt), dtype=np.uint8)
mean, std = 0.5, 0.5
n_tiles = (Zt//cz) * (Yt//cy) * (Xt//cx)
print(f'tiling: {Zt//cz} x {Yt//cy} x {Xt//cx} = {n_tiles} tiles', flush=True)
keep = np.load(args.keep_mask) if args.keep_mask and os.path.exists(args.keep_mask) else None
KF = args.keep_factor
skipped = 0
count = 0
for z0 in range(0, Zt, cz):
    for y0 in range(0, Yt, cy):
        for x0 in range(0, Xt, cx):
            # tile in em_p (which is already halo-padded) starts at (z0, y0, x0)
            if keep is not None and not keep[z0:z0+cz, y0//KF:-(-(y0+cy)//KF), x0//KF:-(-(x0+cx)//KF)].any():
                skipped += 1; continue
            patch = em_p[z0:z0+tz, y0:y0+ty, x0:x0+tx].astype(np.float32) / 255.0
            patch = (patch - mean) / std
            t = torch.from_numpy(patch[None,None]).to(device)
            with torch.no_grad(): o = model(t)
            if isinstance(o,(list,tuple)): o = o[0]
            o = torch.sigmoid(o).cpu().numpy()[0]   # (3, tz, ty, tx)
            # keep only core (discard halo)
            core = o[:, hz:hz+cz, hy:hy+cy, hx:hx+cx]
            out[:, z0:z0+cz, y0:y0+cy, x0:x0+cx] = (core * 255).clip(0,255).astype(np.uint8)
            count += 1
            if count % 100 == 0: print(f'  {count}/{n_tiles}', flush=True)

out = out[:, :Z, :Y, :X]
tifffile.imwrite(args.out, out, compression='zlib')
print(f'wrote {args.out} shape {out.shape}  tiles run {count}, skipped (far from skeleton) {skipped}', flush=True)
