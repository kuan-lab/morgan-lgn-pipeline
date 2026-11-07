#!/usr/bin/env python3
"""Tiled synapse inference on a local EM TIF (Z,Y,X uint8).
Uses jj's v6ft model. Tiles at model input size, pads on each side, writes one
big pred TIF matching input shape.
"""
import argparse, os, time
import numpy as np, torch, tifffile


def load_model(cfg_path, checkpoint, device):
    from connectomics.config import get_cfg_defaults
    from connectomics.model import build_model, update_state_dict
    cfg = get_cfg_defaults(); cfg.merge_from_file(cfg_path)
    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    state = update_state_dict(cfg, state, mode='test')
    inner = model.module if hasattr(model, 'module') else model
    stripped = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in state.items()}
    inner.load_state_dict(stripped, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--em-tif', required=True)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out-tif', required=True)
    ap.add_argument('--tile', default='128,128,16', help='Valid tile size (X,Y,Z)')
    ap.add_argument('--pad',  default='64,64,8',   help='Model pad each side (X,Y,Z)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--keep-mask', default=None, help='npy (Z, Y/f, X/f): tiles with no kept voxel are skipped')
    ap.add_argument('--keep-factor', type=int, default=16)
    args = ap.parse_args()

    em = tifffile.imread(args.em_tif)  # (Z,Y,X)
    Z_orig, Y_orig, X_orig = em.shape
    tx, ty, tz = [int(v) for v in args.tile.split(',')]
    px, py, pz = [int(v) for v in args.pad.split(',')]
    # Pad EM to next multiple of tile size so tiling divides cleanly; crop back at end
    def ceil_mult(a, m): return ((a + m - 1) // m) * m
    Zp, Yp, Xp = ceil_mult(Z_orig, tz), ceil_mult(Y_orig, ty), ceil_mult(X_orig, tx)
    em = np.pad(em, ((0, Zp - Z_orig), (0, Yp - Y_orig), (0, Xp - X_orig)), mode='reflect')
    Z, Y, X = em.shape
    print(f'orig EM {Z_orig, Y_orig, X_orig} -> padded to {em.shape} for clean tiling')
    # pad the whole em volume on all sides for pad-inclusive reads
    em_pad = np.pad(em, ((pz, pz), (py, py), (px, px)), mode='reflect')
    pred = np.zeros((Z, Y, X), dtype=np.uint8)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.cfg, args.checkpoint, device)

    nz, ny, nx = Z // tz, Y // ty, X // tx
    print(f'em {em.shape}, tiling {nx*ny*nz} = {nx}x{ny}x{nz} tiles ({tx},{ty},{tz} + pad {px},{py},{pz})')
    keep = np.load(args.keep_mask) if args.keep_mask and os.path.exists(args.keep_mask) else None
    KF = args.keep_factor; skipped = 0
    t0 = time.time()
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # valid region in output pred
                z0, z1 = iz*tz, iz*tz+tz
                y0, y1 = iy*ty, iy*ty+ty
                x0, x1 = ix*tx, ix*tx+tx
                if keep is not None and not keep[z0:z1, y0//KF:-(-y1//KF), x0//KF:-(-x1//KF)].any():
                    skipped += 1; continue
                # pad-inclusive read window in em_pad
                zr0, zr1 = z0, z1 + 2*pz
                yr0, yr1 = y0, y1 + 2*py
                xr0, xr1 = x0, x1 + 2*px
                em_win = em_pad[zr0:zr1, yr0:yr1, xr0:xr1]  # (Z,Y,X)
                xt = em_win.astype(np.float32) / 255.0
                xt = (xt - 0.5) / 0.5
                xt = torch.from_numpy(xt).to(device)[None, None]
                with torch.no_grad():
                    y = torch.sigmoid(model(xt))
                y_np = y[0, 0].cpu().numpy()  # (Z,Y,X) padded
                pred[z0:z1, y0:y1, x0:x1] = (y_np[pz:pz+tz, py:py+ty, px:px+tx] * 255).astype(np.uint8)
    print(f'tiled inference: {time.time()-t0:.1f}s  tiles skipped (far from skeleton): {skipped} of {nx*ny*nz}')
    pred = pred[:Z_orig, :Y_orig, :X_orig]   # crop back to original EM extent
    tifffile.imwrite(args.out_tif, pred, compression='zlib')
    print(f'wrote {args.out_tif} shape={pred.shape} nonzero={int((pred>0).sum())}')


if __name__ == '__main__':
    main()
