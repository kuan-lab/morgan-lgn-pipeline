#!/usr/bin/env python3
"""Merge multiple partial cell-B tile plans (from SLURM array tasks) into one."""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True, help="Glob for partial JSONs")
    ap.add_argument("--dst", required=True, help="Output merged JSON")
    args = ap.parse_args()

    parts = sorted(glob.glob(args.in_glob))
    print(f"found {len(parts)} partial plan(s)")
    all_tiles = set()
    meta = None
    for p in parts:
        with open(p) as f:
            plan = json.load(f)
        if meta is None:
            meta = {k: v for k, v in plan.items() if k != 'tiles'}
        for t in plan['tiles']:
            all_tiles.add(tuple(t))
        print(f"  {p}: {len(plan['tiles'])} tiles (cumulative: {len(all_tiles)})")
    tiles_uniq = sorted(all_tiles)
    meta['n_tiles'] = len(tiles_uniq)
    meta['tiles'] = [list(t) for t in tiles_uniq]
    with open(args.dst, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.dst}: {len(tiles_uniq)} unique tiles")


if __name__ == "__main__":
    main()
