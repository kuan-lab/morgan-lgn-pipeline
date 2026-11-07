#!/usr/bin/env python3
"""Transfer the existing precomputed to a sharded + gzip-compressed pyramid.

Steps:
  1. mip0 transfer: existing unsharded chunks -> sharded gzipped shards
  2. downsample x N: factor (2, 2, 1), mode-aware for segmentation, sharded gzipped
"""

import argparse
import os
import time

os.environ["CLOUDFILES_MONITORING"] = "false"

from igneous.task_creation import (
    create_image_shard_downsample_tasks,
    create_image_shard_transfer_tasks,
)
from taskqueue import LocalTaskQueue


def as_url(path):
    return path if "://" in path else f"file://{os.path.abspath(path)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Existing precomputed (file:// or s3://)")
    ap.add_argument("--dst", required=True, help="Output sharded precomputed dir")
    ap.add_argument("--num-mips", type=int, default=3,
                    help="Downsample levels to add on top of mip0 (default 3 -> 128,256,512 nm xy)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--memory-target", type=int, default=3_500_000_000,
                    help="Per-task memory budget (default 3.5 GB)")
    args = ap.parse_args()

    src = as_url(args.src)
    dst = as_url(args.dst)
    print(f"src: {src}")
    print(f"dst: {dst}")
    print(f"workers: {args.workers}  memory_target: {args.memory_target/1e9:.1f} GB")

    tq = LocalTaskQueue(parallel=args.workers)

    print("\n=== Step 1: Transfer mip0 -> sharded gzip ===")
    t0 = time.time()
    tasks = create_image_shard_transfer_tasks(
        src, dst,
        mip=0,
        fill_missing=True,
        encoding="raw",
        compress=True,
        truncate_scales=True,
        memory_target=args.memory_target,
    )
    tq.insert(tasks)
    tq.execute()
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"\n=== Step 2: Downsample x{args.num_mips} (factor 2x2x1) ===")
    for src_mip in range(args.num_mips):
        t0 = time.time()
        print(f"  mip{src_mip} -> mip{src_mip + 1}")
        tasks = create_image_shard_downsample_tasks(
            dst,
            mip=src_mip,
            sparse=False,   # False for multi-channel image data (e.g. 3-ch aff); True was 1-ch only
            factor=(2, 2, 1),
            fill_missing=True,
            memory_target=args.memory_target,
        )
        tq.insert(tasks)
        tq.execute()
        print(f"    done in {time.time()-t0:.1f}s")

    print("\nAll done.")


if __name__ == "__main__":
    main()
