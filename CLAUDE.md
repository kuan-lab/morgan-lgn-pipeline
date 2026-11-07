# morgan-lgn-pipeline

Snapshot of the scripts behind the cell B (VAST "cell 201 tcr") connectivity results in the Morgan & Lichtman mouse LGN
ssTEM volume (BossDB `morgan2020/lgn/em`). Stages and dependencies are in `README.md`. The scripts keep their absolute
Misha paths: they live and were run in `/gpfs/radev/project/kuan/ntl5/cell_A_full/` (Nghia's project), with large volumes
on Marilyn `/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/`. To run something, copy the script and change its paths.
Don't write into Nghia's folders.

The skills in `.claude/skills/` (`tile-inference`, `graph-seg`, `pairing-analysis`, `fiji-crop`, `wk-check-files`) describe
each workflow. Their paths point at the original locations. magneton's own skills (`hpc`, `check`, `config`, ...) live in
the kuan-lab/magneton repo; open Claude in a magneton clone to use them.

## Coordinates

- **wk** = webKnossos, 4 × 4 × 30 nm. Pipeline voxels are 8 × 8 × 30: absolute x, y = wk / 2, z the same.
- Cell B intermediates (EM, affinity) are stored local: absolute − (25114, 33058, 134). Segmentations and anything shown
  in wk carry the absolute `voxel_offset`. Set it when creating the volume, never after sharding.
- Affinity channels: 0 = z, 1 = y, 2 = x; `aff_z[z]` is the link between z−1 and z.

## Rules

- **Never run dense inference on a whole cell**: it filled 1.4 TB of shared disk. Use the tile plan
  (`blob_infer_from_plan*.py`), with 3 µm dilation from the skeleton.
- Ask before any GPU job or whole-cell run, with an estimate of GPU-hours and disk.
- Never delete large intermediates without asking. Before calling a derived volume good, decode real voxels from it.
- Remove blank sections before inference. Some are white or noise, not black: detect them by contrast (`empty_frames.py`).
- Myelin in graph segmentation: no setting is clean (see the README table); v3 is the reference.
- A synapse whose pre-side ray is longer than 180 nm is almost always a false positive or a cut: flag and exclude it.
- Save every analysis result to disk (CSV + the script + a chart), and give the full path of every file you make.

## Cluster (Misha)

- The login node kills Python that loads volumes: use `srun -p devel` or sbatch.
- Slurm counts requested memory: check `scontrol show node <n> | grep AllocMem`. Graph edges need ~20 GB per worker on
  dense cores; the global merge needs ~1.3 TB (bigmem).
- Env: `/gpfs/radev/home/yf354/.conda/envs/magneton`. In sbatch, source `conda.sh` directly, and put `HOME` on node
  scratch, because CloudVolume lock files fill the home inode quota.
