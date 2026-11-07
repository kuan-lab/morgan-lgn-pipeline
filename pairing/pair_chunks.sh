#!/bin/bash
# Synapse instances + pre/post pairing for cell B chunks, on the merged segmentation (no inference).
#   pair_chunks.sh <chunks.csv dir> <tag> <n parallel> [chunk names...]   (default: every chunk in chunks.csv)
# Large files live in node-local /tmp and are deleted per chunk; tables go to <dir>/chunks/<chunk>/.
set -uo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
export ROOT=/gpfs/radev/project/kuan/ntl5/cell_A_full
export RUN_DIR=$(realpath "$1") TAG=$2; NPAR=$3; shift 3
export BOX_RUN=cellB_pairing BOX_BOXES_CSV=$RUN_DIR/chunks.csv BOX_SKELETON=$RUN_DIR/skeleton_nodes.npy
export BOX_WORK_DIR=/tmp/cellBpair_${SLURM_JOB_ID:-$$}
export PYTHONPATH=$ROOT/magneton_local:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$BOX_WORK_DIR" "$RUN_DIR/chunks" "$RUN_DIR/chunk_logs"
trap 'rm -rf "$BOX_WORK_DIR"' EXIT
one() {
  c=$1; DEST=$RUN_DIR/chunks/$c; D=$BOX_WORK_DIR/$c
  [ -f "$DEST/.done" ] && return 0
  t0=$SECONDS
  (
    set -e
    python -u $ROOT/cellB_connectivity/pair_prep_chunk.py --box "$c" --tag "$TAG"
    cd $ROOT/box_pipeline && python -u step3_seg_and_pair.py --box "$c"
    mkdir -p "$DEST"
    cp "$D"/box.json "$D"/synapses.csv "$D"/boutons.csv "$D"/summary.json "$DEST"/
    [ -f "$D/skel_segs.csv" ] && cp "$D/skel_segs.csv" "$DEST"/
    touch "$DEST/.done"
  ) 2>&1 | grep -vE "Warning|warn\(|google|Python version|upgrade|best-effort|with any other|^\s*$" > "$RUN_DIR/chunk_logs/$c.log"
  [ -f "$DEST/.done" ] || echo "$c" >> "$RUN_DIR/failed.txt"
  echo "$c $([ -f "$DEST/.done" ] && echo ok || echo FAILED) $((SECONDS - t0))s"
  rm -rf "$D"
}
export -f one
if [ $# -gt 0 ]; then printf '%s\n' "$@"; else tail -n +2 "$BOX_BOXES_CSV" | cut -d, -f1; fi | xargs -P "$NPAR" -I{} bash -c 'one {}'
echo "finished: $(ls "$RUN_DIR"/chunks/*/.done 2>/dev/null | wc -l) chunks done"
