#!/bin/bash
# Autonomous end of the night (gelinotte): wait for the meta-acquisition training, then
# dev 0-99 with the final weights, then val 1000-1199 (3 run seeds, paired with bo_roi_dlf),
# then the paired comparison tables. Decisions fixed in advance: checkpoint = final CMA mean.
cd ~/c12_bench2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=challenge2:hackathon:challenge1
export C12_DL_CKPT=$HOME/c12_bench2/ckpt/unet_lofo_u32_all.pt C12_DL_THR=0.5
PY=~/c12/.venv/bin/python
echo "[$(date)] waiting for training"
until [ -f ckpt/meta_acq/w_final.npy ]; do sleep 60; done
export C12_META_W=$HOME/c12_bench2/ckpt/meta_acq/w_final.npy
echo "[$(date)] dev 0-99 with final weights"
$PY -m evaluation.run --methods bo_roi_meta_dlf --split dev --n 100 --workers 15 --out results/metafinal_dev100.jsonl > results/metafinal_dev100.log 2>&1
echo "[$(date)] val 1000-1199, 3 run seeds"
$PY -m evaluation.run --methods bo_roi_meta_dlf bo_roi_dlf --split val --n 200 --run-seeds 3 --workers 15 --out results/night_val200x3.jsonl > results/night_val200x3.log 2>&1
echo "[$(date)] comparisons"
{
  echo "## dev 0-99 (ref bo_roi_dlf)"
  $PY -m evaluation.compare --ref bo_roi_dlf results/diag2_dev100.jsonl results/classic_dev100.jsonl results/roicma_dev100.jsonl results/lit_dev100.jsonl results/meta25_dev100.jsonl results/metafinal_dev100.jsonl 2>&1
  echo
  echo "## val 1000-1199, 3 run seeds averaged per device (ref bo_roi_dlf)"
  $PY -m evaluation.compare --ref bo_roi_dlf results/night_val200x3.jsonl 2>&1
} > results/NIGHT_compare.md
echo "[$(date)] done"
