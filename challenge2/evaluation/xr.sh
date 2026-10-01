#!/bin/bash
# usage: xr.sh <out_name> <evaluation.run args...>   (runs detached, log in results/<out_name>.log)
cd ~/c12_bench2
name=$1; shift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=challenge2:hackathon:challenge1 C12_DL_CKPT=$HOME/c12_bench2/ckpt/unet_lofo_u32_all.pt C12_DL_THR=0.5
nohup ~/c12/.venv/bin/python -m evaluation.run "$@" --out results/$name.jsonl > results/$name.log 2>&1 &
echo started $name pid $!
