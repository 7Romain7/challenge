#!/bin/bash
# usage (on a GPU host, from the c12 root): bash detection/lofo_job.sh <name> <train args...>
# Copies the pool to local /tmp once, then trains and scores the run on the robustness suite
# (eval.json in the run dir), detached with nohup. Skips finished runs.
set -e
cd ~/c12
name=$1; shift
mkdir -p /tmp/ra_c12 runs/$name
[ -f /tmp/ra_c12/pool/meta.json ] || { rm -rf /tmp/ra_c12/pool.part; cp -r data/pool /tmp/ra_c12/pool.part && mv /tmp/ra_c12/pool.part /tmp/ra_c12/pool; }
[ -f runs/$name/DONE ] && exit 0
echo "host=$(hostname) start=$(date)" >> runs/$name/host.txt
PY=".venv/bin/python"
nohup bash -c "export PYTHONUNBUFFERED=1; \
  $PY -m detection.train --pool /tmp/ra_c12/pool --out runs/$name $* > runs/$name/stdout.log 2>&1 && \
  $PY -m detection.evaluate --runs runs/$name --eval-dir data/robustness --out runs/$name/robustness >> runs/$name/stdout.log 2>&1 && \
  touch runs/$name/DONE" > /dev/null 2>&1 &
