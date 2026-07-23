#!/bin/bash
# usage: bash run.sh EXP-ID [args...]
#        SCRIPT=debug_main.py GPU=1 bash run.sh EXP-0001 --batch-size 1
set -u
EXP_ID=$1; shift
SCRIPT=${SCRIPT:-main.py}
GPU=${GPU:-1}

RUN_DIR="runs/${EXP_ID}_$(date +%m%d_%H%M)"
mkdir -p "$RUN_DIR"

git rev-parse HEAD  > "$RUN_DIR/commit.txt" 2>/dev/null || echo "no git" > "$RUN_DIR/commit.txt"
git diff HEAD       > "$RUN_DIR/diff.patch" 2>/dev/null
git status --short  > "$RUN_DIR/git_status.txt" 2>/dev/null
pip freeze          > "$RUN_DIR/env.txt"
echo "$SCRIPT $@"   > "$RUN_DIR/cmd.txt"
nvidia-smi          > "$RUN_DIR/gpu.txt"
date                > "$RUN_DIR/started_at.txt"

echo "▶ $RUN_DIR"
CUDA_VISIBLE_DEVICES=$GPU python -u "$SCRIPT" "$@" 2>&1 | tee "$RUN_DIR/train.log"
