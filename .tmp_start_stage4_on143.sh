#!/usr/bin/env bash
set -euo pipefail

cd /home/lqma/npux-mlir
mkdir -p logs

if pgrep -af 'bootstrap_env.sh --host-only' >/dev/null 2>&1; then
  echo "BOOTSTRAP_ALREADY_RUNNING"
  pgrep -af 'bootstrap_env.sh --host-only'
  exit 0
fi

nohup bash -lc 'source /home/lqma/anaconda3/etc/profile.d/conda.sh; ./scripts/bootstrap_env.sh --host-only --jobs 16' \
  > logs/stage4_bootstrap_host_on143.log 2>&1 &

echo "BOOTSTRAP143_PID=$!"
