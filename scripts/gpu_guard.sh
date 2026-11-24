#!/usr/bin/env bash
# Run a command and kill it as soon as its GPU memory (nvidia-smi, all its processes) exceeds a limit.
# Usage: scripts/gpu_guard.sh LIMIT_MIB LOG_FILE -- command args...
limit=$1; log=$2; shift 3
"$@" &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
  pids=$(pgrep -g "$(ps -o pgid= -p $pid | tr -d ' ')" | paste -sd'|')
  used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', ' -v p="^($pids)$" '$1 ~ p {s += $2} END {print s+0}')
  (( used > peak )) && peak=$used
  if (( used > limit )); then
    echo "[gpu_guard] $(date +%T) ${used} MiB > ${limit} MiB: killing $pid" | tee -a "$log"
    kill -9 "$pid"; wait "$pid"; exit 99
  fi
  echo "$(date +%T) ${used}" >> "$log"
  sleep 1
done
wait "$pid"; rc=$?
echo "[gpu_guard] exit $rc, peak ${peak} MiB" | tee -a "$log"
exit $rc
