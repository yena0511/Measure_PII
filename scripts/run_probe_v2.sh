#!/usr/bin/env bash
# [V2 backup] Current version: scripts/run_probe_v3.sh.
# Run the V2 hypothesis probe for one or more models, with a GPU memory cap and logs.
#
# Usage:  bash scripts/run_probe_v2.sh <model> [<model> ...] [-- <extra run_hypothesis_probe.py args>]
#   <model>: Hub id (DISLab/SummLlama3-8B) or file stem in outputs/model_info_v2 (DISLab__SummLlama3-8B)
# Env:    GPU=0  GPU_LIMIT_GIB=40  PROBE_HF_HOME=/data2/yena/hf_cache  DRY_RUN=1 (print commands only)
#
# Per model: outputs/hypothesis_probe_v2/<model>/ (results), <model>.log (console), <model>.gpu.log (GPU memory).
# The cap is two-layered: PyTorch allocator at GPU_LIMIT_GIB-1 (scripts/gpu_cap.py) and a kill at GPU_LIMIT_GIB
# measured by nvidia-smi, CUDA context included (scripts/gpu_guard.sh). A failed model does not stop the others.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${GPU:-0}
export HF_HOME=${PROBE_HF_HOME:-/data2/yena/hf_cache}  # not ~/.bashrc's HF_HOME: that disk is full
limit=${GPU_LIMIT_GIB:-40}
out=outputs/hypothesis_probe_v2

models=(); extra=()
while (($#)); do
  if [[ $1 == -- ]]; then shift; extra=("$@"); break; fi
  models+=("${1//\//__}"); shift
done
((${#models[@]})) || { sed -n 2,10p "$0"; exit 2; }
mkdir -p "$out"

status=0
for m in "${models[@]}"; do
  info=outputs/model_info_v2/$m.json
  [[ -f $info ]] || { echo "missing $info (run scripts/extract_model_info.py first)"; status=1; continue; }
  cmd=(bash scripts/gpu_guard.sh $((limit * 1024)) "$out/$m.gpu.log" --
       .venv/bin/python scripts/gpu_cap.py $((limit - 1)) scripts/run_hypothesis_probe_v2.py
       --model-info "$info" --output-dir "$out/$m" "${extra[@]}")
  echo "[run_probe_v2] $m on GPU $CUDA_VISIBLE_DEVICES, limit ${limit} GiB, HF_HOME=$HF_HOME"
  if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${cmd[@]}"; echo; continue; fi
  "${cmd[@]}" 2>&1 | tee "$out/$m.log" || { echo "[run_probe_v2] $m failed (see $out/$m.log)"; status=1; }
done
exit $status
