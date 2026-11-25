"""Run a Python script with a hard PyTorch GPU memory cap.

Usage: python gpu_cap.py LIMIT_GIB script.py [args...]
PyTorch raises CUDA OOM instead of allocating past LIMIT_GIB on each visible GPU.
(The CUDA context, ~0.5 GiB, is not counted by this cap.)
"""
import runpy
import sys

import torch

limit_gib = float(sys.argv[1])
for i in range(torch.cuda.device_count()):
    total_gib = torch.cuda.get_device_properties(i).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit_gib / total_gib), i)
print(f"[gpu_cap] PyTorch allocator capped at {limit_gib} GiB per GPU", file=sys.stderr)

sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
