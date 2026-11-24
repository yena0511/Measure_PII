"""Strict view of Experiment 5: drop own-leaks inferable from a name in the context; leak rate by PII frequency."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from exp_context_extraction import email_inferable  # noqa: E402

out = ROOT / "outputs/experiments/similarity_ladder/summllama"
rows = [json.loads(line) for line in (out / "results.jsonl").open()]
gt = json.loads((ROOT / "outputs/dataset_audit/news/feedsum/pii_ground_truth.json").read_text())
freq = {f"{e['type']}:{e['key']}": len(e["instance_ids"]) for e in gt["pii"]}
res = {"by_variant": {}, "by_type": {}, "by_frequency": {}}
for v in sorted({r["variant"] for r in rows}):
    g = [r for r in rows if r["exact_leaks"] and r["variant"] == v]
    own = [r for r in g if r["own_leak_greedy"]]
    strict = [r for r in own if not email_inferable(r["pii_id"], r["text"])]
    res["by_variant"][v] = {"n": len(g), "own_greedy": len(own) / len(g), "own_greedy_strict": len(strict) / len(g)}
    for t in ("Email", "Phone"):
        gt_ = [r for r in g if r["pii_id"].startswith(t)]
        res["by_type"][f"{v}/{t}"] = {"n": len(gt_), "own_greedy": float(np.mean([r["own_leak_greedy"] for r in gt_]))
                                      if gt_ else None}
bins = defaultdict(list)
for r in rows:
    if r["exact_leaks"] and r["variant"] not in ("exact",):
        f = freq.get(r["pii_id"], 1)
        bins["1" if f == 1 else "2-4" if f < 5 else "5-19" if f < 20 else "20+"].append(r["own_leak_greedy"])
res["by_frequency"] = {k: {"n": len(v), "own_greedy": float(np.mean(v))} for k, v in sorted(bins.items())}
(out / "strict_analysis.json").write_text(json.dumps(res, indent=2))
for k, v in res["by_variant"].items():
    print(f"{k:16s} own={v['own_greedy']:.3f} strict={v['own_greedy_strict']:.3f}")
print(json.dumps(res["by_frequency"]))
print({k: round(v["own_greedy"], 3) for k, v in res["by_type"].items() if v["own_greedy"] is not None and k.split('/')[0] in ('exact', 'cue_only', 'para_full', 'new_topic', 'noise_body_0.3')})
