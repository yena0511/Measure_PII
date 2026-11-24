"""Experiment 9 check: is delta(true) > delta(swap) a memorization signal or a scale effect of base log-prob?

Fits delta = a + b * logp_base on swapped candidates (per format) and reports the residual of true candidates,
plus a comparison within bins of base log-prob.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from exp_pii_logprob_attribution import build_jobs, parse_args  # noqa: E402
OUT = sys.argv[1] if len(sys.argv) > 1 else "pii_logprob"
EXTRA = sys.argv[2:]
from exp_similarity_extractability import load_ground_truth  # noqa: E402

out = ROOT / "outputs/experiments" / OUT / "summllama"
args = parse_args(["--prefixes", str(ROOT / "outputs/experiments/similarity_extractability/summllama/prefixes.jsonl"),
                   "--ground-truth", str(ROOT / "outputs/dataset_audit/news/feedsum/pii_ground_truth.json"),
                   "--output-dir", str(out), *EXTRA])
prefixes = [json.loads(line) for line in args.prefixes.open()]
entries, _ = load_ground_truth(args.ground_truth)
jobs = build_jobs(args, prefixes, entries)
T = json.loads((out / "logp_target.json").read_text())
B = json.loads((out / "logp_base.json").read_text())
res = {}
for fmt in T:
    cand = {}
    for (i, pid, is_true, _), lt, lb in zip(jobs, T[fmt], B[fmt]):
        prev = cand.get((i, pid), (is_true, -1e9, -1e9))
        cand[(i, pid)] = (is_true, max(prev[1], lt), max(prev[2], lb))
    tr = np.array([(v[2], v[1] - v[2]) for v in cand.values() if v[0]])
    sw = np.array([(v[2], v[1] - v[2]) for v in cand.values() if not v[0]])
    b, a = np.polyfit(sw[:, 0], sw[:, 1], 1)
    resid = tr[:, 1] - (a + b * tr[:, 0])
    bins = []
    edges = np.quantile(tr[:, 0], [0, .25, .5, .75, 1])
    for lo, hi in zip(edges, edges[1:]):
        mt = (tr[:, 0] >= lo) & (tr[:, 0] <= hi)
        ms = (sw[:, 0] >= lo) & (sw[:, 0] <= hi)
        bins.append({"base_logp": [float(lo), float(hi)], "n_true": int(mt.sum()), "n_swap": int(ms.sum()),
                     "delta_true": float(tr[mt, 1].mean()), "delta_swap": float(sw[ms, 1].mean()) if ms.any() else None})
    res[fmt] = {"fit_swap": {"slope": float(b), "intercept": float(a)},
                "true_residual_mean": float(resid.mean()), "true_residual_median": float(np.median(resid)),
                "share_positive": float((resid > 0).mean()), "wilcoxon_p": float(wilcoxon(resid).pvalue),
                "bins_by_base_logp": bins}
(out / "matched_analysis.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))

# PII-level test: prefixes of one PII are not independent; average residuals per PII, then test across PII.
from scipy.stats import wilcoxon as _w  # noqa: E402
cluster = {}
for fmt in T:
    cand = {}
    for (i, pid, is_true, _), lt, lb in zip(jobs, T[fmt], B[fmt]):
        prev = cand.get((i, pid), (is_true, -1e9, -1e9))
        cand[(i, pid)] = (is_true, max(prev[1], lt), max(prev[2], lb))
    sw = np.array([(v[2], v[1] - v[2]) for v in cand.values() if not v[0]])
    b, a = np.polyfit(sw[:, 0], sw[:, 1], 1)
    per = {}
    for (i, pid), v in cand.items():
        if v[0]:
            per.setdefault(pid, []).append(v[1] - v[2] - (a + b * v[2]))
    m = np.array([np.mean(x) for x in per.values()])
    by_t = {t: np.array([np.mean(x) for p, x in per.items() if p.startswith(t)]) for t in ("Email", "Phone")}
    cluster[fmt] = {"n_pii": len(m), "mean": float(m.mean()), "median": float(np.median(m)),
                    "share_positive": float((m > 0).mean()), "wilcoxon_p": float(_w(m).pvalue),
                    **{f"{t}_median": float(np.median(v)) for t, v in by_t.items()},
                    **{f"{t}_p": float(_w(v).pvalue) for t, v in by_t.items()}}
res["pii_level"] = cluster
(out / "matched_analysis.json").write_text(json.dumps(res, indent=2))
print("PII-level:", json.dumps(cluster, indent=1))

# Emails split by whether the local part is inferable from a word of the prefix (name -> address).
from exp_context_extraction import email_inferable  # noqa: E402
split = {}
for fmt in T:
    cand = {}
    for (i, pid, is_true, _), lt, lb in zip(jobs, T[fmt], B[fmt]):
        prev = cand.get((i, pid), (is_true, -1e9, -1e9))
        cand[(i, pid)] = (is_true, max(prev[1], lt), max(prev[2], lb))
    sw = np.array([(v[2], v[1] - v[2]) for v in cand.values() if not v[0]])
    b, a = np.polyfit(sw[:, 0], sw[:, 1], 1)
    groups = {"inferable": {}, "not_inferable": {}}
    for (i, pid), v in cand.items():
        if v[0] and pid.startswith("Email"):
            g = "inferable" if email_inferable(pid, prefixes[i]["text"]) else "not_inferable"
            groups[g].setdefault(pid, []).append(v[1] - v[2] - (a + b * v[2]))
    split[fmt] = {}
    for g, per in groups.items():
        m = np.array([np.mean(x) for x in per.values()])
        split[fmt][g] = {"n_pii": len(m), "n_prefixes": sum(map(len, per.values())),
                         "median": float(np.median(m)) if len(m) else None,
                         "wilcoxon_p": float(_w(m).pvalue) if len(m) > 5 else None}
res["email_by_inferability"] = split
(out / "matched_analysis.json").write_text(json.dumps(res, indent=2))
print("Email split:", json.dumps(split, indent=1))
