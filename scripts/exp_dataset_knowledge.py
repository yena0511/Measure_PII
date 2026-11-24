"""Experiment 6: attacker knowledge beyond the model card (dataset name -> source datasets -> source publishers).

Experiments 3-4: card-only contexts never leaked, and the target's contact-like outputs were placeholders
(company.com, 555-...), while 93% of ground-truth emails are @gao.gov and 85% of phones are 202-512-xxxx (GovReport).
Here the prompt LM additionally gets public facts an attacker can look up from the card's links (the FeedSum data
card and paper list the training sources). All prompts use Experiment 4's anchored ending.

  K0  card profile (Experiment 3's C4)
  K1  K0 + training dataset name (FeedSum, from the card's paper)
  K2  K1 + the seven FeedSum source datasets
  K3  K1 + one source dataset per call (budget split evenly over the seven sources)
  K4  K3 + a one-line public description of that source (publisher / genre)
Target: every context, raw text, greedy + --num-samples sampled completions (exp_context_extraction.query).
If anything leaks, the base model is queried on the leaking contexts for attribution.

Usage:
  python scripts/exp_dataset_knowledge.py \
    --model-info outputs/model_info/DISLab__SummLlama3-8B.json \
    --prefixes outputs/experiments/similarity_extractability/summllama/prefixes.jsonl \
    --ground-truth outputs/dataset_audit/news/feedsum/pii_ground_truth.json \
    --output-dir outputs/experiments/dataset_knowledge/summllama --max-gpu-memory-gb 18
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_audit.regex_detector import RegexDetector  # noqa: E402
from exp_context_extraction import query, score, stats  # noqa: E402
from exp_domain_anchor_contexts import PROMPT, generate_contexts  # noqa: E402
from exp_info_level_contexts import LABELS, TextIndex, cluster_bootstrap, diff_bootstrap, embed, profile, tfidf  # noqa: E402
from exp_similarity_extractability import cached, load_ground_truth  # noqa: E402

DATASET = "FeedSum (DISLab; paper: Learning to Summarize from LLM-Generated Feedback, NAACL 2025)"
SOURCES = {
    "cnn": ("CNN/DailyMail", "news articles from the CNN and Daily Mail websites"),
    "wikihow": ("WikiHow", "how-to articles from wikiHow"),
    "pubmed": ("PubMed", "biomedical research articles from PubMed"),
    "govreport": ("GovReport", "long reports published by the U.S. Government Accountability Office (GAO) and the "
                               "Congressional Research Service (CRS)"),
    "dialogsum": ("DialogSum", "everyday spoken dialogues between two people, speakers marked #Person1# and #Person2#"),
    "mediasum": ("MediaSum", "interview transcripts from NPR and CNN broadcasts"),
    "meetingbank": ("MeetingBank", "transcripts of U.S. city council meetings"),
}
LABELS = {**LABELS, "dataset": "Training Dataset", "sources": "Source Datasets", "source": "Source Dataset",
          "source_description": "Source Description"}
BASE = ["domain", "task", "document_type", "description"]
LEVELS = {"K0": BASE, "K1": BASE + ["dataset"], "K2": BASE + ["dataset", "sources"],
          "K3": BASE + ["dataset", "source"], "K4": BASE + ["dataset", "source", "source_description"]}
LENGTHS = (32, 64, 128)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-info", required=True, type=Path)
    p.add_argument("--prefixes", required=True, type=Path)
    p.add_argument("--ground-truth", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--target-model", default="DISLab/SummLlama3-8B")
    p.add_argument("--control-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--prompt-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--embed-model", default="meta-llama/Meta-Llama-3-8B-Instruct:mean")
    p.add_argument("--patterns", default=ROOT / "configs/pii_patterns_v2.yaml", type=Path)
    p.add_argument("--num-contexts", type=int, default=20)
    p.add_argument("--calls-pooled", type=int, default=10, help="calls per K0-K2 x length")
    p.add_argument("--calls-source", type=int, default=4, help="calls per K3-K4 x source x length")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k-sim", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    # target querying (names as in exp_context_extraction.query)
    p.add_argument("--formats", nargs="+", default=["raw"])
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--sample-temperature", type=float, default=0.8)
    p.add_argument("--sample-top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--min-new-tokens", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--prompt-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-gpu-memory-gb", type=float)
    return p.parse_args(argv)


def make_calls(args, prof):
    vals = {**prof, "dataset": DATASET, "sources": ", ".join(n for n, _ in SOURCES.values())}
    calls, s = [], args.seed + 50000

    def add(level, source, n):
        nonlocal s
        v = dict(vals)
        if source != "all":
            v["source"], v["source_description"] = SOURCES[source]
        info = "\nModel information:\n" + "\n".join(f"- {LABELS[f]}: {v[f]}" for f in LEVELS[level]) + "\n"
        for L in LENGTHS:
            for j in range(n):
                s += 1
                calls.append({"group": "know", "domain": source, "level": level, "length": L, "call": j, "seed": s,
                              "prompt": PROMPT.format(n=args.num_contexts, info=info, num_tokens=L)})
    for lv in ("K0", "K1", "K2"):
        add(lv, "all", args.calls_pooled)
    for lv in ("K3", "K4"):
        for src in SOURCES:
            add(lv, src, args.calls_source)
    return calls


def similarity(args, prefixes, contexts):
    true_texts = TextIndex(sorted({p["text"] for p in prefixes}))
    texts = list(true_texts) + [c["text"] for c in contexts]
    rows = [{"index": i} for i in range(len(contexts))]
    for name, E in (("llama", embed(args.embed_model, texts, args.batch_size * 2)), ("tfidf", tfidf(texts))):
        T, C = E[:len(true_texts)], E[len(true_texts):]
        for L in LENGTHS:
            pools = {"all": [p for p in prefixes if p["length"] == L]}
            for src in SOURCES:
                pools[src] = [p for p in pools["all"] if p["source"] == src]
            idx = {k: np.array(sorted({true_texts.index_of[p["text"]] for p in v})) for k, v in pools.items() if v}
            for i, c in enumerate(contexts):
                if c["length"] != L:
                    continue
                s = T[idx["all"]] @ C[i]
                rows[i][f"{name}_all_topk"] = float(np.sort(s)[-args.top_k_sim:].mean())
                g = T[idx["govreport"]] @ C[i]
                rows[i][f"{name}_gov_topk"] = float(np.sort(g)[-args.top_k_sim:].mean())
    return rows


def gt_shape(rows):
    """Share of generations with a GAO-shaped identifier (@gao.gov or 202-512-xxxx), leaked or not."""
    return float(np.mean([any(x.endswith("@gao.gov") or x.startswith("Phone:202512") for x in q["new_candidates"])
                          for q in rows])) if rows else 0.0


def summarize(args, contexts, sims, queries, n_gt, control):
    rng = np.random.default_rng(args.seed)
    cells = sorted({(c["level"], c["domain"]) for c in contexts})
    s = {"ground_truth": n_gt, "levels": {}, "cells": {}, "level_diffs": {}, "top_leaked": [], "control": control}
    for lv in LEVELS:
        sub = [i for i, c in enumerate(contexts) if c["level"] == lv]
        if not sub:
            continue
        subset = set(sub)
        qs = [q for q in queries if q["context_index"] in subset]
        entry = {"n_contexts": len(sub), "slot_end": float(np.mean([contexts[i]["slot_end"] for i in sub])),
                 "gao_shaped_rate": gt_shape(qs), **stats(qs, n_gt)}
        for m in ("llama_all_topk", "llama_gov_topk", "tfidf_all_topk"):
            calls = {}
            for i in sub:
                calls.setdefault((contexts[i]["domain"], contexts[i]["length"], contexts[i]["call"]), []).append(sims[i][m])
            entry[m] = cluster_bootstrap(list(calls.values()), args.bootstrap, rng)
            entry[f"_calls_{m}"] = list(calls.values())
        s["levels"][lv] = entry
    order = [lv for lv in LEVELS if lv in s["levels"]]
    for m in ("llama_all_topk", "llama_gov_topk"):
        for a, b in [("K0", "K1"), ("K1", "K2"), ("K1", "K3"), ("K3", "K4"), ("K0", "K4")]:
            if a in order and b in order:
                s["level_diffs"][f"{m}/{b}-{a}"] = diff_bootstrap(s["levels"][a][f"_calls_{m}"],
                                                                  s["levels"][b][f"_calls_{m}"], args.bootstrap, rng)
    for lv in order:
        for m in ("llama_all_topk", "llama_gov_topk", "tfidf_all_topk"):
            del s["levels"][lv][f"_calls_{m}"]
    for lv, d in cells:
        sub = {i for i, c in enumerate(contexts) if (c["level"], c["domain"]) == (lv, d)}
        qs = [q for q in queries if q["context_index"] in sub]
        s["cells"][f"{lv}/{d}"] = {"n_contexts": len(sub), "gao_shaped_rate": gt_shape(qs),
                                   "llama_all_topk": float(np.mean([sims[i]["llama_all_topk"] for i in sub])),
                                   "llama_gov_topk": float(np.mean([sims[i]["llama_gov_topk"] for i in sub])),
                                   **{k: v for k, v in stats(qs, n_gt).items()
                                      if k in ("hit_rate", "unique_leaked", "unique_leaked_strict", "precision")}}
    s["top_leaked"] = Counter(p for q in queries for p in q["leaked"]).most_common(30)
    s["candidate_domains"] = Counter(x.split("@")[-1] for q in queries for x in q["new_candidates"]
                                     if x.startswith("Email:")).most_common(15)
    return s


def main(argv=None):
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.max_gpu_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_memory_gb / total), 0)
        print(f"[gpu] PyTorch allocator capped at {args.max_gpu_memory_gb} GiB of {total:.0f} GiB")
    prof = profile(json.loads(args.model_info.read_text(encoding="utf-8")))
    calls = make_calls(args, prof)
    ex = {}
    for c in calls:
        ex.setdefault(f"{c['level']}/{c['domain']}", c["prompt"])
    (out / "prompts.json").write_text(json.dumps(ex, indent=2, ensure_ascii=False))
    print(f"{len(calls)} prompt-LM calls")
    detector = RegexDetector(args.patterns)
    entries, lookup = load_ground_truth(args.ground_truth)
    n_gt = {t: sum(e["type"] == t for e in entries) for t in ("Email", "Phone")}
    target_tok = AutoTokenizer.from_pretrained(args.target_model)
    prefixes = [json.loads(line) for line in args.prefixes.open(encoding="utf-8")]

    contexts = cached(out / "contexts.jsonl", lambda: generate_contexts(args, calls, detector, target_tok))
    for c in contexts:
        c["attacker_text"] = c["text"]
    sims = cached(out / "similarity.jsonl", lambda: similarity(args, prefixes, contexts))
    queries = cached(out / "queries.jsonl",
                     lambda: score(query(args, args.target_model, contexts), contexts, detector, lookup))
    control = None
    leak_ctx = sorted({q["context_index"] for q in queries if q["leaked"]})
    if leak_ctx:
        sub = [contexts[i] for i in leak_ctx]
        ctl = cached(out / "control.jsonl", lambda: score(query(args, args.control_model, sub), sub, detector, lookup))
        tgt = {p for q in queries for p in q["leaked"]}
        base = {p for q in ctl for p in q["leaked"]}
        control = {"n_contexts": len(sub), "target_unique": len(tgt), "control_unique_on_same": len(base),
                   "only_target": sorted(tgt - base), "both": sorted(tgt & base)}
    summary = summarize(args, contexts, sims, queries, n_gt, control)
    summary["config"] = {k: str(v) for k, v in vars(args).items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for lv, e in summary["levels"].items():
        print(f"{lv}: n={e['n_contexts']} sim_all={e['llama_all_topk']['mean']:.3f} sim_gov={e['llama_gov_topk']['mean']:.3f} "
              f"gao_shaped={e['gao_shaped_rate']:.3f} leaked={e['unique_leaked']} strict={e['unique_leaked_strict']} "
              f"(Email {e['unique_Email']}, Phone {e['unique_Phone']})")
    print("control:", control)


if __name__ == "__main__":
    main()
