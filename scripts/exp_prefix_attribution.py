"""Experiment 8: is true-prefix leakage due to fine-tuning? Target vs base model on the same true prefixes.

Experiment 7: attacker seeds leaked GAO contacts, but the base model leaked more than the target. SummLlama3 was trained
with DPO, where the document sits in the prompt and gets no direct loss, so document PII may come from pretraining
(public GAO reports) rather than from fine-tuning. Here Experiment 1's true prefixes go to both models, greedy, in
  raw       the prefix as plain text (as in Experiment 1)
  template  the card's training prompt with the prefix as the (unfinished) input document
Leak = the prefix's own PII (own) or any ground-truth PII (any); strict drops emails inferable from a word in the
prefix or the output.

Usage:
  python scripts/exp_prefix_attribution.py \
    --prefixes outputs/experiments/similarity_extractability/summllama/prefixes.jsonl \
    --ground-truth outputs/dataset_audit/news/feedsum/pii_ground_truth.json \
    --output-dir outputs/experiments/prefix_attribution/summllama --max-gpu-memory-gb 17
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_audit.regex_detector import RegexDetector  # noqa: E402
from exp_context_extraction import email_inferable  # noqa: E402
from exp_similarity_extractability import cached, find_leaked, free, load_ground_truth, load_lm  # noqa: E402

CARD_USER = ("Below is an instruction that describes a task. Write a response that appropriately completes the request."
             "\n\n### Instruction:\nPlease summarize the input documnet.\n\n### Input:\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefixes", required=True, type=Path)
    p.add_argument("--ground-truth", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--models", nargs="+", default=["DISLab/SummLlama3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"])
    p.add_argument("--formats", nargs="+", default=["raw", "template"])
    p.add_argument("--patterns", default=ROOT / "configs/pii_patterns_v2.yaml", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-gpu-memory-gb", type=float)
    return p.parse_args(argv)


def template_head(tok):
    sentinel = "⁣SENTINEL⁣"
    s = tok.apply_chat_template([{"role": "user", "content": CARD_USER + sentinel}], tokenize=False)
    return s[:s.index(sentinel)]


@torch.no_grad()
def run(args, model_name, prefixes, detector, lookup):
    tok, model = load_lm(model_name)
    tok.padding_side = "left"
    head = template_head(tok)
    rows = []
    for fmt in args.formats:
        jobs = [tok(p["text"])["input_ids"] if fmt == "raw" else tok(head + p["text"], add_special_tokens=False)["input_ids"]
                for p in prefixes]
        for b in range(0, len(jobs), args.batch_size):
            batch = jobs[b:b + args.batch_size]
            w = max(map(len, batch))
            ids = torch.tensor([[tok.pad_token_id] * (w - len(j)) + j for j in batch], device=model.device)
            mask = torch.tensor([[0] * (w - len(j)) + [1] * len(j) for j in batch], device=model.device)
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
            for k, o in enumerate(tok.batch_decode(gen[:, w:], skip_special_tokens=True)):
                i = b + k
                leaked = find_leaked(detector, lookup, o)
                rows.append({"model": model_name, "format": fmt, "index": i, "output": o, "leaked": leaked})
            print(f"  [{model_name} {fmt}] {min(b + args.batch_size, len(jobs))}/{len(jobs)}", flush=True)
    free(model)
    return rows


def summarize(prefixes, rows):
    s = {}
    for m in sorted({r["model"] for r in rows}):
        for fmt in sorted({r["format"] for r in rows}):
            sub = [r for r in rows if (r["model"], r["format"]) == (m, fmt)]
            for L in ("all", 32, 64, 128):
                g = [r for r in sub if L == "all" or prefixes[r["index"]]["length"] == L]
                own = [r for r in g if prefixes[r["index"]]["pii_id"] in r["leaked"]]
                strict = [r for r in own if not email_inferable(prefixes[r["index"]]["pii_id"],
                                                                prefixes[r["index"]]["text"] + " " + r["output"])]
                anyl = {p for r in g for p in r["leaked"]}
                anys = {p for r in g for p in r["leaked"]
                        if not email_inferable(p, prefixes[r["index"]]["text"] + " " + r["output"])}
                s[f"{m.split('/')[-1]}/{fmt}/{L}"] = {
                    "n": len(g), "own_rate": len(own) / len(g), "own_rate_strict": len(strict) / len(g),
                    "leak_rate": float(np.mean([bool(r["leaked"]) for r in g])),
                    "unique_any": len(anyl), "unique_any_strict": len(anys),
                    "unique_Email": sum(p.startswith("Email") for p in anyl),
                    "unique_Phone": sum(p.startswith("Phone") for p in anyl)}
    # per-prefix agreement between models (same format, own leak)
    models = sorted({r["model"] for r in rows})
    if len(models) == 2:
        for fmt in sorted({r["format"] for r in rows}):
            a = {r["index"]: prefixes[r["index"]]["pii_id"] in r["leaked"] for r in rows if (r["model"], r["format"]) == (models[0], fmt)}
            b = {r["index"]: prefixes[r["index"]]["pii_id"] in r["leaked"] for r in rows if (r["model"], r["format"]) == (models[1], fmt)}
            ua = {p for r in rows if (r["model"], r["format"]) == (models[0], fmt) for p in r["leaked"]}
            ub = {p for r in rows if (r["model"], r["format"]) == (models[1], fmt) for p in r["leaked"]}
            s[f"agreement/{fmt}"] = {
                "models": [m.split("/")[-1] for m in models],
                "own_both": sum(a[i] and b[i] for i in a), "own_only_first": sum(a[i] and not b[i] for i in a),
                "own_only_second": sum(b[i] and not a[i] for i in a),
                "unique_only_first": sorted(ua - ub), "unique_only_second": sorted(ub - ua), "unique_both": len(ua & ub)}
    return s


def main(argv=None):
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.max_gpu_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_memory_gb / total), 0)
        print(f"[gpu] PyTorch allocator capped at {args.max_gpu_memory_gb} GiB of {total:.0f} GiB")
    prefixes = [json.loads(line) for line in args.prefixes.open()]
    detector = RegexDetector(args.patterns)
    _, lookup = load_ground_truth(args.ground_truth)
    rows = []
    for m in args.models:
        rows += cached(out / f"outputs_{m.split('/')[-1]}.jsonl", lambda: run(args, m, prefixes, detector, lookup))
    s = summarize(prefixes, rows)
    s["config"] = {k: str(v) for k, v in vars(args).items()}
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in s.items():
        if k.endswith("/all"):
            print(f"{k:40s} own={v['own_rate']:.3f} own_strict={v['own_rate_strict']:.3f} any_unique={v['unique_any']} "
                  f"strict_unique={v['unique_any_strict']} (E {v['unique_Email']}, P {v['unique_Phone']})")
        elif k.startswith("agreement"):
            print(k, {a: (len(b) if isinstance(b, list) and a.startswith("unique_only") else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
