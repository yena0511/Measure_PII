"""Experiment 9: does fine-tuning raise the likelihood of the true PII after its prefix? (target vs base log-probs)

Experiment 8: greedy leakage from true prefixes is the same for the target and its base model. Greedy output is coarse,
so here we score log p(PII | prefix) under both models by teacher forcing. For each true prefix (Experiment 1):
  true       its own PII (max over the PII's surface forms)
  swapped    --num-swaps other ground-truth PII of the same type (same scoring)
Memorization of the prefix -> PII association by fine-tuning would show as delta = logp_target - logp_base larger for
true than for swapped candidates, and as a better rank of the true PII among the candidates in the target.

Usage:
  python scripts/exp_pii_logprob_attribution.py \
    --prefixes outputs/experiments/similarity_extractability/summllama/prefixes.jsonl \
    --ground-truth outputs/dataset_audit/news/feedsum/pii_ground_truth.json \
    --output-dir outputs/experiments/pii_logprob/summllama --max-gpu-memory-gb 17
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from exp_prefix_attribution import template_head  # noqa: E402
from exp_similarity_extractability import cached, free, load_ground_truth, load_lm  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefixes", required=True, type=Path)
    p.add_argument("--ground-truth", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--target-model", default="DISLab/SummLlama3-8B")
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--formats", nargs="+", default=["raw", "template"])
    p.add_argument("--num-swaps", type=int, default=5)
    p.add_argument("--matched-swaps", action="store_true",
                   help="draw swaps from the same family (email domain / first 6 phone digits) when possible")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-gpu-memory-gb", type=float)
    return p.parse_args(argv)


def family(pid):
    t, k = pid.split(":", 1)
    return k.split("@")[-1] if t == "Email" else k[:6]


def build_jobs(args, prefixes, entries):
    rng = random.Random(args.seed)
    forms = {f"{e['type']}:{e['key']}": sorted(set(e["surface_forms"]))[:3] for e in entries}
    by_type = {t: [pid for pid in forms if pid.startswith(t)] for t in ("Email", "Phone")}
    jobs = []  # (prefix index, candidate pid, is_true, surface form)
    for i, p in enumerate(prefixes):
        if p["pii_id"] not in forms:
            continue
        t = p["pii_id"].split(":")[0]
        pool = [x for x in by_type[t] if x != p["pii_id"]]
        if args.matched_swaps:
            fam = family(p["pii_id"])
            same = [x for x in pool if family(x) == fam]
            pool = same if len(same) >= args.num_swaps else pool
        swaps = rng.sample(pool, args.num_swaps)
        for pid, is_true in [(p["pii_id"], True)] + [(x, False) for x in swaps]:
            for f in forms[pid]:
                jobs.append((i, pid, is_true, f))
    return jobs


@torch.no_grad()
def score(args, model_name, prefixes, jobs):
    tok, model = load_lm(model_name)
    head = template_head(tok)
    out = {}
    for fmt in args.formats:
        seqs = []
        for i, pid, is_true, form in jobs:
            text = prefixes[i]["text"].rstrip()
            ctx = tok(text)["input_ids"] if fmt == "raw" else tok(head + text, add_special_tokens=False)["input_ids"]
            cand = tok(" " + form, add_special_tokens=False)["input_ids"]
            seqs.append((ctx, cand))
        lp = []
        for b in range(0, len(seqs), args.batch_size):
            batch = seqs[b:b + args.batch_size]
            w = max(len(c) + len(d) for c, d in batch)
            ids = torch.full((len(batch), w), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), w), dtype=torch.long)
            for k, (c, d) in enumerate(batch):  # right padding
                ids[k, :len(c) + len(d)] = torch.tensor(c + d)
                mask[k, :len(c) + len(d)] = 1
            logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits
            for k, (c, d) in enumerate(batch):
                pos = torch.arange(len(c) - 1, len(c) + len(d) - 1, device=model.device)
                lps = torch.log_softmax(logits[k, pos].float(), dim=-1)
                tgt = torch.tensor(d, device=model.device)
                lp.append(float(lps.gather(1, tgt[:, None]).sum()))
            del logits
            if b % (args.batch_size * 200) == 0:
                print(f"  [{model_name} {fmt}] {b}/{len(seqs)}", flush=True)
        out[fmt] = lp
    free(model)
    return out


def summarize(prefixes, jobs, target, base):
    s = {}
    for fmt in target:
        cand = {}  # (prefix, pid) -> (is_true, max lp target, max lp base)
        for (i, pid, is_true, _), lt, lb in zip(jobs, target[fmt], base[fmt]):
            k = (i, pid)
            prev = cand.get(k, (is_true, -1e9, -1e9))
            cand[k] = (is_true, max(prev[1], lt), max(prev[2], lb))
        per_prefix = {}
        for (i, pid), v in cand.items():
            per_prefix.setdefault(i, []).append(v)
        d_true, d_swap, rank_t, rank_b, lt_true, lb_true = [], [], [], [], [], []
        for i, vs in per_prefix.items():
            t = [v for v in vs if v[0]][0]
            sw = [v for v in vs if not v[0]]
            d_true.append(t[1] - t[2])
            d_swap.append(float(np.mean([v[1] - v[2] for v in sw])))
            rank_t.append(1 + sum(v[1] > t[1] for v in sw))
            rank_b.append(1 + sum(v[2] > t[2] for v in sw))
            lt_true.append(t[1])
            lb_true.append(t[2])
        d_true, d_swap = np.array(d_true), np.array(d_swap)
        diff = d_true - d_swap
        stat = wilcoxon(diff) if np.any(diff != 0) else None
        s[fmt] = {"n_prefixes": len(d_true),
                  "mean_logp_true_target": float(np.mean(lt_true)), "mean_logp_true_base": float(np.mean(lb_true)),
                  "delta_true_mean": float(d_true.mean()), "delta_swap_mean": float(d_swap.mean()),
                  "delta_true_minus_swap_mean": float(diff.mean()), "delta_true_minus_swap_median": float(np.median(diff)),
                  "share_delta_true_gt_swap": float(np.mean(diff > 0)),
                  "wilcoxon_p": None if stat is None else float(stat.pvalue),
                  "top1_target": float(np.mean(np.array(rank_t) == 1)), "top1_base": float(np.mean(np.array(rank_b) == 1)),
                  "mean_rank_target": float(np.mean(rank_t)), "mean_rank_base": float(np.mean(rank_b))}
        for L in (32, 64, 128):
            idx = [n for n, i in enumerate(per_prefix) if prefixes[i]["length"] == L]
            s[fmt][f"delta_true_minus_swap_mean_L{L}"] = float(diff[idx].mean())
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
    entries, _ = load_ground_truth(args.ground_truth)
    jobs = build_jobs(args, prefixes, entries)
    print(f"{len(jobs)} scored sequences per model and format")

    def load_or(name, model):
        p = out / f"logp_{name}.json"
        if p.exists():
            return json.loads(p.read_text())
        r = score(args, model, prefixes, jobs)
        p.write_text(json.dumps(r))
        return r
    target = load_or("target", args.target_model)
    base = load_or("base", args.base_model)
    s = summarize(prefixes, jobs, target, base)
    s["config"] = {k: str(v) for k, v in vars(args).items()}
    (out / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
