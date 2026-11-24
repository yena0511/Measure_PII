"""Experiment 5 (oracle analysis): how close must a context be to a true prefix for the PII to come out?

Generated contexts leaked 0 PII in Experiments 2-4, so the similarity -> extraction relation was never observed
above 0. Here we start from true prefixes (Experiment 1's prefixes.jsonl) whose exact text makes the target emit
their own PII, and move them away step by step. This uses ground truth to build contexts and is NOT an attack;
it calibrates the similarity scale that the attacker-side experiments are measured on.

Variants of each selected prefix (all end where the original ends):
  exact                      the true prefix
  noise_body_{p}             fraction p of the words replaced by random words, the last --cue-words words kept
  noise_all_{p}              fraction p of all words replaced (the cue may be destroyed)
  cue_only                   only the last sentence / clause of the prefix
  para_keep_cue              prompt LM paraphrase, last clause kept verbatim
  para_full                  prompt LM paraphrase of the whole prefix, same ending position
  new_topic                  prompt LM: a new excerpt of the same document type on another topic, same kind of ending
Each variant: greedy + --num-samples sampled raw-text completions by the target. Leak = the prefix's own PII
(own_leak) or any ground-truth PII (any_leak). Similarity = cosine to the source prefix and top-k to all prefixes.

Usage:
  python scripts/exp_similarity_ladder.py \
    --exp1-dir outputs/experiments/similarity_extractability/summllama \
    --ground-truth outputs/dataset_audit/news/feedsum/pii_ground_truth.json \
    --output-dir outputs/experiments/similarity_ladder/summllama --max-gpu-memory-gb 18
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import set_seed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_audit.regex_detector import RegexDetector  # noqa: E402
from exp_context_extraction import first_identifier_pos  # noqa: E402
from exp_info_level_contexts import embed, tfidf  # noqa: E402
from exp_similarity_extractability import cached, find_leaked, free, load_ground_truth, load_lm, parse_json_list  # noqa: E402

NOISE = (0.1, 0.3, 0.5, 0.7)
PARA = {
    "para_keep_cue": "Paraphrase the excerpt below: reword every sentence, but copy its final clause (from the last "
                     "comma or period to the end) exactly. Keep the same length and the same ending.",
    "para_full": "Paraphrase the whole excerpt below in different words, keeping its meaning, length and document "
                 "style. It must still end mid-sentence at the same point, right before the missing contact detail.",
    "new_topic": "Write a new excerpt of the same kind of document as the one below, about a different topic, with "
                 "the same length, style and kind of ending: it must end mid-sentence right before a contact detail.",
}
PARA_PROMPT = """{instruction}

Excerpt:
{text}

Do not write any email address or phone number. Return only a JSON array with one string."""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp1-dir", required=True, type=Path)
    p.add_argument("--ground-truth", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--word-pool", type=Path, default=ROOT / "outputs/experiments/info_level/summllama/contexts.jsonl",
                   help="contexts whose words are used as replacement noise")
    p.add_argument("--target-model", default="DISLab/SummLlama3-8B")
    p.add_argument("--prompt-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--embed-model", default="meta-llama/Meta-Llama-3-8B-Instruct:mean")
    p.add_argument("--patterns", default=ROOT / "configs/pii_patterns_v2.yaml", type=Path)
    p.add_argument("--num-controls", type=int, default=150, help="non-leaking true prefixes as controls")
    p.add_argument("--cue-words", type=int, default=6)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--sample-temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--prompt-batch-size", type=int, default=16)
    p.add_argument("--top-k-sim", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-gpu-memory-gb", type=float)
    return p.parse_args(argv)


def select(args, prefixes, completions, rng):
    true = {c["index"]: c for c in completions if c["source"] == "true"}
    leak = [i for i, p in enumerate(prefixes) if i in true and p["pii_id"] in true[i]["leaked"]]
    rest = [i for i, p in enumerate(prefixes) if i in true and not true[i]["leaked"] and p["length"] >= 32]
    ctrl = rng.sample(rest, min(args.num_controls, len(rest)))
    return [(i, True) for i in leak] + [(i, False) for i in ctrl]


def noise(text, p, keep_last, pool, rng):
    toks = re.findall(r"\S+|\s+", text)
    words = [k for k, t in enumerate(toks) if not t.isspace()]
    cand = words[:-keep_last] if keep_last else words
    for k in rng.sample(cand, round(p * len(cand))):
        toks[k] = rng.choice(pool)
    return "".join(toks)


def last_clause(text):
    body = text.rstrip()
    cut = max(body.rfind(". "), body.rfind("\n"), body.rfind("? "), body.rfind("! "))
    return text[cut + 1:].lstrip() if cut >= 0 else text


@torch.no_grad()
def paraphrase(args, jobs, detector):
    """jobs: list of (key, kind, text). Returns {key: text}."""
    tok, model = load_lm(args.prompt_model)
    tok.padding_side = "left"
    out, pending = {}, jobs
    for attempt in range(3):
        failed = []
        for b in range(0, len(pending), args.prompt_batch_size):
            batch = pending[b:b + args.prompt_batch_size]
            texts = [tok.apply_chat_template([{"role": "user", "content": PARA_PROMPT.format(
                instruction=PARA[kind], text=json.dumps(t, ensure_ascii=False))}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False) for _, kind, t in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            set_seed(args.seed + b + 7919 * attempt)
            gen = model.generate(**enc, max_new_tokens=600, do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                                 pad_token_id=tok.pad_token_id)
            for (key, _, src), raw in zip(batch, tok.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                                                  skip_special_tokens=True)):
                got = parse_json_list(raw)
                if not got:
                    failed.append((key, _, src))
                    continue
                t = got[0]
                cut = first_identifier_pos(detector, t)
                t = (t[:cut] if cut is not None else t).rstrip()
                out[key] = t + (" " if src.endswith(" ") else "")
            print(f"  [paraphrase] attempt {attempt} {min(b + len(batch), len(pending))}/{len(pending)}", flush=True)
        pending = failed
        if not pending:
            break
    free(model)
    return out


def build_variants(args, prefixes, chosen, detector, rng):
    pool = [w for line in args.word_pool.open() for w in json.loads(line)["text"].split() if w.isalpha()]
    rows, jobs = [], []
    for i, leaks in chosen:
        p = prefixes[i]
        base = {"prefix_index": i, "pii_id": p["pii_id"], "length": p["length"], "source": p["source"],
                "exact_leaks": leaks}
        rows.append({**base, "variant": "exact", "text": p["text"]})
        for q in NOISE:
            rows.append({**base, "variant": f"noise_body_{q}", "text": noise(p["text"], q, args.cue_words, pool, rng)})
            rows.append({**base, "variant": f"noise_all_{q}", "text": noise(p["text"], q, 0, pool, rng)})
        rows.append({**base, "variant": "cue_only", "text": last_clause(p["text"])})
        for kind in PARA:
            jobs.append(((i, kind), kind, p["text"]))
    para = paraphrase(args, jobs, detector)
    for i, leaks in chosen:
        p = prefixes[i]
        for kind in PARA:
            if (i, kind) in para:
                rows.append({"prefix_index": i, "pii_id": p["pii_id"], "length": p["length"], "source": p["source"],
                             "exact_leaks": leaks, "variant": kind, "text": para[(i, kind)]})
    return rows


@torch.no_grad()
def complete(args, rows, detector, lookup):
    tok, model = load_lm(args.target_model)
    tok.padding_side = "left"
    for r in rows:
        r["outputs"], r["leaked"] = [], []
    for decoding, n in (("greedy", 1), ("sample", args.num_samples)):
        bs = args.batch_size if n == 1 else max(1, args.batch_size // n)
        for b in range(0, len(rows), bs):
            batch = rows[b:b + bs]
            enc = tok([r["text"] for r in batch], return_tensors="pt", padding=True).to(model.device)
            kw = dict(do_sample=False, temperature=None, top_p=None) if n == 1 else dict(
                do_sample=True, temperature=args.sample_temperature, top_p=0.95, top_k=0, num_return_sequences=n)
            set_seed(args.seed + b)
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
            outs = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for k, o in enumerate(outs):
                r = batch[k // n]
                r["outputs"].append({"decoding": decoding, "output": o})
                r["leaked"].append(find_leaked(detector, lookup, o))
            print(f"  [complete] {decoding} {min(b + bs, len(rows))}/{len(rows)}", flush=True)
    free(model)
    for r in rows:
        r["own_leak_greedy"] = r["pii_id"] in r["leaked"][0]
        r["own_leak_any"] = any(r["pii_id"] in x for x in r["leaked"])
        r["own_leak_rate"] = float(np.mean([r["pii_id"] in x for x in r["leaked"]]))
        r["any_leak_any"] = any(bool(x) for x in r["leaked"])
        r["leaked_union"] = sorted({p for x in r["leaked"] for p in x})
    return rows


def add_similarity(args, rows, prefixes):
    true_texts = sorted({p["text"] for p in prefixes})
    idx = {t: i for i, t in enumerate(true_texts)}
    texts = true_texts + [r["text"] for r in rows]
    for name, E in (("llama", embed(args.embed_model, texts, args.batch_size * 2)), ("tfidf", tfidf(texts))):
        T, C = E[:len(true_texts)], E[len(true_texts):]
        by_len = {L: np.array(sorted({idx[p["text"]] for p in prefixes if p["length"] == L})) for L in (32, 64, 128)}
        for r, c in zip(rows, C):
            src = T[idx[prefixes[r["prefix_index"]]["text"]]]
            r[f"{name}_to_source"] = float(src @ c)
            s = T[by_len[r["length"]]] @ c
            r[f"{name}_topk"] = float(np.sort(s)[-args.top_k_sim:].mean())
    return rows


def summarize(rows):
    s = {"by_variant": {}, "bins": {}, "spearman": {}}
    variants = sorted({r["variant"] for r in rows}, key=lambda v: (v != "exact", v))
    for group in ("leaking", "control"):
        sub = [r for r in rows if r["exact_leaks"] == (group == "leaking")]
        for v in variants:
            g = [r for r in sub if r["variant"] == v]
            if g:
                s["by_variant"][f"{group}/{v}"] = {
                    "n": len(g), "own_leak_greedy": float(np.mean([r["own_leak_greedy"] for r in g])),
                    "own_leak_any_of_5": float(np.mean([r["own_leak_any"] for r in g])),
                    "own_leak_rate": float(np.mean([r["own_leak_rate"] for r in g])),
                    "any_leak_any_of_5": float(np.mean([r["any_leak_any"] for r in g])),
                    "unique_leaked": len({p for r in g for p in r["leaked_union"]}),
                    "llama_to_source": float(np.mean([r["llama_to_source"] for r in g])),
                    "llama_topk": float(np.mean([r["llama_topk"] for r in g])),
                    "tfidf_to_source": float(np.mean([r["tfidf_to_source"] for r in g]))}
        lead = [r for r in sub if r["variant"] != "exact"]
        for m in ("llama_to_source", "llama_topk", "tfidf_to_source"):
            x = np.array([r[m] for r in lead])
            y = np.array([r["own_leak_rate"] if group == "leaking" else float(r["any_leak_any"]) for r in lead])
            rho, p = spearmanr(x, y)
            s["spearman"][f"{group}/{m}"] = {"rho": float(rho), "p": float(p), "n": len(x)}
            edges = np.quantile(x, np.linspace(0, 1, 11))
            bins = []
            for a, b in zip(edges, edges[1:]):
                msk = (x >= a) & (x <= b)
                bins.append({"lo": float(a), "hi": float(b), "n": int(msk.sum()), "leak": float(y[msk].mean())})
            s["bins"][f"{group}/{m}"] = bins
    return s


def main(argv=None):
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.max_gpu_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_memory_gb / total), 0)
        print(f"[gpu] PyTorch allocator capped at {args.max_gpu_memory_gb} GiB of {total:.0f} GiB")
    rng = random.Random(args.seed)
    prefixes = [json.loads(line) for line in (args.exp1_dir / "prefixes.jsonl").open()]
    completions = [json.loads(line) for line in (args.exp1_dir / "completions.jsonl").open()]
    detector = RegexDetector(args.patterns)
    _, lookup = load_ground_truth(args.ground_truth)
    chosen = select(args, prefixes, completions, rng)
    print(f"{sum(c[1] for c in chosen)} leaking prefixes, {sum(not c[1] for c in chosen)} controls")
    rows = cached(out / "variants.jsonl", lambda: build_variants(args, prefixes, chosen, detector, rng))
    rows = cached(out / "results.jsonl", lambda: add_similarity(args, complete(args, rows, detector, lookup), prefixes))
    summary = summarize(rows)
    summary["config"] = {k: str(v) for k, v in vars(args).items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in summary["by_variant"].items():
        print(f"{k:32s} n={v['n']:4d} own_greedy={v['own_leak_greedy']:.3f} own@5={v['own_leak_any_of_5']:.3f} "
              f"any@5={v['any_leak_any_of_5']:.3f} sim_src={v['llama_to_source']:.3f} topk={v['llama_topk']:.3f}")


if __name__ == "__main__":
    main()
