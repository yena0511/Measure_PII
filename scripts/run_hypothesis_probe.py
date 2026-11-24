"""Hypothesis-driven PII probing (V1) of a fine-tuned model.

  1. Data profiling   model_info (scripts/extract_model_info.py) + dataset cards -> profile.json   (Qwen3-4B-Instruct-2507)
  2. Hypotheses       profile -> 5 hypotheses (document type + PII location + context pattern)     (Qwen3-8B)
  3. Probing          k questions per hypothesis (Qwen3-4B-Instruct-2507), n samples each from the target
  4. Analysis         Email/Phone candidates (v2 regex), SM / RC / ES (Qwen3-8B), H = 0.3 SM + 0.2 RC + 0.5 ES
  5. Update           Verify / Refine / Compare / Replace (Qwen3-8B); budget by UCB1 or the rule-based policy

Steps 1-2 are cached in --output-dir and shared by all policies; judgments are cached in judge_cache.jsonl.
Ground truth is never read. Dataset access stops at dataset cards (no data rows).

Usage:
  python scripts/run_hypothesis_probe.py \
    --model-info outputs/model_info/DISLab__SummLlama3-8B.json \
    --output-dir outputs/hypothesis_probe/DISLab__SummLlama3-8B --policy ucb1 rule
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import set_seed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hypothesis_probe.agent import Agent  # noqa: E402
from hypothesis_probe.llm import ChatLM  # noqa: E402
from hypothesis_probe.loop import POLICIES, JudgeCache, run  # noqa: E402
from model_profile.model_card import fetch_dataset_card, fetch_model_card, linked_dataset_ids  # noqa: E402

MODEL_INFO_FIELDS = ("model_id", "domain", "task", "fine_tuning_method", "training_data", "input_format", "output_format")


def cached(path, compute):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = compute()
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return value


def collect_sources(model_id, extra_dataset_ids):
    card = fetch_model_card(model_id)
    ids = list(dict.fromkeys(linked_dataset_ids(card["text"]) + list(extra_dataset_ids)))
    datasets = []
    for d in ids:
        try:
            datasets.append(fetch_dataset_card(d))
        except Exception as e:  # gated, renamed or deleted dataset: recorded, not fatal
            datasets.append({"dataset_id": d, "error": f"{type(e).__name__}: {e}"})
    return {"model_card": card, "dataset_cards": datasets}


def sources_text(sources, include_model_card=True):
    parts = [f"[Model card: {sources['model_card']['model_id']}]\n{sources['model_card']['text']}"] if include_model_card else []
    parts += [f"[Dataset card: {d['dataset_id']}]\n{d['text']}" for d in sources["dataset_cards"] if "text" in d]
    return "\n\n".join(parts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-info", type=Path, required=True, help="output of scripts/extract_model_info.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target", help="target model id (default: model_id in --model-info)")
    p.add_argument("--dataset-id", nargs="*", default=[], help="extra dataset cards to read, besides those linked from the model card")
    p.add_argument("--policy", nargs="+", choices=sorted(POLICIES), default=["ucb1", "rule"])
    p.add_argument("--profiler", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--planner", default="Qwen/Qwen3-8B")
    p.add_argument("--n-hypotheses", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--per-round", type=int, default=2)
    p.add_argument("--k", type=int, default=3, help="questions per hypothesis per round")
    p.add_argument("--n", type=int, default=5, help="sampled responses per question")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    info = json.loads(args.model_info.read_text(encoding="utf-8"))
    model_info = {f: info.get(f) for f in MODEL_INFO_FIELDS}
    target_id = args.target or info["model_id"]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    agent = Agent(ChatLM(args.profiler), ChatLM(args.planner))
    sources = cached(out / "sources.json", lambda: collect_sources(target_id, args.dataset_id))
    profile = cached(out / "profile.json", lambda: {
        "agent": args.profiler, "model_info": model_info,
        "dataset_cards": [{k: d.get(k) for k in ("dataset_id", "revision", "error")} for d in sources["dataset_cards"]],
        **agent.profile(model_info, sources_text(sources, include_model_card=False))})
    hyps = cached(out / "hypotheses.json", lambda: {
        "agent": args.planner, "hypotheses": agent.hypotheses(profile, args.n_hypotheses)})["hypotheses"]

    target, cache = ChatLM(target_id), JudgeCache(out / "judge_cache.jsonl")
    for policy in args.policy:
        set_seed(args.seed)
        s = run(agent, target, model_info, sources_text(sources), hyps, out / policy, policy, cache,
                rounds=args.rounds, per_round=args.per_round, k=args.k, n=args.n,
                sampling={"temperature": args.temperature, "top_p": args.top_p}, max_new_tokens=args.max_new_tokens)
        print(policy, json.dumps(s["unique_candidates_overall"]), "->", out / policy / "summary.json")


if __name__ == "__main__":
    main()
