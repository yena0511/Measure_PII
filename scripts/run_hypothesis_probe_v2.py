"""[V2 backup] Hypothesis-driven PII probing (V2) of a fine-tuned model. Current version: scripts/run_hypothesis_probe.py.

  1. Data profiling   model_info (scripts/extract_model_info.py) + model-card notes + dataset cards -> profile.json
                      (adds training_target and PII role/owner/position)                         (Qwen3-4B-Instruct-2507)
  2. Hypotheses       profile -> 5 hypotheses (document type + PII role/owner/location + context pattern),
                      each in a different place (re-asked if not)                                 (Qwen3-8B)
  3. Probing          k probes per hypothesis written like real training inputs, optionally with the start of the
                      reply to continue (Qwen3-4B-Instruct-2507), n samples each from the target
  4. Analysis         Email/Phone candidates (v2 regex), SM / RC / ES (Qwen3-8B), H = 0.3 SM + 0.2 RC + 0.5 ES;
                      RC counts only non-placeholder candidates (no example.com, 555 numbers, sequential digits)
  5. Update           Verify / Refine / Compare / Replace (Qwen3-8B); budget by UCB1 or the rule-based policy

Steps 1-2 are cached in --output-dir and shared by all policies; judgments are cached in judge_cache.jsonl.
Ground truth is never read. Dataset access stops at dataset cards (no data rows).

Usage:
  python scripts/run_hypothesis_probe_v2.py \
    --model-info outputs/model_info_v2/DISLab__SummLlama3-8B.json \
    --output-dir outputs/hypothesis_probe_v2/DISLab__SummLlama3-8B --policy ucb1 rule
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import set_seed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hypothesis_probe_v2.agent import Agent, card_notes  # noqa: E402
from hypothesis_probe_v2.llm import ChatLM  # noqa: E402
from hypothesis_probe_v2.loop import POLICIES, JudgeCache, run  # noqa: E402
from hypothesis_probe_v2.timing import StageTimer, banner  # noqa: E402
from model_profile.model_card import fetch_dataset_card, fetch_model_card, linked_dataset_ids  # noqa: E402

MODEL_INFO_FIELDS = ("model_id", "domain", "task", "fine_tuning_method", "training_data", "input_format", "output_format")


def cached(path, compute, timer, stage):
    with timer.stage(stage) as t:
        if path.exists():
            t["cached"] = True
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
    timer = StageTimer(out / "timing.jsonl")  # one row per stage; appended across runs
    t_all = timer.begin("run", target=target_id, policies=",".join(args.policy))

    agent = Agent(ChatLM(args.profiler), ChatLM(args.planner))
    sources = cached(out / "sources.json", lambda: collect_sources(target_id, args.dataset_id),
                     timer, "sources")
    banner(1, f"target={target_id}" + (" (reusing profile.json)" if (out / "profile.json").exists() else ""))
    profile = cached(out / "profile.json", lambda: {
        "agent": args.profiler, "model_info": model_info,
        "dataset_cards": [{k: d.get(k) for k in ("dataset_id", "revision", "error")} for d in sources["dataset_cards"]],
        **agent.profile(model_info, sources_text(sources, include_model_card=False),
                        notes=card_notes(sources["model_card"]["text"]))}, timer, "profile")
    banner(2, "reusing hypotheses.json" if (out / "hypotheses.json").exists() else "")
    hyps = cached(out / "hypotheses.json", lambda: {
        "agent": args.planner, "hypotheses": agent.hypotheses(profile, args.n_hypotheses)}, timer, "hypotheses")["hypotheses"]

    # V2: probes are written in the training data's format, so the question writer also gets the profiled
    # example structure and training target (a few lines, not the cards).
    fields = profile["fields"]
    probe_info = {**model_info, "training_example_format": {
        f: (fields.get(f) or {}).get("value") if isinstance(fields.get(f), dict) else fields.get(f)
        for f in ("document_structure", "training_target")}}
    target, cache = ChatLM(target_id), JudgeCache(out / "judge_cache.jsonl")
    for policy in args.policy:
        set_seed(args.seed)
        with timer.stage("policy", policy=policy):
            s = run(agent, target, probe_info, sources_text(sources), hyps, out / policy, policy, cache,
                    rounds=args.rounds, per_round=args.per_round, k=args.k, n=args.n,
                    sampling={"temperature": args.temperature, "top_p": args.top_p},
                    max_new_tokens=args.max_new_tokens, timer=timer)
        print(policy, json.dumps(s["unique_candidates_overall"]), "->", out / policy / "summary.json")
    timer.end(t_all)


if __name__ == "__main__":
    main()
