"""Hypothesis pool, budget policies (UCB1 / rule-based) and the probe -> analyze -> update loop (steps 3-5)."""
import hashlib
import json
import math
from pathlib import Path

from . import metrics as M
from .agent import hypothesis_view


class JudgeCache:
    """Judgments keyed by a hash of their input, kept in a JSON-lines file so no response is judged twice."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.data[row["key"]] = row["value"]

    @staticmethod
    def key(kind, *parts):
        return kind + ":" + hashlib.sha1(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()

    def get(self, key, compute):
        if key not in self.data:
            self.data[key] = compute()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": self.data[key]}, ensure_ascii=False) + "\n")
        return self.data[key]


class Arm:
    """One hypothesis version; its statistics restart when it is refined or replaced."""

    def __init__(self, hid, hypothesis, version=1, parent=None, asked=()):
        self.id, self.hypothesis, self.version, self.parent = hid, hypothesis, version, parent
        self.asked = list(asked)  # questions already sent for this hypothesis id (kept across Refine)
        self.pulls, self.total_reward, self.last_h, self.last_decision = 0, 0.0, None, None
        self.history = []  # [{round, SM, RC, ES, H_score, decision}]

    @property
    def mean(self):
        return self.total_reward / self.pulls if self.pulls else 0.0

    def state(self):
        return {"id": self.id, "version": self.version, "parent": self.parent, "hypothesis": self.hypothesis,
                "asked": self.asked, "pulls": self.pulls, "mean_h": self.mean, "last_h": self.last_h,
                "last_decision": self.last_decision, "history": self.history}


def select_ucb1(arms, m):
    """UCB1 with reward = H_score; unprobed arms first."""
    t = sum(a.pulls for a in arms)

    def ucb(a):
        return math.inf if a.pulls == 0 else a.mean + math.sqrt(2 * math.log(t) / a.pulls)

    return sorted(arms, key=ucb, reverse=True)[:m]


def select_rule(arms, m):
    """Agent-decision rule: unprobed, then last decision Verify/Refine, then the rest; ties by last H_score."""
    def rank(a):
        group = 0 if a.pulls == 0 else 1 if a.last_decision in ("Verify", "Refine") else 2
        return group, -(a.last_h or 0.0)

    return sorted(arms, key=rank)[:m]


POLICIES = {"ucb1": select_ucb1, "rule": select_rule}


def feedback_text(arm, r):
    s = r["scores"]
    lines = [
        f"Hypothesis {arm.id} (v{arm.version}) — {arm.hypothesis['document_type']}",
        f"PII Location: {arm.hypothesis['pii_location']}",
        f"Context Pattern: {arm.hypothesis['context_pattern']}",
        f"Structure {s['SM']:.2f} / Consistency {s['RC']:.2f} / Evidence {s['ES']:.2f}"
        f" / H_score {s['H_score']:.2f} ({s['confidence']} confidence)",
        f"Measured — {r['n_responses']} responses to {len(r['questions'])} questions; PII candidates "
        f"{r['candidates']['total']} total, {r['candidates']['unique']} unique, duplication {r['candidates']['duplication_rate']:.2f}",
        "Structure elements present (responses): "
        + ("; ".join(f"{e}: {c}/{r['n_responses']}" for e, c in r["structure_counts"].items()) or "none expected"),
        "Evidence — " + "; ".join(f"{c['claim']}: {c['label']}" + (f' ("{c["quote"]}")' if c["label"] != "unsupported" else "")
                                   for c in r["evidence"]["claims"]),
        "Control Result — not run (negative control is excluded in V1)",
    ]
    if arm.history:
        lines.append("Previous rounds — " + "; ".join(
            f"round {h['round']}: H {h['H_score']:.2f}, {h['decision']}" for h in arm.history))
    return "\n".join(lines)


def run(agent, target, model_info, sources, initial_hypotheses, out_dir, policy, cache,
        rounds=5, per_round=2, k=3, n=5, sampling=None, max_new_tokens=256):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampling = sampling or {"temperature": 0.8, "top_p": 0.95}
    arms = [Arm(f"H{i + 1}", h) for i, h in enumerate(initial_hypotheses)]
    next_id, select = len(arms) + 1, POLICIES[policy]
    log = []

    for rnd in range(1, rounds + 1):
        chosen = list(arms) if rnd == 1 else select(arms, per_round)
        # Phases are ordered by model so each round loads every LM once.
        qs = {a.id: agent.questions(a.hypothesis, model_info, k, a.asked) for a in chosen}
        for a in chosen:
            a.asked += qs[a.id]
        resp = {a.id: [target.chat([{"role": "user", "content": q}], n=n, max_new_tokens=max_new_tokens, **sampling)
                       for q in qs[a.id]] for a in chosen}
        results = {}
        for a in chosen:
            expected = a.hypothesis["expected_structure"]
            per_q = [[M.candidates(t) for t in texts] for texts in resp[a.id]]
            flat_resp = [t for texts in resp[a.id] for t in texts]
            sm_j = [cache.get(cache.key("SM", expected, t), lambda t=t: agent.judge_structure(expected, t))
                    for t in flat_resp]
            es_j = cache.get(cache.key("ES", hypothesis_view(a.hypothesis), sources),
                             lambda: agent.judge_evidence(a.hypothesis, sources))
            sm = sum(M.structural_match(j, len(expected)) for j in sm_j) / len(sm_j)
            rc = sum(M.response_consistency(c) for c in per_q) / len(per_q)
            es = M.evidence_support(es_j["claims"])
            h = M.h_score(sm, rc, es)
            results[a.id] = {
                "arm": a.id, "version": a.version, "hypothesis": a.hypothesis, "questions": qs[a.id],
                "responses": [{"question": q, "samples": [{"text": t, "candidates": c, "structure": j}
                                                         for t, c, j in zip(texts, cands, sm_j[i * n:(i + 1) * n])]}
                              for i, (q, texts, cands) in enumerate(zip(qs[a.id], resp[a.id], per_q))],
                "n_responses": len(flat_resp),
                "candidates": M.candidate_stats([c for cs in per_q for c in cs]),
                "structure_counts": {e: sum(j[i]["present"] == "yes" for j in sm_j) for i, e in enumerate(expected)},
                "evidence": es_j,
                "scores": {"SM": sm, "RC": rc, "ES": es, "H_score": h, "confidence": M.confidence(h)},
            }

        # Step 5: agent review and actions
        for a in chosen:
            r = results[a.id]
            r["feedback"] = feedback_text(a, r)
            pool = [{"id": b.id, **hypothesis_view(b.hypothesis)} for b in arms]
            r["review"] = review = agent.review(r["feedback"], pool)
            a.pulls += 1
            a.total_reward += r["scores"]["H_score"]
            a.last_h, a.last_decision = r["scores"]["H_score"], review["decision"]
            a.history.append({"round": rnd, **{x: r["scores"][x] for x in ("SM", "RC", "ES", "H_score")},
                              "decision": review["decision"]})
            if review["decision"] == "Refine":
                new = Arm(a.id, review["hypothesis"], a.version + 1, parent=f"{a.id}v{a.version}", asked=a.asked)
            elif review["decision"] == "Replace":
                new, next_id = Arm(f"H{next_id}", review["hypothesis"], parent=f"{a.id}v{a.version}"), next_id + 1
            else:
                continue
            r["retired"] = a.state()
            arms[arms.index(a)] = new

        record = {"round": rnd, "policy": policy, "selected": [a.id for a in chosen],
                  "results": list(results.values()), "pool_after": [a.state() for a in arms]}
        (out_dir / f"round_{rnd}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.append({"round": rnd, "selected": record["selected"],
                    "scores": {i: r["scores"] for i, r in results.items()},
                    "decisions": {i: r["review"]["decision"] for i, r in results.items()},
                    "unique_candidates": {i: r["candidates"]["unique"] for i, r in results.items()}})

    all_keys = set()
    for rnd in range(1, rounds + 1):
        rec = json.loads((out_dir / f"round_{rnd}.json").read_text(encoding="utf-8"))
        for r in rec["results"]:
            for q in r["responses"]:
                for s in q["samples"]:
                    all_keys.update(map(tuple, s["candidates"]))
    summary = {"policy": policy, "rounds": rounds, "per_round": per_round, "k": k, "n": n,
               "sampling": {**sampling, "max_new_tokens": max_new_tokens},
               "total_queries": sum(len(x["selected"]) for x in log) * k,
               "unique_candidates_overall": {t: sum(1 for l, _ in all_keys if l == t) for t in M.PII_TYPES},
               "rounds_log": log, "final_pool": [a.state() for a in arms]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
