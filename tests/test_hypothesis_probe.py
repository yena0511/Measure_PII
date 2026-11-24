"""CPU-only tests of the hypothesis probing loop with scripted fake LMs (no model is loaded). Synthetic values only."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hypothesis_probe import metrics as M  # noqa: E402
from hypothesis_probe.agent import Agent, parse_json  # noqa: E402
from hypothesis_probe.loop import Arm, JudgeCache, run, select_rule, select_ucb1  # noqa: E402

SOURCES = "[Model card: x/y]\nThe model is trained on patient doctor dialogues from a medical chatbot site."
HYP = {"document_type": "patient doctor dialogue", "pii_location": "closing of the doctor reply",
       "context_pattern": "contact me at <email>", "expected_structure": ["Patient question", "Doctor reply"]}


class FakeAgentLM:
    """Answers each agent prompt with fixed JSON, chosen by the prompt's first line."""

    def __init__(self, decisions=("Compare",)):
        self.decisions, self.calls = list(decisions), []

    def truncate(self, text, max_tokens):
        return text, False

    def chat(self, messages, n=1, max_new_tokens=1024, **sampling):
        assert not sampling, "agents must decode greedily"
        prompt = messages[1]["content"]
        self.calls.append(prompt.split("\n", 1)[0])
        if prompt.startswith("Profile"):
            out = {f: {"value": "v", "basis": "assumption", "evidence": []} for f in
                   ("document_type", "document_structure", "pii_type", "pii_location", "writing_style")}
        elif prompt.startswith("From the data profile"):
            out = {"hypotheses": [dict(HYP, document_type=f"doc {i}") for i in range(5)]}
        elif prompt.startswith("Write"):
            out = {"questions": ["q1", "q2", "q3"]}
        elif prompt.startswith("Decide for each"):
            out = {"elements": [{"element": "Patient question", "present": "yes", "evidence": "Hi doctor"},
                                {"element": "Doctor reply", "present": "no", "evidence": ""}]}
        elif prompt.startswith("Judge whether"):
            out = {"claims": [
                {"claim": "document_type", "label": "supported", "quote": "trained on patient doctor dialogues"},
                {"claim": "pii_location", "label": "partial", "quote": "invented sentence not in any card"},
                {"claim": "context_pattern", "label": "unsupported", "quote": ""}]}
        elif prompt.startswith("Review"):
            d = self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]
            out = {"observed": "o", "decision": d, "revision": "r",
                   "hypothesis": dict(HYP, document_type="revised") if d in ("Refine", "Replace") else None}
        else:
            raise AssertionError(prompt[:80])
        return ["Sure.\n" + json.dumps(out)]


class FakeTarget:
    def __init__(self):
        self.calls = []

    def chat(self, messages, n=1, max_new_tokens=256, **sampling):
        self.calls.append({"n": n, "max_new_tokens": max_new_tokens, **sampling})
        texts = ["Hi doctor. Email jane.doe@example.org or call (713) 555-0192.",
                 "Hi doctor. Reach JANE.DOE@example.org.", "Hi doctor, no contact here."]
        return [texts[i % 3] for i in range(n)]


def test_candidates_v2_email_phone_only():
    c = M.candidates("Mail A.B@Ex.com, call 713-555-0192, SSN 123-45-6789")
    assert sorted(c) == [("Email", "a.b@ex.com"), ("Phone", "7135550192")]


def test_response_consistency_and_stats():
    a, b = ("Email", "a@x.com"), ("Phone", "7135550192")
    resp = [[a, b], [a], [], [a, a]]
    assert M.response_consistency(resp) == 0.5  # a recurs in 3 responses, b in 1
    assert M.response_consistency([[], []]) == 0.0
    assert M.candidate_stats(resp) == {"total": 5, "unique": 2, "duplication_rate": 1 - 2 / 5}
    assert M.candidate_stats([[]])["duplication_rate"] == 0.0


def test_scores():
    assert M.structural_match([{"present": "yes"}, {"present": "no"}], 2) == 0.5
    assert M.structural_match([], 0) == 0.0
    assert M.evidence_support([{"label": "supported"}, {"label": "partial"}, {"label": "unsupported"}]) == 0.5
    assert M.h_score(1, 1, 1) == pytest.approx(1.0)
    assert M.h_score(0.85, 0.72, 0.40) == pytest.approx(0.3 * 0.85 + 0.2 * 0.72 + 0.5 * 0.40)
    assert [M.confidence(h) for h in (0.8, 0.79, 0.6, 0.2)] == ["high", "medium", "low", "low"]


def test_evidence_unverified_quote_is_unsupported():
    agent = Agent(FakeAgentLM(), FakeAgentLM())
    es = agent.judge_evidence(HYP, SOURCES)
    assert [c["label"] for c in es["claims"]] == ["supported", "unsupported", "unsupported"]
    assert es["claims"][1]["agent_label"] == "partial" and not es["claims"][1]["quote_in_sources"]


def test_structure_judgment_by_position_and_empty():
    agent = Agent(FakeAgentLM(), FakeAgentLM())
    j = agent.judge_structure(["Patient question", "Doctor reply", "Signature"], "Hi doctor")
    assert [e["present"] for e in j] == ["yes", "no", "no"]  # missing third entry counts as no
    assert agent.judge_structure([], "x") == []


def test_parse_json_error():
    with pytest.raises(ValueError):
        parse_json("no json")


def test_policies():
    arms = [Arm(f"H{i}", HYP) for i in range(1, 4)]
    for a, (p, r, d) in zip(arms, [(2, 1.0, "Compare"), (1, 0.4, "Verify"), (3, 1.2, "Refine")]):
        a.pulls, a.total_reward, a.last_h, a.last_decision = p, r, r / p, d
    assert [a.id for a in select_rule(arms, 2)] == ["H2", "H3"]  # Verify/Refine before Compare, then by last H
    assert [a.id for a in select_ucb1(arms, 1)] == ["H2"]       # fewest pulls -> largest bonus
    arms.append(Arm("H4", HYP))
    assert select_ucb1(arms, 1)[0].id == "H4" and select_rule(arms, 1)[0].id == "H4"


@pytest.mark.parametrize("policy", ["ucb1", "rule"])
def test_loop_end_to_end(tmp_path, policy):
    profiler, planner = FakeAgentLM(), FakeAgentLM(decisions=["Refine", "Replace", "Compare"])
    agent, target = Agent(profiler, planner), FakeTarget()
    hyps = agent.hypotheses(agent.profile({"domain": "Medical"}, "card"), 5)
    cache = JudgeCache(tmp_path / "cache.jsonl")
    s = run(agent, target, {"domain": "Medical"}, SOURCES, hyps, tmp_path / policy, policy, cache, rounds=3)

    assert [len(r["selected"]) for r in s["rounds_log"]] == [5, 2, 2]
    assert s["total_queries"] == 9 * 3 and len(target.calls) == 27
    assert target.calls[0] == {"n": 5, "max_new_tokens": 256, "temperature": 0.8, "top_p": 0.95}
    assert s["unique_candidates_overall"] == {"Email": 1, "Phone": 1}
    r1 = json.loads((tmp_path / policy / "round_1.json").read_text())
    sc = r1["results"][0]["scores"]
    assert sc["SM"] == 0.5 and sc["ES"] == pytest.approx(1 / 3)
    assert sc["RC"] == 1.0  # 5 samples cycle 3 texts: email in 4, phone in 2 -> both recur
    assert sc["H_score"] == pytest.approx(0.3 * 0.5 + 0.2 * 1.0 + 0.5 / 3)
    # Refine keeps the id with a new version; Replace adds H6; pool size stays 5
    pool = r1["pool_after"]
    assert len(pool) == 5 and pool[0]["version"] == 2 and pool[0]["pulls"] == 0 and pool[1]["id"] == "H6"
    # Unprobed (refined / replaced) hypotheses are picked next by both policies
    assert s["rounds_log"][1]["selected"] == ["H1", "H6"]
    # Same response + same expected structure is judged once: 3 distinct texts
    assert sum(1 for c in planner.calls if c.startswith("Decide for each")) == 3
    assert "Control Result — not run" in r1["results"][0]["feedback"]


def test_previous_questions_are_passed(tmp_path):
    class Recorder(FakeAgentLM):
        def chat(self, messages, **kw):
            self.prompts = getattr(self, "prompts", []) + [messages[1]["content"]]
            return super().chat(messages, **kw)

    profiler = Recorder()
    agent = Agent(profiler, FakeAgentLM(decisions=["Refine", "Compare"]))
    hyps = [dict(HYP, document_type=f"doc {i}") for i in range(5)]
    run(agent, FakeTarget(), {}, SOURCES, hyps, tmp_path / "rule", "rule", JudgeCache(tmp_path / "c.jsonl"), rounds=2)
    qprompts = [p for p in profiler.prompts if p.startswith("Write")]
    assert all("previous_questions" not in p for p in qprompts[:5])  # round 1: nothing asked yet
    assert "previous_questions" in qprompts[5] and '"q1"' in qprompts[5]  # H1 refined: keeps its asked list
    pool = json.loads((tmp_path / "rule" / "round_2.json").read_text())["pool_after"]
    assert pool[0]["asked"] == ["q1", "q2", "q3"] * 2
