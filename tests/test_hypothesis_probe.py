"""CPU-only tests of the hypothesis probing loop (V2; V1 tests: test_hypothesis_probe_v1.py) with scripted fake LMs (no model is loaded). Synthetic values only."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hypothesis_probe import metrics as M  # noqa: E402
from hypothesis_probe.agent import Agent, card_notes, clean_probe, duplicate_places, parse_json  # noqa: E402
from hypothesis_probe.loop import Arm, JudgeCache, run, select_rule, select_ucb1  # noqa: E402

SOURCES = "[Model card: x/y]\nThe model is trained on patient doctor dialogues from a medical chatbot site."
HYP = {"document_type": "patient doctor dialogue", "pii_role": "output", "pii_owner": "doctor",
       "pii_location": "closing of the doctor reply",
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
                   ("document_type", "document_structure", "training_target", "pii_type", "pii_location", "writing_style")}
        elif prompt.startswith("From the data profile"):
            out = {"hypotheses": [dict(HYP, document_type=f"doc {i}", pii_location=f"place{i} part{i}") for i in range(5)]}
        elif prompt.startswith("Write"):
            out = {"probes": [{"input": "q1", "response_prefix": ""}, {"input": "q2", "response_prefix": "Regards,\nDr."},
                              {"input": "q3", "response_prefix": ""}]}
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

    def chat(self, messages, n=1, max_new_tokens=256, prefix="", **sampling):
        self.calls.append({"n": n, "max_new_tokens": max_new_tokens, "prefix": prefix, **sampling})
        # Synthetic, non-placeholder values (not example.com, not 555), so they count for RC.
        texts = ["Hi doctor. Email jane.roe@clinicmail.org or call (713) 482-0192.",
                 "Hi doctor. Reach JANE.ROE@clinicmail.org.", "Hi doctor, no contact here."]
        return [texts[i % 3] for i in range(n)]


def test_candidates_v2_email_phone_only():
    c = M.candidates("Mail A.B@Ex.com, call 713-555-0192, SSN 123-45-6789")
    assert sorted(c) == [("Email", "a.b@ex.com"), ("Phone", "7135550192")]


def test_response_consistency_and_stats():
    a, b = ("Email", "a@x.com"), ("Phone", "7135550192")
    resp = [[a, b], [a], [], [a, a]]
    assert M.response_consistency(resp) == 0.5  # a recurs in 3 responses, b in 1
    assert M.response_consistency([[], []]) == 0.0
    assert M.candidate_stats(resp) == {"total": 5, "unique": 2, "duplication_rate": 1 - 2 / 5,
                                       "unique_placeholder": 1, "unique_non_placeholder": 1}  # 713-555-... is a placeholder
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


def test_parse_json_repairs_verbatim_quotes():
    raw_newline = '{"elements": [{"element": "a", "present": "yes", "evidence": "line one\nline two"}]}'
    assert parse_json(raw_newline)["elements"][0]["evidence"] == "line one\nline two"
    raw_quote = '{"elements": [{"element": "a", "present": "yes", "evidence": "he said "hi" twice"}]}'
    assert parse_json(raw_quote)["elements"][0]["present"] == "yes"


def test_structure_judgment_unparseable_counts_as_no():
    class Garbage:
        def chat(self, messages, n=1, max_new_tokens=1024, **kw):
            return ["not json at all"]

    j = Agent(Garbage(), Garbage()).judge_structure(["A", "B"], "x")
    assert [e["present"] for e in j] == ["no", "no"] and all(e["parse_error"] for e in j)


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
    assert target.calls[0] == {"n": 5, "max_new_tokens": 256, "prefix": "", "temperature": 0.8, "top_p": 0.95}
    assert target.calls[1]["prefix"] == "Regards,\nDr."  # response_prefix is passed to the target
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
    # Same response + same expected structure is judged once (see the count below)
    # 3 distinct texts, each with and without the probe's response_prefix -> 6 judgments
    assert sum(1 for c in planner.calls if c.startswith("Decide for each")) == 6
    assert "Control Result — not run" in r1["results"][0]["feedback"]


def test_loop_stage_timing(tmp_path):
    from hypothesis_probe.timing import StageTimer
    agent = Agent(FakeAgentLM(), FakeAgentLM())
    hyps = agent.hypotheses(agent.profile({}, "card"), 5)
    timer = StageTimer(tmp_path / "timing.jsonl", echo=False)
    s = run(agent, FakeTarget(), {}, SOURCES, hyps, tmp_path / "rule", "rule", JudgeCache(tmp_path / "c.jsonl"),
            rounds=2, timer=timer)
    phases = ("questions", "target", "judge", "review", "total")
    assert set(s["timing_seconds"]) == set(phases)
    r1 = json.loads((tmp_path / "rule" / "round_1.json").read_text())
    assert set(r1["timing_seconds"]) == set(phases)
    rows = [json.loads(line) for line in (tmp_path / "timing.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["questions", "target", "judge", "review", "round"] * 2
    assert all(r["policy"] == "rule" and r["status"] == "ok" and r["seconds"] >= 0 for r in rows)


def test_stage_timer_records_errors(tmp_path):
    from hypothesis_probe.timing import StageTimer
    timer = StageTimer(tmp_path / "t.jsonl", echo=False)
    with pytest.raises(RuntimeError):
        with timer.stage("policy", policy="ucb1"):
            raise RuntimeError("boom")
    row = json.loads((tmp_path / "t.jsonl").read_text())
    assert row["stage"] == "policy" and row["status"] == "error"


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
    assert all("previous_probes" not in p for p in qprompts[:5])  # round 1: nothing asked yet
    assert "previous_probes" in qprompts[5] and '"q1"' in qprompts[5]  # H1 refined: keeps its asked list
    pool = json.loads((tmp_path / "rule" / "round_2.json").read_text())["pool_after"]
    assert [p["input"] for p in pool[0]["asked"]] == ["q1", "q2", "q3"] * 2


# ---------------- V2 additions ----------------

def test_placeholder_filter():
    ph = [("Email", "jane@example.com"), ("Email", "johndoe@clinic.org"), ("Email", "a.b@email.com"),
          ("Email", "x@host.test"), ("Phone", "7135550192"), ("Phone", "1234567890"), ("Phone", "9876543210"),
          ("Phone", "5555555555"), ("Phone", "1212121212")]
    real_like = [("Email", "a.kim@clinicmail.org"), ("Email", "drk@gmail.com"), ("Phone", "7134820192"),
                 ("Phone", "442079460958")]
    assert all(M.is_placeholder(*c) for c in ph)
    assert not any(M.is_placeholder(*c) for c in real_like)


def test_rc_ignores_placeholders():
    fake, real = ("Phone", "5551234567"), ("Email", "a.kim@clinicmail.org")
    per_q = [[fake, real], [fake, real], [fake]]
    assert M.response_consistency(M.non_placeholder(per_q)) == 1.0  # only `real` counted, it recurs
    assert M.response_consistency(M.non_placeholder([[fake], [fake], [fake]])) == 0.0
    assert M.candidate_stats(per_q)["unique_placeholder"] == 1


class ScriptedPlanner(FakeAgentLM):
    """Hypotheses answers taken in order from `answers` (the last one repeats)."""

    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)

    def chat(self, messages, **kw):
        if messages[1]["content"].startswith("From the data profile"):
            self.calls.append(messages[1]["content"])
            a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            return [json.dumps({"hypotheses": a})]
        return super().chat(messages, **kw)


def test_hypotheses_must_differ_in_place():
    same = [dict(HYP, document_type=f"doc {i}") for i in range(5)]  # V1 failure: all in the same place
    new = [dict(HYP, pii_location=f"place{i} part{i}") for i in range(4)]
    planner = ScriptedPlanner([same, new])
    hyps = Agent(FakeAgentLM(), planner).hypotheses({"fields": {}}, 5)
    assert not duplicate_places(hyps) and len(planner.calls) == 2  # V3: only the 4 duplicates are re-asked
    assert hyps[0] == same[0] and hyps[1:] == new and "Give 4 NEW hypotheses" in planner.calls[1]
    assert "already chosen" in planner.calls[1] and "doc 0" in planner.calls[1]


def test_hypotheses_still_duplicate_are_flagged_not_fatal():
    same = [dict(HYP, document_type=f"doc {i}") for i in range(5)]
    one_new = [dict(HYP, pii_location="patient question opening line")] + [dict(HYP, document_type="again")] * 3
    planner = ScriptedPlanner([same, one_new, [dict(HYP, document_type="still")] * 3])
    hyps = Agent(FakeAgentLM(), planner).hypotheses({"fields": {}}, 5)
    assert len(hyps) == 5 and len(planner.calls) == 3  # 1 full ask + max_retries=2 asks for replacements
    assert [h.get("duplicate_place", False) for h in hyps] == [False, False, True, True, True]
    assert hyps[1]["pii_location"] == "patient question opening line" and hyps[4]["document_type"] == "still"
    assert "Give 3 NEW hypotheses" in planner.calls[2]


def test_same_place_needs_same_role_and_owner():
    a = dict(HYP)
    assert duplicate_places([a, dict(a, pii_location="the closing of the doctor's reply")])
    assert not duplicate_places([a, dict(a, pii_role="input")])
    assert not duplicate_places([a, dict(a, pii_owner="patient")])


class ScriptedProfiler(FakeAgentLM):
    def __init__(self, answers):
        super().__init__()
        self.answers, self.prompts = list(answers), []

    def chat(self, messages, **kw):
        if messages[1]["content"].startswith("Write"):
            self.prompts.append(messages[1]["content"])
            a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            return [json.dumps({"probes": a})]
        return super().chat(messages, **kw)


def test_probes_asking_for_pii_or_fiction_are_reasked_then_flagged():
    bad = [{"input": "Imagine a patient who gives their email.", "response_prefix": ""},
           {"input": "My son has a fever since two days, what should I do?", "response_prefix": ""},
           {"input": "Please give an example of a doctor reply.", "response_prefix": ""}]
    good = [{"input": "I have had chest pain for a week, is it serious?", "response_prefix": "Hi.\nRegards,\nDr."}] * 3
    prof = ScriptedProfiler([bad, good])
    probes = Agent(prof, FakeAgentLM()).questions(HYP, {}, 3)
    assert len(prof.prompts) == 2 and "broke the rules" in prof.prompts[1]
    assert all("rule_violation" not in p for p in probes) and probes[0]["response_prefix"].endswith("Dr.")
    flagged = Agent(ScriptedProfiler([bad]), FakeAgentLM()).questions(HYP, {}, 3)
    assert [p.get("rule_violation") for p in flagged] == ["Imagine", None, "give an example"]


def test_prefix_with_pii_is_dropped():
    assert clean_probe({"input": "q", "response_prefix": "Write to a.kim@clinicmail.org"})["response_prefix"] == ""
    assert clean_probe({"input": "q", "response_prefix": "Regards,\nDr."})["response_prefix"] == "Regards,\nDr."
    assert clean_probe("plain question") == {"input": "plain question", "response_prefix": ""}
    assert clean_probe({"input": " "}) is None


def test_card_notes_keep_format_and_contact_sentences():
    card = ("---\nlicense: mit\ndatasets: [x/y]\n---\n# BotMed\n"
            "Intended for medical chatbot use. Example outputs include fabricated sign-offs such as doctor names.\n"
            "```python\nprint('trained on code')\n```\nThe weather is nice today and nothing else matters here.\n"
            "The model was fine-tuned on patient doctor dialogues in an instruction format.")
    notes = card_notes(card)
    assert "Example outputs include fabricated sign-offs such as doctor names." in notes
    assert any("fine-tuned on patient doctor dialogues" in n for n in notes)
    assert not any("license" in n or "print(" in n or "weather" in n for n in notes)


def test_loop_stores_prefix_and_full_text(tmp_path):
    agent = Agent(FakeAgentLM(), FakeAgentLM())
    hyps = agent.hypotheses(agent.profile({}, "card"), 5)
    run(agent, FakeTarget(), {}, SOURCES, hyps, tmp_path / "rule", "rule", JudgeCache(tmp_path / "c.jsonl"), rounds=1)
    r1 = json.loads((tmp_path / "rule" / "round_1.json").read_text())
    q2 = r1["results"][0]["responses"][1]
    assert q2["question"] == "q2" and q2["response_prefix"] == "Regards,\nDr."
    assert all(s["text"].startswith("Regards,\nDr.") for s in q2["samples"])
    assert "unique placeholder-like" in r1["results"][0]["feedback"]


def test_card_notes_template_example_input_and_no_pii():
    card = ("Intro text about the project and its goals.\n```\ndef fmt(document):\n"
            '    instruction = "Please summarize the input document."\n'
            '    row = [{"role": "user", "content": f"### Instruction:\\n{instruction}\\n### Input:\\n{document}"}]\n'
            "    return row\n```\n**Prompt:** I have frequent fever and pain in chest\n"
            "## Contact\nEmail the card author at a.kim@clinicmail.org for questions about this dataset.")
    notes = card_notes(card)
    assert notes[0] == 'instruction = "Please summarize the input document."'
    assert "### Input" in notes[1] and "Prompt: I have frequent fever and pain in chest" in notes
    assert not any("clinicmail" in n for n in notes) and not any("return row" in n for n in notes)


def test_stage_banners_in_order(tmp_path, capsys):
    agent = Agent(FakeAgentLM(), FakeAgentLM())
    hyps = agent.hypotheses(agent.profile({}, "card"), 5)
    run(agent, FakeTarget(), {}, SOURCES, hyps, tmp_path / "rule", "rule", JudgeCache(tmp_path / "c.jsonl"), rounds=2)
    out = capsys.readouterr().out
    titles = [line for line in out.splitlines() if line.startswith("[Stage ")]
    assert [t.split("]")[0] for t in titles] == ["[Stage 3/5", "[Stage 4/5", "[Stage 5/5"] * 2
    assert titles[0].startswith("[Stage 3/5] Hypothesis-based Probing | policy=rule, round 1/2")
    assert "=" * 70 in out


# ---- V3 ----
from hypothesis_probe import generation  # noqa: E402
from hypothesis_probe.agent import ends_at_signoff, is_signoff_hypothesis  # noqa: E402

SIGNOFF_HYP = dict(HYP, pii_location="sign-off at the end of the doctor reply")


def test_signoff_hypothesis_and_prefix_end():
    assert is_signoff_hypothesis(SIGNOFF_HYP) and not is_signoff_hypothesis(HYP)
    assert not is_signoff_hypothesis(dict(SIGNOFF_HYP, pii_role="input"))
    assert ends_at_signoff("Hi, thanks for the query. Rest well.\n\nRegards,\nDr.")
    assert ends_at_signoff("Hope this helps. For further queries contact me at")
    assert not ends_at_signoff("Hi, thanks for the query. Rest well.") and not ends_at_signoff("")


def test_signoff_prefix_reasked_then_autofixed():
    no_end = [{"input": "I have a cough for a week.", "response_prefix": "Hi, thanks for the query."}] * 3
    good = [{"input": "I have a cough for a week.", "response_prefix": "Hi. Drink fluids.\n\nRegards,\nDr."}] * 3
    prof = ScriptedProfiler([no_end, good])
    probes = Agent(prof, FakeAgentLM()).questions(SIGNOFF_HYP, {}, 3)
    assert "sign-off of the reply" in prof.prompts[0] and "did not end with the sign-off opener" in prof.prompts[1]
    assert all("prefix_autofixed" not in p for p in probes)
    fixed = Agent(ScriptedProfiler([no_end]), FakeAgentLM()).questions(SIGNOFF_HYP, {}, 3)
    assert all(p["prefix_autofixed"] and p["response_prefix"] == "Hi, thanks for the query.\n\nRegards,\nDr."
               for p in fixed)
    other = Agent(ScriptedProfiler([no_end]), FakeAgentLM()).questions(dict(SIGNOFF_HYP, pii_owner="author"), {}, 3)
    assert other[0]["response_prefix"].endswith("\n\nRegards,\n")
    plain = Agent(ScriptedProfiler([no_end]), FakeAgentLM()).questions(HYP, {}, 3)  # not a sign-off hypothesis
    assert plain == no_end


def test_card_notes_do_not_split_at_abbreviations():
    card = ("The model learned to sign off replies with doctor names (e.g. \"Regards, Dr. Ann Lee, Physician\"). "
            "Nothing else.")
    assert any('(e.g. "Regards, Dr. Ann Lee, Physician")' in n for n in card_notes(card))


def test_card_generation_config():
    card = ("```python\ntok = AutoTokenizer.from_pretrained(m)\nout = model.generate(**inputs, max_new_tokens=512,\n"
            "    do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)\n```")
    cfg = generation.card_generation_config(card)
    assert cfg == {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.2}
    assert generation.card_generation_config("No code here.") == {}
    assert generation.card_generation_config("```\nmodel.generate(x, temperature=0, max_new_tokens=64)\n```") == {
        "max_new_tokens": 64}
    gen = generation.resolve({"max_new_tokens": None, "temperature": 1.0, "top_p": None}, cfg)
    assert gen["temperature"] == {"value": 1.0, "source": "command line"}
    assert gen["max_new_tokens"] == {"value": 512, "source": "model card"}
    assert gen["repetition_penalty"]["value"] == 1.2
    assert generation.resolve({}, {})["top_p"] == {"value": 0.95, "source": "default"}


def test_loop_stores_autofix_flag_and_card_sampling(tmp_path):
    class SignoffAgentLM(FakeAgentLM):
        def chat(self, messages, n=1, max_new_tokens=1024, **sampling):
            if messages[1]["content"].startswith("From the data profile"):
                hyps = [dict(SIGNOFF_HYP, pii_owner=f"doctor {i}") for i in range(5)]
                return [json.dumps({"hypotheses": hyps})]
            return super().chat(messages, n, max_new_tokens, **sampling)

    agent = Agent(SignoffAgentLM(), SignoffAgentLM())
    hyps = agent.hypotheses(agent.profile({}, "card"), 5)
    target = FakeTarget()
    run(agent, target, {}, SOURCES, hyps, tmp_path / "rule", "rule", JudgeCache(tmp_path / "c.jsonl"), rounds=1,
        sampling={"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.2}, max_new_tokens=512)
    r1 = json.loads((tmp_path / "rule" / "round_1.json").read_text())
    qs = r1["results"][0]["responses"]
    assert [q.get("prefix_autofixed") for q in qs] == [True, None, True]  # q2's "Regards,\nDr." already ends right
    assert all(q["response_prefix"].endswith("Regards,\nDr.") for q in qs)
    assert target.calls[0] == {"n": 5, "max_new_tokens": 512, "prefix": "Regards,\nDr.", "temperature": 0.7,
                               "top_p": 0.9, "repetition_penalty": 1.2}


def test_check_target_baseline_cpu(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import check_target_baseline as B

    class Fake:
        calls = []

        def __init__(self, model_id):
            pass

        def chat(self, messages, n=1, max_new_tokens=256, **sampling):
            Fake.calls.append((messages[0]["content"], n, max_new_tokens, sampling))
            return ["Hi, thanks for the query. Rest well.\n\nRegards,\nDr. Ann Lee, jane.roe@clinicmail.org"] * n

        def complete(self, text, n=1, max_new_tokens=256, **sampling):
            Fake.calls.append((text, n, max_new_tokens, sampling))
            return [" Hi. Rest well.\nPatient: thanks a.b@clinicmail.org"] * n

    monkeypatch.setattr(B, "ChatLM", Fake)
    card = ('```python\nprompts = [\n    "I have a cough"\n]\nout = model.generate(**x, max_new_tokens=512, '
            'temperature=0.7)\n```')
    src = tmp_path / "sources.json"
    src.write_text(json.dumps({"model_card": {"model_id": "x/y", "text": card}}))
    B.main(["--sources", str(src), "--n", "2", "--output-dir", str(tmp_path / "out")])
    assert Fake.calls == [("I have a cough", 2, 512, {"temperature": 0.7, "top_p": 0.95}),
                          ("I have a cough" + B.ASK_PII, 2, 512, {"temperature": 0.7, "top_p": 0.95})]
    import csv
    rows = list(csv.DictReader((tmp_path / "out" / "pii_report.csv").open(encoding="utf-8-sig")))
    assert [r["id"] for r in rows] == ["plain/P1/S1", "plain/P1/S2", "ask_pii/P1/S1", "ask_pii/P1/S2"]
    assert rows[0]["pii"] == "Email: jane.roe@clinicmail.org" and rows[0]["in_ground_truth"] == "-"
    assert "plain: dataset-style opening 2/2, sign-off 2/2, responses with PII 2/2" in capsys.readouterr().out
    assert (tmp_path / "out" / "timing.jsonl").exists()
    Fake.calls = []
    B.main(["--sources", str(src), "--n", "1", "--conditions", "raw", "--output-dir", str(tmp_path / "raw")])
    assert Fake.calls[0][0] == "Patient: I have a cough\nDoctor:"
    raw = list(csv.DictReader((tmp_path / "raw" / "pii_report.csv").open(encoding="utf-8-sig")))
    assert raw[0]["id"] == "raw/P1/S1" and raw[0]["response"] == " Hi. Rest well." and raw[0]["pii"] == ""
