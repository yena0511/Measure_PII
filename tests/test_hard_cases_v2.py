"""Regression test: configs/pii_patterns_v2.yaml must pass every scored hard case (synthetic values only)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_audit.pii_values import normalize  # noqa: E402
from dataset_audit.regex_detector import RegexDetector  # noqa: E402

CASES = [json.loads(l) for l in (ROOT / "tests" / "fixtures" / "pii_hard_cases.jsonl").open(encoding="utf-8") if l.strip()]
PATTERNS = {p.label: p for p in RegexDetector(ROOT / "configs" / "pii_patterns_v2.yaml").patterns}


@pytest.mark.parametrize("case", [c for c in CASES if c["scored"]], ids=lambda c: c["id"])
def test_v2_hard_case(case):
    got = {normalize(case["type"], v) for v in PATTERNS[case["type"]].find(case["text"])} - {""}
    assert got == {normalize(case["type"], e) for e in case["expected"]}


def test_v2_changes_only_declared_patterns():
    v1 = {p.name: p for p in RegexDetector(ROOT / "configs" / "pii_patterns.yaml").patterns}
    v2 = RegexDetector(ROOT / "configs" / "pii_patterns_v2.yaml")
    changed = {p.name for p in v2.patterns if p.regex.pattern != v1[p.name].regex.pattern}
    assert [p.name for p in v2.patterns] == list(v1)
    assert changed == set(v2.modified_patterns) == {"email", "phone"}


def test_default_audit_config_uses_v2():
    import yaml
    assert yaml.safe_load((ROOT / "configs" / "pii_audit.yaml").read_text())["patterns_file"] == "configs/pii_patterns_v2.yaml"


@pytest.mark.parametrize("a, b", [
    ("+44 (0) 20 7946 0958", "+44 20 7946 0958"),
    ("+44 20 7946 0958", "011 44 20 7946 0958".removeprefix("011 ")),
    ("+1 (713) 555-0192", "713-555-0192"),
])
def test_v2_phone_normalization(a, b):
    assert normalize("Phone", a) == normalize("Phone", b)


def test_v2_methodology_marks_replaced_patterns():
    from dataset_audit.reproducibility import methodology_for
    rows = {r["setting"]: r for r in methodology_for(RegexDetector(ROOT / "configs" / "pii_patterns_v2.yaml"))}
    assert rows["Regex patterns"]["value"].startswith("verbatim for 18 patterns")
    assert "replaced" in rows["Regex pattern 'email'"]["value"]
    assert "implementation decision" in rows["Regex pattern 'phone'"]["provenance"]
