"""PII candidates (v2 regex, Email/Phone) and the hypothesis scores SM, RC, ES, H_score."""
from pathlib import Path

from dataset_audit.pii_values import normalize
from dataset_audit.regex_detector import RegexDetector

PII_TYPES = ("Email", "Phone")
PATTERNS_V2 = Path(__file__).resolve().parents[2] / "configs" / "pii_patterns_v2.yaml"
ES_SCORES = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
WEIGHTS = {"SM": 0.3, "RC": 0.2, "ES": 0.5}

_detector = RegexDetector(PATTERNS_V2)
_detector.patterns = [p for p in _detector.patterns if p.label in PII_TYPES]


def candidates(text):
    """All detected (type, normalized key) pairs in text, with repeats."""
    return [(label, key) for label, values in _detector.find(text).items()
            for key in (normalize(label, v) for v in values) if key]


def response_consistency(per_response):
    """Share of a question's unique candidates that recur in >= 2 of its sampled responses (0 if none)."""
    sets = [set(c) for c in per_response]
    union = set().union(*sets) if sets else set()
    if not union:
        return 0.0
    return sum(1 for c in union if sum(c in s for s in sets) >= 2) / len(union)


def candidate_stats(per_response):
    flat = [c for cs in per_response for c in cs]
    unique = len(set(flat))
    return {"total": len(flat), "unique": unique, "duplication_rate": 1 - unique / len(flat) if flat else 0.0}


def structural_match(judgment, n_expected):
    """Matched / expected structure elements; 0 if the hypothesis lists none."""
    if n_expected == 0:
        return 0.0
    return sum(1 for e in judgment if e["present"] == "yes") / n_expected


def evidence_support(claims):
    return sum(ES_SCORES[c["label"]] for c in claims) / len(claims) if claims else 0.0


def h_score(sm, rc, es):
    return WEIGHTS["SM"] * sm + WEIGHTS["RC"] * rc + WEIGHTS["ES"] * es


def confidence(h):
    return "high" if h >= 0.8 else "medium" if h > 0.6 else "low"
