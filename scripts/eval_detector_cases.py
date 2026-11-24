"""Score detector variants on the hand-written hard cases (synthetic values only).

A case passes when the set of normalized keys the detector emits for the case's type equals `expected`.
Cases with "scored": false are reported but left out of the totals.

Usage:
  python scripts/eval_detector_cases.py [--cases tests/fixtures/pii_hard_cases.jsonl] [--variants measure_v1 measure_c2 measure_v2 pi]
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_audit.pii_values import normalize  # noqa: E402
from dataset_audit.regex_detector import RegexDetector  # noqa: E402

# Candidate email pattern from the Enron comparison (not yet adopted into configs/).
EMAIL_C2 = re.compile(
    r"(?<![A-Za-z0-9._%+\-$@])[A-Za-z0-9](?:[A-Za-z0-9._+\-]*[A-Za-z0-9_])?"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}(?![A-Za-z0-9\-])"
)
# PrivateInvestigator pii_collector.py, verbatim.
PI_EMAIL = re.compile(r"(([\w\-]+\.)*[\w\-]+@([\w\-]+\.)+[\w\-]+)")
PI_PHONE = re.compile(r"(?=(\d{3}[ .-]\d{3}[ .-]\d{4}|\(\d{3}\)[ .-]\d{3}[ .-]\d{4}))")


def build_variants():
    v1 = RegexDetector(ROOT / "configs" / "pii_patterns.yaml")
    v1_by_label = {p.label: p for p in v1.patterns}

    def measure_v1(label, text):
        return list(v1_by_label[label].find(text))

    v2 = RegexDetector(ROOT / "configs" / "pii_patterns_v2.yaml")
    v2_by_label = {p.label: p for p in v2.patterns}

    def measure_v2(label, text):
        return list(v2_by_label[label].find(text))

    def measure_c2(label, text):
        if label == "Email":
            return [m.group(0) for m in EMAIL_C2.finditer(text)]
        return measure_v1(label, text)

    def pi(label, text):
        if label == "Email":
            return [m[0] for m in PI_EMAIL.findall(text) if m[0]]
        return PI_PHONE.findall(text)

    return {"measure_v1": measure_v1, "measure_c2": measure_c2, "measure_v2": measure_v2, "pi": pi}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=ROOT / "tests" / "fixtures" / "pii_hard_cases.jsonl")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--show", choices=["fail", "all", "none"], default="fail")
    args = p.parse_args()

    cases = [json.loads(line) for line in args.cases.open(encoding="utf-8") if line.strip()]
    variants = build_variants()
    names = args.variants or list(variants)

    rows = {}
    for name in names:
        detect = variants[name]
        per_cat = collections.defaultdict(lambda: [0, 0])
        tp = fp = fn = 0
        results = []
        for c in cases:
            expected = {normalize(c["type"], e) for e in c["expected"]}
            got = {normalize(c["type"], v) for v in detect(c["type"], c["text"])} - {""}
            ok = got == expected
            results.append((c, got, expected, ok))
            if not c["scored"]:
                continue
            key = (c["type"], c["category"])
            per_cat[key][0] += ok
            per_cat[key][1] += 1
            tp += len(got & expected)
            fp += len(got - expected)
            fn += len(expected - got)
        rows[name] = (per_cat, tp, fp, fn, results)

    cats = sorted({k for r in rows.values() for k in r[0]})
    width = max(len(f"{t}/{c}") for t, c in cats) + 2
    print(f"{'type/category':<{width}}" + "".join(f"{n:>14}" for n in names))
    for k in cats:
        print(f"{k[0] + '/' + k[1]:<{width}}" + "".join(f"{rows[n][0][k][0]:>8}/{rows[n][0][k][1]:<5}" for n in names))
    for label in ("Email", "Phone"):
        line = f"{label + ' cases passed':<{width}}"
        for n in names:
            ok = sum(v[0] for k, v in rows[n][0].items() if k[0] == label)
            tot = sum(v[1] for k, v in rows[n][0].items() if k[0] == label)
            line += f"{ok:>8}/{tot:<5}"
        print(line)
    print(f"{'values TP/FP/FN':<{width}}" + "".join(f"{f'{rows[n][1]}/{rows[n][2]}/{rows[n][3]}':>14}" for n in names))

    if args.show != "none":
        for n in names:
            print(f"\n## {n}")
            for c, got, expected, ok in rows[n][4]:
                if args.show == "fail" and ok and c["scored"]:
                    continue
                tag = "PASS" if ok else "FAIL"
                if not c["scored"]:
                    tag = "info"
                print(f"  [{tag}] {c['id']} {c['category']:<17} got={sorted(got)} expected={sorted(expected)}  | {c['text']!r}")


if __name__ == "__main__":
    main()
