"""Per-response PII report for a hypothesis-probe run: does each response contain PII (v2 rules),
and is that PII in the fine-tuning dataset's ground-truth PII list?

Usage:
  python scripts/make_probe_pii_report.py \
      --probe-dir outputs/hypothesis_probe/BJyotibrat__Llama-3-8B-Instruct-BotMed-v1 \
      --ground-truth outputs/dataset_audit/healthcare/ai_medical_chatbot/pii_ground_truth.json

--ground-truth takes a reviewed pii_ground_truth.json (scripts/build_ground_truth.py) or, before review,
the raw pii_values.jsonl.gz (scripts/extract_regex_pii.py / audit_pii.py --dump-pii-values).
Without it, the in_ground_truth column is "-".

Output <probe-dir>/pii_report.csv, one row per target response:
  id               policy/round/hypothesis/question/sample, e.g. ucb1/R1/H2/Q1/S3
  question         the probe question
  response         the target model's reply
  pii              PII in the reply, v2 rules + v2 normalization ("Email: a@b.com; Phone: 7135550192"), empty if none
  in_ground_truth  the part of `pii` found in the ground truth, empty if none
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_audit.pii_values import read_pii_values  # noqa: E402
from hypothesis_probe.metrics import PII_TYPES, candidates  # noqa: E402  (v2 Email/Phone + v2 normalize)

COLUMNS = ["id", "question", "response", "pii", "in_ground_truth"]


def load_ground_truth(path):
    """{(type, key)} from pii_ground_truth.json (key + aliases) or pii_values.jsonl.gz (key)."""
    if path.name.endswith(".json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = {(p["type"], k) for p in data["pii"] for k in [p["key"], *p.get("aliases", [])]}
        reviewed = True
    else:
        keys = {(p["type"], p["key"]) for p in read_pii_values(path)}
        reviewed = False
    return {k for k in keys if k[0] in PII_TYPES}, reviewed


def fmt(pairs):
    return "; ".join(f"{t}: {k}" for t, k in pairs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-dir", required=True, type=Path)
    p.add_argument("--ground-truth", type=Path)
    p.add_argument("--output", type=Path, help="default: <probe-dir>/pii_report.csv")
    args = p.parse_args(argv)

    gt, reviewed = load_ground_truth(args.ground_truth) if args.ground_truth else (None, False)
    rows, mismatched = [], 0
    for policy_dir in sorted(d for d in args.probe_dir.iterdir() if d.is_dir()):
        rounds = sorted(policy_dir.glob("round_*.json"), key=lambda f: int(f.stem.split("_")[1]))
        for rf in rounds:
            rec = json.loads(rf.read_text(encoding="utf-8"))
            for res in rec["results"]:
                for qi, q in enumerate(res["responses"], 1):
                    for si, s in enumerate(q["samples"], 1):
                        found = sorted(set(candidates(s["text"])))  # re-detected with v2, not read from the run
                        mismatched += found != sorted({tuple(c) for c in s["candidates"]})
                        hit = [c for c in found if c in gt] if gt is not None else None
                        rows.append({"id": f"{rec['policy']}/R{rec['round']}/{res['arm']}/Q{qi}/S{si}",
                                     "question": q["question"], "response": s["text"], "pii": fmt(found),
                                     "in_ground_truth": "-" if hit is None else fmt(hit),
                                     "_found": found, "_hit": hit or []})

    out = args.output or args.probe_dir / "pii_report.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel opens UTF-8 correctly
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Summary (console only; everything in it can be recomputed from the CSV)
    print(f"{out}  ({len(rows)} responses)")
    print(f"  ground truth: {args.ground_truth or 'none'}"
          + ("" if gt is None else f" [{'reviewed' if reviewed else 'NOT reviewed'}, {len(gt)} Email/Phone values]"))
    for policy in sorted({r["id"].split("/")[0] for r in rows}):
        rs = [r for r in rows if r["id"].startswith(policy + "/")]
        uniq = {c for r in rs for c in r["_found"]}
        line = (f"  {policy}: responses with PII {sum(bool(r['_found']) for r in rs)}/{len(rs)}, "
                f"unique PII {len(uniq)}")
        if gt is not None:
            uhit = {c for r in rs for c in r["_hit"]}
            line += f" | responses with ground-truth PII {sum(bool(r['_hit']) for r in rs)}, unique ground-truth PII {len(uhit)}"
        print(line)
    if mismatched:
        print(f"  WARNING: {mismatched} responses where v2 re-detection differs from the candidates stored by the run")


if __name__ == "__main__":
    main()
