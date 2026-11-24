"""Collect PII values with selected Measure_PII regex types only (no NER).

Same detectors and normalization as audit_pii.py --dump-pii-values, restricted to --types.

Usage:
  python scripts/extract_regex_pii.py --input datasets/email/enron_cmu/pi_split/train.jsonl [--patterns configs/pii_patterns.yaml] \
      --types Email Phone --output-dir outputs/dataset_audit/email/enron_pi_train
"""
import argparse
import collections
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_audit.pii_values import PIIValueCollector  # noqa: E402
from dataset_audit.regex_detector import RegexDetector  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="JSON lines with a 'text' field")
    p.add_argument("--types", nargs="+", required=True, help="PII labels to detect, e.g. Email Phone")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--patterns", type=Path, default=ROOT / "configs" / "pii_patterns_v2.yaml")
    args = p.parse_args()

    regex = RegexDetector(args.patterns)
    regex.patterns = [pat for pat in regex.patterns if pat.label in args.types]
    missing = set(args.types) - {pat.label for pat in regex.patterns}
    if missing:
        sys.exit(f"unknown types: {sorted(missing)}")

    values = PIIValueCollector()
    detections = collections.Counter()
    instances = collections.Counter()
    n = 0
    with args.input.open(encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, unit="inst")):
            found = regex.find(json.loads(line)["text"])
            values.add(i, found)
            for label, vals in found.items():
                detections[label] += len(vals)
                instances[label] += 1
            n += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    values.write(args.output_dir / "pii_values.jsonl.gz")
    report = {
        "input": str(args.input),
        "types": args.types,
        "patterns": str(args.patterns),
        "ner": None,
        "total_instances": n,
        "per_type": {t: {"total_detections": detections[t], "instances": instances[t]} for t in args.types},
        "pii_values": values.metadata("pii_values.jsonl.gz"),
    }
    (args.output_dir / "pii_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("total_instances", "per_type")}, indent=2))
    print("unique:", report["pii_values"]["unique_values_per_type"])


if __name__ == "__main__":
    main()
