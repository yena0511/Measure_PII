"""Build the final PII ground-truth JSON from a reviewed pii_review.csv.

Usage:
  python scripts/build_ground_truth.py --audit-dir outputs/dataset_audit/healthcare/ultramedical

Rows are matched to pii_values.jsonl.gz by review_id, not by the CSV's value/key columns,
because spreadsheet programs often rewrite long numbers (e.g. 3.0087E+13, dropped leading 0).
decision: 1 / keep / yes = PII, 0 / drop / no = not PII,
2 / fix = PII after correction: the corrected value is written in `note` (e.g. the regex cut
off part of the value). The row's key becomes normalize(type, note) and the originally
detected key is kept in `aliases`, since the same detector will cut generated text the same way.
Empty = undecided (excluded, reported).
"""
import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_audit.pii_values import normalize, read_pii_values  # noqa: E402

KEEP = {"1", "keep", "yes", "y", "true", "o"}
DROP = {"0", "drop", "no", "n", "false", "x"}
FIX = {"2", "fix"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-dir", required=True, type=Path)
    p.add_argument("--review", type=Path, help="default: <audit-dir>/review/pii_review.csv")
    p.add_argument("--types", nargs="+", help="types the review CSV was built with (default: all except Name)")
    p.add_argument("--output", type=Path, help="default: <audit-dir>/pii_ground_truth.json")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    review_path = args.review or args.audit_dir / "review" / "pii_review.csv"
    report = json.loads((args.audit_dir / "pii_analysis.json").read_text(encoding="utf-8"))
    meta = report["pii_values"]

    # Same filter and order as make_pii_review.py, so review_id N is the N-th row.
    values = [
        r for r in read_pii_values(args.audit_dir / meta["file"])
        if (r["type"] in args.types if args.types else r["type"] != "Name")
    ]
    with review_path.open(encoding="utf-8-sig", newline="") as f:
        reviewed = list(csv.DictReader(f))

    kept, dropped, undecided, fixed = [], [], [], []
    for row in reviewed:
        n = int(row["review_id"].rsplit("-", 1)[1])
        v = values[n - 1]
        if v["type"] != row["type"] or str(v["count"]) != row["count"].strip():
            sys.exit(f"{row['review_id']}: does not match pii_values (was the CSV rebuilt with other --types?)")
        decision = row["decision"].strip().lower()
        if decision in KEEP:
            kept.append({"review_id": row["review_id"], **v, "note": row.get("note", "").strip()})
        elif decision in FIX:
            corrected = row.get("note", "").strip()
            if not corrected:
                sys.exit(f"{row['review_id']}: decision 2 needs the corrected value in note")
            kept.append({
                "review_id": row["review_id"], **v,
                "key": normalize(v["type"], corrected),
                "corrected_value": corrected,
                "aliases": [v["key"]],
                "note": "corrected in review",
            })
            fixed.append(row["review_id"])
        elif decision in DROP:
            dropped.append(row["review_id"])
        elif decision:
            sys.exit(f"{row['review_id']}: unknown decision {row['decision']!r}")
        else:
            undecided.append(row["review_id"])

    types = sorted({v["type"] for v in values})
    counts = {t: sum(1 for k in kept if k["type"] == t) for t in types}
    out = {
        "dataset": report["dataset"],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "pii_values": str(args.audit_dir / meta["file"]),
            "review_file": str(review_path),
            "normalization_version": meta["normalization_version"],
            "normalization_rules": meta["normalization_rules"],
            "match_on": "key or aliases: apply dataset_audit.pii_values.normalize(type, value) to detected values",
        },
        "review": {
            "reviewed_types": types,
            "not_reviewed_types": sorted(set(meta["unique_values_per_type"]) - set(types)),
            "total": len(reviewed),
            "kept": len(kept),
            "corrected": fixed,
            "dropped": len(dropped),
            "undecided": undecided,
        },
        "pii_count_by_type": {t: c for t, c in counts.items() if c},
        "pii": kept,
    }
    output = args.output or args.audit_dir / "pii_ground_truth.json"
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"kept {len(kept)} (corrected {len(fixed)}), dropped {len(dropped)}, undecided {len(undecided)} -> {output}")
    print(out["pii_count_by_type"])


if __name__ == "__main__":
    main()
