"""Build a CSV for manually reviewing the values in pii_values.jsonl.gz, with context.

Usage:
  python scripts/make_pii_review.py --audit-dir outputs/dataset_audit/healthcare/ultramedical

Reads the dataset location from <audit-dir>/pii_analysis.json, re-reads only the records
listed in instance_ids, and writes <audit-dir>/review/pii_review.csv. Fill the `decision`
column with keep or drop (and an optional `note`); nothing is removed automatically.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_audit.dataset_loader import DatasetSpec, extract_text, iter_records  # noqa: E402
from dataset_audit.pii_values import read_pii_values  # noqa: E402

NEWLINE_MARK = " ⏎ "


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-dir", required=True, type=Path)
    p.add_argument("--types", nargs="+", help="PII types to include (default: all except Name)")
    p.add_argument("--contexts", type=int, default=3, help="contexts per value")
    p.add_argument("--window", type=int, default=150, help="characters on each side of the value")
    p.add_argument("--output", type=Path)
    return p.parse_args(argv)


def snippet(text, surface_forms, window):
    """Return the text around the first occurrence of any surface form, value in 【】."""
    hits = [(text.find(f), f) for f in surface_forms if f and f in text]
    if not hits:
        return ""
    start, form = min(hits)
    end = start + len(form)
    left = text[max(0, start - window):start]
    right = text[end:end + window]
    s = ("…" if start > window else "") + left + "【" + form + "】" + right + ("…" if end + window < len(text) else "")
    return s.replace("\r", "").replace("\n", NEWLINE_MARK)


def main(argv=None):
    args = parse_args(argv)
    report = json.loads((args.audit_dir / "pii_analysis.json").read_text(encoding="utf-8"))
    values_meta = report.get("pii_values")
    if not values_meta:
        sys.exit("pii_analysis.json has no pii_values section; rerun audit_pii.py with --dump-pii-values")

    rows = [
        r for r in read_pii_values(args.audit_dir / values_meta["file"])
        if (r["type"] in args.types if args.types else r["type"] != "Name")
    ]

    wanted = {i for r in rows for i in r["instance_ids"][: args.contexts]}
    ds = report["dataset"]
    spec = DatasetSpec(
        source=ds["source"] if Path(ds["source"]).is_absolute() else str(ROOT / ds["source"]),
        text_columns=tuple(ds["text_columns"]),
        split=ds.get("split") or "train",
        config_name=ds.get("config_name"),
        revision=(ds.get("origin") or {}).get("revision") if ds.get("type") != "local" else None,
    )
    texts = {}
    for index, record in enumerate(iter_records(spec)):
        if index in wanted:
            texts[index], _ = extract_text(record, spec.text_columns)
            if len(texts) == len(wanted):
                break

    out = args.output or args.audit_dir / "review" / "pii_review.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["review_id", "type", "value", "decision", "note", "count", "n_instances"]
    header += [f"context_{i + 1}" for i in range(args.contexts)]
    header += ["key", "surface_forms", "instance_ids"]
    # utf-8-sig so that Excel detects the encoding.
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for n, r in enumerate(rows, 1):
            forms = list(r["surface_forms"])
            contexts = [snippet(texts.get(i) or "", forms, args.window) for i in r["instance_ids"][: args.contexts]]
            contexts += [""] * (args.contexts - len(contexts))
            w.writerow(
                [f"{r['type'][:2].upper()}-{n:05d}", r["type"], forms[0].strip(), "", "", r["count"], r["n_instances"]]
                + contexts
                + [r["key"], " | ".join(forms), " ".join(map(str, r["instance_ids"]))]
            )
    print(f"{len(rows)} values -> {out}")


if __name__ == "__main__":
    main()
