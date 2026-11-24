"""Measure PII prevalence in a dataset, following Kaneko & Baldwin (2025), Section 2.1.

Usage:
  python scripts/audit_pii.py --dataset <hub id or local path> \
      --text-column input output --output-dir outputs/dataset_audit/<name>
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_audit.dataset_loader import DatasetSpec, describe_dataset, iter_indexed_texts, reservoir_sample  # noqa: E402
from dataset_audit.metrics import PERSON_LABEL, PIIStatistics  # noqa: E402
from dataset_audit.ner_detector import PersonNameDetector  # noqa: E402
from dataset_audit.pii_values import PIIValueCollector  # noqa: E402
from dataset_audit.regex_detector import RegexDetector  # noqa: E402
from dataset_audit.reproducibility import build_record  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Hugging Face dataset ID, or a local file, directory or glob")
    p.add_argument("--text-column", nargs="+", required=True, help="Columns to analyze; dots enter nested fields")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--split", default="train")
    p.add_argument("--config-name", help="Hugging Face dataset configuration")
    p.add_argument("--revision", help="Hugging Face dataset revision")
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sample-size", type=int, help="Analyze a uniform random sample of this many instances")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "pii_audit.yaml")
    p.add_argument("--spacy-model")
    p.add_argument("--ner-labels", nargs="+")
    p.add_argument("--ner-batch-size", type=int)
    p.add_argument("--n-process", type=int)
    p.add_argument(
        "--dump-pii-values",
        action="store_true",
        help="Also write the detected values to pii_values.jsonl.gz (contains raw PII)",
    )
    return p.parse_args(argv)


def load_settings(args):
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    overrides = {
        "model": args.spacy_model,
        "labels": args.ner_labels,
        "batch_size": args.ner_batch_size,
        "n_process": args.n_process,
    }
    settings["ner"].update({k: v for k, v in overrides.items() if v is not None})
    patterns_file = Path(settings["patterns_file"])
    settings["patterns_file"] = patterns_file if patterns_file.is_absolute() else ROOT / patterns_file
    return settings


def text_stream(spec, skipped, args):
    """Return (iterable of (record index, text), count of analyzable texts before sampling or None)."""
    texts = iter_indexed_texts(spec, skipped)
    if args.sample_size is None:
        return texts, None
    sample, seen = reservoir_sample(texts, args.sample_size, args.seed)
    return sample, seen


def analyze(texts, regex, ner, stats, values=None, total=None):
    def items():
        for index, text in texts:
            n_bytes = len(text.encode("utf-8", "surrogatepass"))
            yield text, (index, regex.find(text), n_bytes)

    results = ner.iter_entities(items())
    for (index, found, n_bytes), names in tqdm(results, total=total, unit="inst", file=sys.stderr):
        if names:
            found[PERSON_LABEL] = names
        stats.add({label: len(v) for label, v in found.items()}, n_bytes)
        if values is not None:
            values.add(index, found)


def print_summary(report):
    s = report["dataset_summary"]
    print("\n=== PII Analysis ===")
    print(f"Total Instances:                 {s['total_instances']:,}")
    print(f"PII-containing Instances:        {s['pii_containing_instances']:,}")
    print(f"PII-containing Instance Rate:    {s['pii_containing_instance_rate_percent']}%")
    print(f"PII-containing Data Size Rate:   {s['pii_containing_data_size_rate_percent']}%")
    print(f"Skipped records:                 {s['skipped_records']}")

    print("\nType                  Detections   Instances")
    for label, row in report["pii_type_distribution"].items():
        if row["total_detections"]:
            print(f"{label:<20}{row['total_detections']:>12,}{row['pii_containing_instances']:>12,}")

    co = report["pii_co_occurrence"]
    print(f"\nInstances with 2+ PII types: {co['instances_with_multiple_pii_types']:,}")
    for name, n in co["key_combinations_at_least_these_types"].items():
        print(f"  {name} (at least): {n:,}")
    for row in co["exact_type_combinations"][:5]:
        print(f"  {row['types']} (exact): {row['instances']:,}")


def main(argv=None):
    args = parse_args(argv)
    settings = load_settings(args)

    spec = DatasetSpec(
        source=args.dataset,
        text_columns=tuple(args.text_column),
        split=args.split,
        config_name=args.config_name,
        revision=args.revision,
        streaming=args.streaming,
    )
    regex = RegexDetector(settings["patterns_file"])
    ner_settings = settings["ner"]
    ner = PersonNameDetector(
        ner_settings["model"],
        labels=ner_settings["labels"],
        batch_size=ner_settings["batch_size"],
        n_process=ner_settings["n_process"],
        max_chunk_chars=ner_settings["max_chunk_chars"],
    )
    stats = PIIStatistics(regex.labels, settings["key_combinations"])
    dataset_info = describe_dataset(spec)

    skipped = collections.Counter({"null": 0, "empty": 0, "invalid": 0})
    texts, population = text_stream(spec, skipped, args)
    values = PIIValueCollector() if args.dump_pii_values else None
    analyze(texts, regex, ner, stats, values, total=len(texts) if args.sample_size is not None else None)

    report = {
        "dataset_summary": {**stats.summary(), "skipped_records": dict(skipped)},
        "pii_type_distribution": stats.type_distribution(),
        "pii_co_occurrence": stats.co_occurrence(),
        "dataset": dataset_info,
        "reproducibility": build_record(
            regex,
            ner,
            settings["patterns_file"],
            sampling={
                "sample_size": args.sample_size,
                "seed": args.seed if args.sample_size is not None else None,
                "analyzable_instances_before_sampling": population,
            },
            ner_settings=ner_settings,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if values is not None:
        values_file = "pii_values.jsonl.gz"
        values.write(args.output_dir / values_file)
        report["pii_values"] = values.metadata(values_file)
    output = args.output_dir / "pii_analysis.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"\nSaved: {output}")
    if values is not None:
        print(f"Saved: {args.output_dir / report['pii_values']['file']}")


if __name__ == "__main__":
    main()
