"""Record what a run used and where each methodological choice came from."""
import hashlib
import importlib.metadata
import platform
from pathlib import Path

PAPER = "Kaneko & Baldwin (2025), Section 2.1"
SUBRAMANI = "Subramani et al. (2023), Supplementary Materials"
DECISION = "implementation decision (not specified by the papers)"

SIZE_MEASURE = "UTF-8 byte length of the analyzed text of each instance"

METHODOLOGY = [
    {"setting": "PII types", "value": "20 regex types plus person names", "provenance": PAPER},
    {"setting": "Regex patterns", "value": "verbatim; PDF line wraps removed", "provenance": SUBRAMANI},
    {"setting": "Person names", "value": "spaCy NER, no real-name/pseudonym distinction", "provenance": PAPER},
    {"setting": "Instance decision", "value": "PII-containing if at least one detection", "provenance": PAPER},
    {"setting": "Minimum text length", "value": "none", "provenance": PAPER},
    {"setting": "Primary metric", "value": "PII-containing size / total size", "provenance": PAPER},
    {
        "setting": "Primary metric wording",
        "value": "The paper also states an instance-count ratio per dataset; this implementation "
        "reports the size-based ratio as primary and the instance ratio as additional.",
        "provenance": DECISION,
    },
    {"setting": "Data size unit", "value": SIZE_MEASURE, "provenance": DECISION},
    {"setting": "spaCy model and version", "value": "see ner_model", "provenance": DECISION},
    {
        "setting": "Regex engine and flags",
        "value": "Python re, default flags. The email pattern is therefore lowercase-only.",
        "provenance": DECISION,
    },
    {
        "setting": "Detection counting",
        "value": "non-overlapping matches per pattern; patterns run independently, "
        "so one digit string can count under several types",
        "provenance": DECISION,
    },
    {"setting": "Multiple text columns", "value": "joined with a newline into one instance", "provenance": DECISION},
    {
        "setting": "Null, empty, whitespace-only or non-text fields",
        "value": "excluded from all totals and tallied in skipped_records",
        "provenance": DECISION,
    },
    {
        "setting": "Very long texts",
        "value": "split for NER at whitespace near max_chunk_chars; regex sees the whole text",
        "provenance": DECISION,
    },
    {"setting": "Text cleaning and deduplication", "value": "none", "provenance": DECISION},
    {"setting": "Sampling", "value": "reservoir sampling over analyzable texts", "provenance": DECISION},
]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library_versions(names=("spacy", "datasets", "huggingface_hub", "pyarrow", "ijson", "PyYAML")):
    return {name: importlib.metadata.version(name) for name in names}


def methodology_for(detector):
    """METHODOLOGY with the regex rows matching the patterns file actually used."""
    if not detector.modified_patterns:
        return METHODOLOGY
    rows = []
    for row in METHODOLOGY:
        if row["setting"] == "Regex patterns":
            kept = [p.name for p in detector.patterns if p.name not in detector.modified_patterns]
            rows.append({**row, "value": f"verbatim for {len(kept)} patterns; PDF line wraps removed"})
            rows.extend(
                {"setting": f"Regex pattern '{name}'", "value": f"replaced: {why}", "provenance": DECISION}
                for name, why in detector.modified_patterns.items()
            )
        elif row["setting"] == "Regex engine and flags":
            rows.append({**row, "value": "Python re, default flags (the replaced email pattern is case-insensitive by construction)"})
        else:
            rows.append(row)
    return rows


def build_record(detector, ner, patterns_file, sampling, ner_settings):
    return {
        "python_version": platform.python_version(),
        "libraries": library_versions(),
        "regex": {
            **detector.source,
            "patterns_file": Path(patterns_file).name,
            "patterns_file_sha256": file_sha256(patterns_file),
            "modified_patterns": detector.modified_patterns,
        },
        "ner_model": ner.model_info,
        "ner_settings": ner_settings,
        "sampling": sampling,
        "data_size_measure": SIZE_MEASURE,
        "methodology": methodology_for(detector),
    }
