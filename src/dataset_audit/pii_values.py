"""Collect the detected PII values themselves, as ground truth for extraction evaluation.

Each unique (type, key) becomes one JSON line. `key` is the normalized value used for
matching; evaluation code must build keys from generated text with `normalize()` from
this module, so that both sides go through the same rules.
"""
import collections
import gzip
import io
import json
import re
import unicodedata

NORMALIZATION_VERSION = "2"

_WS = re.compile(r"\s+")
_NON_DIGIT = re.compile(r"\D")
_SEPARATORS = re.compile(r"[\s\-]")

NORMALIZATION_RULES = {
    "Email": "strip, lowercase",
    "Phone": "a '(0)' trunk prefix after '+CC' is dropped; digits only; a leading US country code 1 on 11 digits is dropped",
    "IP Address": "strip",
    "Name": "NFKC, collapse whitespace, strip, casefold",
    "other types (IBAN, US SSN, cards)": "remove spaces and hyphens, uppercase",
}


def normalize(label, value):
    """Return the matching key for a detected value of type `label`."""
    if label == "Email":
        return value.strip().lower()
    if label == "Phone":
        if value.lstrip().startswith("+"):
            value = value.replace("(0)", "")  # +44 (0) 20 ... : the 0 is dialled only inside the country
        digits = _NON_DIGIT.sub("", value)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits
    if label == "IP Address":
        return value.strip()
    if label == "Name":
        return _WS.sub(" ", unicodedata.normalize("NFKC", value)).strip().casefold()
    return _SEPARATORS.sub("", value).upper()


class PIIValueCollector:
    def __init__(self, max_instance_ids=100, max_surface_forms=20):
        self.max_instance_ids = max_instance_ids
        self.max_surface_forms = max_surface_forms
        self.entries = {}

    def add(self, instance_id, found):
        """Record {label: [values]} detected in one instance."""
        for label, values in found.items():
            seen_here = set()
            for value in values:
                key = normalize(label, value)
                if not key:
                    continue
                entry = self.entries.get((label, key))
                if entry is None:
                    entry = self.entries[(label, key)] = {
                        "count": 0,
                        "n_instances": 0,
                        "instance_ids": [],
                        "surface_forms": collections.Counter(),
                    }
                entry["count"] += 1
                entry["surface_forms"][value] += 1
                if key not in seen_here:
                    seen_here.add(key)
                    entry["n_instances"] += 1
                    if len(entry["instance_ids"]) < self.max_instance_ids:
                        entry["instance_ids"].append(instance_id)

    def rows(self):
        """Yield rows sorted by type, then by count (descending), then key."""
        ordered = sorted(self.entries.items(), key=lambda kv: (kv[0][0], -kv[1]["count"], kv[0][1]))
        for (label, key), e in ordered:
            yield {
                "type": label,
                "key": key,
                "count": e["count"],
                "n_instances": e["n_instances"],
                "surface_forms": dict(e["surface_forms"].most_common(self.max_surface_forms)),
                "instance_ids": sorted(e["instance_ids"]),
            }

    def unique_per_type(self):
        return dict(sorted(collections.Counter(label for label, _ in self.entries).items()))

    def write(self, path):
        """Write gzipped JSON lines; mtime is fixed so identical input gives an identical file."""
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8") as f:
                for row in self.rows():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def metadata(self, filename):
        return {
            "file": filename,
            "format": "gzip JSON lines; one row per unique (type, key)",
            "fields": {
                "type": "PII type, same labels as pii_type_distribution",
                "key": "normalized value used for matching",
                "count": "total detections",
                "n_instances": "instances containing the value",
                "surface_forms": f"raw detected strings with counts, top {self.max_surface_forms}",
                "instance_ids": f"0-based record indices in the source stream, first {self.max_instance_ids}",
            },
            "normalization_version": NORMALIZATION_VERSION,
            "normalization_rules": NORMALIZATION_RULES,
            "unique_values_per_type": self.unique_per_type(),
        }


def read_pii_values(path):
    """Yield rows from a pii_values.jsonl.gz file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
