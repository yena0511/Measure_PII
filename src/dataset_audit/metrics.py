"""Aggregate instance-level PII detections. Holds counts only, never text or PII values."""
from collections import Counter

PERSON_LABEL = "Name"


def _percent(part, whole):
    return round(100 * part / whole, 6) if whole else None


class PIIStatistics:
    def __init__(self, regex_labels, key_combinations=()):
        self.type_order = [PERSON_LABEL, *regex_labels]
        self.rank = {label: i for i, label in enumerate(self.type_order)}
        self.key_combinations = [tuple(c) for c in key_combinations]
        self.total_instances = 0
        self.pii_instances = 0
        self.total_bytes = 0
        self.pii_bytes = 0
        self.detections = Counter()
        self.instances_by_type = Counter()
        self.exact_combinations = Counter()
        self.key_combination_counts = Counter()

    def add(self, counts, n_bytes):
        """Record one instance from its {label: count} detections and its size."""
        self.total_instances += 1
        self.total_bytes += n_bytes
        if not counts:
            return
        self.pii_instances += 1
        self.pii_bytes += n_bytes
        self.detections.update(counts)
        self.instances_by_type.update(counts.keys())
        if len(counts) >= 2:
            self.exact_combinations[tuple(sorted(counts, key=self.rank.__getitem__))] += 1
        for combination in self.key_combinations:
            if all(label in counts for label in combination):
                self.key_combination_counts[combination] += 1

    def summary(self):
        return {
            "total_instances": self.total_instances,
            "pii_containing_instances": self.pii_instances,
            "pii_containing_instance_rate_percent": _percent(self.pii_instances, self.total_instances),
            "pii_containing_data_size_rate_percent": _percent(self.pii_bytes, self.total_bytes),
            "total_data_size_bytes": self.total_bytes,
            "pii_containing_data_size_bytes": self.pii_bytes,
        }

    def type_distribution(self):
        return {
            label: {
                "total_detections": self.detections[label],
                "pii_containing_instances": self.instances_by_type[label],
                "instance_rate_percent": _percent(self.instances_by_type[label], self.total_instances),
            }
            for label in self.type_order
        }

    def co_occurrence(self):
        return {
            "instances_with_multiple_pii_types": sum(self.exact_combinations.values()),
            "key_combinations_at_least_these_types": {
                " + ".join(c): self.key_combination_counts[c] for c in self.key_combinations
            },
            "exact_type_combinations": [
                {"types": " + ".join(c), "instances": n} for c, n in self.exact_combinations.most_common()
            ],
        }
