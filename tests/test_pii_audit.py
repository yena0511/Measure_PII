"""Tests on synthetic data only. All numbers and names below are made up."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_audit import dataset_loader as dl  # noqa: E402
from dataset_audit.metrics import PIIStatistics  # noqa: E402
from dataset_audit.ner_detector import PersonNameDetector, split_text  # noqa: E402
from dataset_audit.regex_detector import RegexDetector  # noqa: E402

PATTERNS = ROOT / "configs" / "pii_patterns.yaml"

POSITIVES = {
    "IP Address": "host 192.0.2.1 down",
    "IBAN": "pay GB82WEST12345698765432 now",
    "US SSN": "ssn 123-45-6789 on file",
    "Email": "write to jane.doe@example.com today",
    "Phone": "call (555) 555-0100 now",
    "American Express": "378282246310005",
    "BCGlobal": "6541123456789012",
    "Carte Blanche": "38912345678901",
    "Diners Club": "30569309025904",
    "Discover": "6011111111111117",
    "Insta Payment": "6371234567890123",
    "JCB": "3530111333300000",
    "Korean Local": "9123456789012345",
    "Laser": "6304123456789012",
    "Maestro": "501812345678",
    "Mastercard": "5555555555554444",
    "Solo": "6334123456789012",
    "Switch": "4903123456789012",
    "UnionPay": "6212345678901234",
    "Visa": "4111111111111111",
}


@pytest.fixture(scope="module")
def regex():
    return RegexDetector(PATTERNS)


@pytest.fixture(scope="module")
def ner():
    return PersonNameDetector("en_core_web_sm")


def test_twenty_independent_patterns(regex):
    assert len(regex.patterns) == 20
    assert set(regex.labels) == set(POSITIVES)


@pytest.mark.parametrize("label", POSITIVES)
def test_pattern_detects_synthetic_value(regex, label):
    assert regex.count(POSITIVES[label]).get(label, 0) >= 1


def test_text_without_digits_or_at_sign_has_no_regex_hits(regex):
    assert regex.count("The patient reported mild headaches and fatigue.") == {}


def test_email_regex_is_lowercase_as_published(regex):
    assert "Email" not in regex.count("JANE@EXAMPLE.COM")


def test_detection_count_is_per_match(regex):
    assert regex.count("a@example.com and b@example.com")["Email"] == 2


@pytest.mark.parametrize("name", ["us_ssn", "email"])
def test_speedups_do_not_change_matches(regex, name):
    pattern = next(p for p in regex.patterns if p.name == name)
    rng = random.Random(0)
    alphabet = "0123456789-@. ab\n"
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(40)))
        assert pattern.count(text) == sum(1 for _ in pattern.regex.finditer(text))


def test_ner_counts_person_names(ner):
    texts = ["Dr. John Smith met Mary Johnson in Boston.", "The weather is nice."]
    results = list(ner.iter_counts((t, i) for i, t in enumerate(texts)))
    assert results == [(0, 2), (1, 0)]


def test_long_text_is_split_and_aggregated(ner):
    sentence = "John Smith went home. "
    long_text = sentence * 200
    ner.max_chunk_chars = 500
    try:
        (_, n), = list(ner.iter_counts([(long_text, "ctx")]))
    finally:
        ner.max_chunk_chars = 1_000_000
    assert n > 100


def test_split_text_preserves_content():
    text = "word " * 1000
    pieces = split_text(text, 123)
    assert all(len(p) <= 123 for p in pieces)
    assert "".join(pieces) == text


def test_split_text_without_whitespace():
    pieces = split_text("x" * 250, 100)
    assert [len(p) for p in pieces] == [100, 100, 50]


@pytest.mark.parametrize(
    "record,expected",
    [
        ({"a": "hello"}, ("hello", dl.OK)),
        ({"a": None}, (None, dl.NULL)),
        ({}, (None, dl.NULL)),
        ({"a": ""}, (None, dl.EMPTY)),
        ({"a": "  \n"}, (None, dl.EMPTY)),
        ({"a": 123}, (None, dl.INVALID)),
    ],
)
def test_extract_text_statuses(record, expected):
    assert dl.extract_text(record, ("a",)) == expected


def test_extract_text_joins_columns_and_enters_nested_fields():
    record = {"q": "one", "conv": [{"value": "two"}, {"value": "three"}], "x": None}
    assert dl.extract_text(record, ("q", "conv.value", "x")) == ("one\ntwo\nthree", dl.OK)


def spec_for(path, *columns):
    return dl.DatasetSpec(source=str(path), text_columns=columns or ("text",))


def test_local_formats(tmp_path):
    rows = [{"text": "a"}, {"text": "b"}]
    (tmp_path / "array.json").write_text(json.dumps(rows))
    (tmp_path / "lines.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "table.csv").write_text("text\na\nb\n")
    (tmp_path / "plain.txt").write_text("a\nb\n")
    (tmp_path / "posts.xml").write_text('<posts><row text="a"/><row text="b"/></posts>')
    for name in ("array.json", "lines.jsonl", "table.csv", "plain.txt", "posts.xml"):
        assert [r["text"] for r in dl.iter_records(spec_for(tmp_path / name))] == ["a", "b"], name


def test_local_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"text": ["a", "b"], "other": [1, 2]}), tmp_path / "t.parquet")
    assert list(dl.iter_records(spec_for(tmp_path / "t.parquet"))) == [{"text": "a"}, {"text": "b"}]


def test_iter_texts_tallies_skipped_and_leaves_source_unchanged(tmp_path):
    path = tmp_path / "d.jsonl"
    content = "\n".join(json.dumps({"text": v}) for v in ["ok", None, "", 5]) + "\n"
    path.write_text(content)
    skipped = {"null": 0, "empty": 0, "invalid": 0}
    assert list(dl.iter_texts(spec_for(path), skipped)) == ["ok"]
    assert skipped == {"null": 1, "empty": 1, "invalid": 1}
    assert path.read_text() == content


def test_missing_column_raises(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"body": "x"}\n')
    with pytest.raises(ValueError, match="body"):
        list(dl.iter_texts(spec_for(path, "text"), {"null": 0, "empty": 0, "invalid": 0}))


def test_reservoir_sample_is_deterministic_and_sized():
    a, seen = dl.reservoir_sample(range(1000), 10, seed=7)
    b, _ = dl.reservoir_sample(range(1000), 10, seed=7)
    c, _ = dl.reservoir_sample(range(1000), 10, seed=8)
    assert a == b and a != c and len(a) == 10 and seen == 1000


def test_reservoir_sample_smaller_population():
    assert dl.reservoir_sample(range(3), 10, seed=1) == ([0, 1, 2], 3)


def test_metrics_size_rate_uses_bytes_not_instances():
    stats = PIIStatistics(["Email", "Phone"])
    stats.add({"Email": 1}, 100)
    stats.add({}, 300)
    summary = stats.summary()
    assert summary["pii_containing_instance_rate_percent"] == 50.0
    assert summary["pii_containing_data_size_rate_percent"] == 25.0


def test_metrics_type_distribution_and_co_occurrence():
    stats = PIIStatistics(["Email", "Phone"], [("Name", "Email"), ("Name", "Email", "Phone")])
    stats.add({"Name": 2, "Email": 1}, 10)
    stats.add({"Email": 1, "Name": 1, "Phone": 3}, 10)
    stats.add({"Phone": 1}, 10)

    distribution = stats.type_distribution()
    assert distribution["Name"]["total_detections"] == 3
    assert distribution["Name"]["pii_containing_instances"] == 2
    assert distribution["Phone"]["total_detections"] == 4

    co = stats.co_occurrence()
    assert co["instances_with_multiple_pii_types"] == 2
    assert co["key_combinations_at_least_these_types"] == {"Name + Email": 2, "Name + Email + Phone": 1}
    assert {r["types"] for r in co["exact_type_combinations"]} == {"Name + Email", "Name + Email + Phone"}


def test_metrics_empty_dataset_has_no_rate():
    assert PIIStatistics(["Email"]).summary()["pii_containing_instance_rate_percent"] is None


# --- PII value dump --------------------------------------------------------

from dataset_audit.pii_values import PIIValueCollector, normalize, read_pii_values  # noqa: E402


def test_regex_find_matches_count(regex):
    text = "a@example.com, b@example.com, call (555) 555-0100"
    found = regex.find(text)
    assert found["Email"] == ["a@example.com", "b@example.com"]
    assert {k: len(v) for k, v in found.items()} == regex.count(text)


@pytest.mark.parametrize("name", ["us_ssn", "email"])
def test_speedup_find_returns_same_strings(regex, name):
    pattern = next(p for p in regex.patterns if p.name == name)
    rng = random.Random(1)
    alphabet = "0123456789-@. ab\n"
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(40)))
        assert list(pattern.find(text)) == [m.group(0) for m in pattern.regex.finditer(text)]


def test_ner_entities_agree_with_counts(ner):
    texts = ["Dr. John Smith met Mary Johnson in Boston.", "The weather is nice."]
    entities = list(ner.iter_entities((t, i) for i, t in enumerate(texts)))
    counts = list(ner.iter_counts((t, i) for i, t in enumerate(texts)))
    assert [(c, len(e)) for c, e in entities] == counts
    assert "John Smith" in entities[0][1]


@pytest.mark.parametrize(
    "label, a, b",
    [
        ("Email", "Jane.Doe@Example.com ", "jane.doe@example.com"),
        ("Phone", "(555) 555-0100", "555.555.0100"),
        ("Phone", "+1 555 555 0100", "555-555-0100"),
        ("Name", "John  Smith", "john smith"),
        ("US SSN", "123-45-6789", "123 45 6789"),
    ],
)
def test_normalize_equates_formats(label, a, b):
    assert normalize(label, a) == normalize(label, b)


def test_collector_counts_and_writes(tmp_path):
    c = PIIValueCollector(max_instance_ids=2)
    c.add(0, {"Email": ["a@example.com", "a@example.com"], "Name": ["John Smith"]})
    c.add(5, {"Email": ["a@example.com"]})
    c.add(9, {"Email": ["a@example.com"], "Name": ["john  smith"]})
    path = tmp_path / "v.jsonl.gz"
    c.write(path)
    rows = {(r["type"], r["key"]): r for r in read_pii_values(path)}
    email = rows[("Email", "a@example.com")]
    assert (email["count"], email["n_instances"], email["instance_ids"]) == (4, 3, [0, 5])
    name = rows[("Name", "john smith")]
    assert name["surface_forms"] == {"John Smith": 1, "john  smith": 1}
    first = path.read_bytes()
    c.write(path)
    assert path.read_bytes() == first  # deterministic output


def test_indexed_texts_point_at_source_records(tmp_path):
    path = tmp_path / "d.jsonl"
    rows = [{"text": "a"}, {"text": ""}, {"text": None}, {"text": "b"}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    skipped = {"null": 0, "empty": 0, "invalid": 0}
    spec = dl.DatasetSpec(source=str(path), text_columns=("text",))
    assert list(dl.iter_indexed_texts(spec, skipped)) == [(0, "a"), (3, "b")]
