"""Stream records from a Hugging Face dataset or local files and extract their text."""
import csv
import glob
import hashlib
import json
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import ijson
import pyarrow.parquet as pq

LOCAL_SUFFIXES = {".json", ".jsonl", ".ndjson", ".parquet", ".csv", ".tsv", ".txt", ".xml"}
COLUMN_JOINER = "\n"

OK, NULL, EMPTY, INVALID = "ok", "null", "empty", "invalid"


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    text_columns: tuple
    split: str = "train"
    config_name: str | None = None
    revision: str | None = None
    streaming: bool = True


def resolve_local_files(source):
    """Return the data files for a local path, directory or glob; empty for a Hub ID."""
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in LOCAL_SUFFIXES)
    return sorted(Path(p) for p in glob.glob(source) if Path(p).is_file())


def is_local(source):
    return bool(resolve_local_files(source))


def _iter_json(path):
    with path.open("rb") as f:
        first = f.read(1)
        while first.isspace():
            first = f.read(1)
        f.seek(0)
        if first == b"[":
            yield from ijson.items(f, "item")
        else:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _iter_parquet(path, columns):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=columns):
        yield from batch.to_pylist()


def _iter_delimited(path, delimiter):
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter=delimiter)


def _iter_text(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield {"text": line.rstrip("\n")}


def _iter_xml(path):
    for _, elem in ET.iterparse(path):
        if elem.tag == "row":
            yield dict(elem.attrib)
        elem.clear()


def iter_local_file(path, columns):
    suffix = path.suffix.lower()
    if suffix in (".json", ".jsonl", ".ndjson"):
        return _iter_json(path)
    if suffix == ".parquet":
        return _iter_parquet(path, columns)
    if suffix in (".csv", ".tsv"):
        return _iter_delimited(path, "\t" if suffix == ".tsv" else ",")
    if suffix == ".txt":
        return _iter_text(path)
    if suffix == ".xml":
        return _iter_xml(path)
    raise ValueError(f"Unsupported local file type: {path}")


def _top_level_columns(text_columns):
    return list(dict.fromkeys(c.split(".")[0] for c in text_columns))


def iter_records(spec):
    """Yield raw records one at a time, leaving the source untouched."""
    columns = _top_level_columns(spec.text_columns)
    files = resolve_local_files(spec.source)
    if files:
        for path in files:
            yield from iter_local_file(path, columns)
        return

    from datasets import load_dataset

    dataset = load_dataset(
        spec.source, name=spec.config_name, split=spec.split, revision=spec.revision, streaming=spec.streaming
    )
    yield from dataset.select_columns(columns)


def _leaves(value, path):
    """Yield the values at a dotted path, mapping over lists."""
    if isinstance(value, list):
        for item in value:
            yield from _leaves(item, path)
    elif not path:
        yield value
    elif isinstance(value, dict):
        yield from _leaves(value.get(path[0]), path[1:])
    else:
        yield None


def extract_text(record, text_columns):
    """Return (text, status); text is None unless status is OK."""
    parts, saw_empty, saw_invalid = [], False, False
    for column in text_columns:
        for leaf in _leaves(record, column.split(".")):
            if isinstance(leaf, str):
                if leaf.strip():
                    parts.append(leaf)
                else:
                    saw_empty = True
            elif leaf is not None:
                saw_invalid = True
    if parts:
        return COLUMN_JOINER.join(parts), OK
    if saw_invalid:
        return None, INVALID
    return None, EMPTY if saw_empty else NULL


def check_columns(record, text_columns):
    missing = [c for c in _top_level_columns(text_columns) if c not in record]
    if missing:
        raise ValueError(f"Columns {missing} not found. Available columns: {sorted(record)}")


def iter_texts(spec, skipped):
    """Yield analyzable text, tallying unusable records in the `skipped` dict."""
    for _, text in iter_indexed_texts(spec, skipped):
        yield text


def iter_indexed_texts(spec, skipped):
    """Yield (record index, text) for analyzable records.

    The index is the 0-based position of the record in the source stream (files in
    sorted order), counting skipped records too, so it points back to the raw record.
    """
    for index, record in enumerate(iter_records(spec)):
        if index == 0:
            check_columns(record, spec.text_columns)
        text, status = extract_text(record, spec.text_columns)
        if status == OK:
            yield index, text
        else:
            skipped[status] += 1


def reservoir_sample(items, k, seed):
    """Return (uniform sample of k items, number of items seen) in one pass."""
    rng = random.Random(seed)
    sample = []
    seen = 0
    for item in items:
        seen += 1
        if len(sample) < k:
            sample.append(item)
        else:
            j = rng.randrange(seen)
            if j < k:
                sample[j] = item
    return sample, seen


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _find_manifest(path):
    for parent in path.parents:
        candidate = parent / "manifest.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
        if parent.name == "datasets":
            break
    return None


def describe_dataset(spec):
    """Return provenance information for the output report."""
    info = {
        "source": spec.source,
        "text_columns": list(spec.text_columns),
    }
    files = resolve_local_files(spec.source)
    if files:
        manifest = _find_manifest(files[0].resolve())
        info["type"] = "local"
        info["split"] = None
        info["files"] = [{"path": str(p), "sha256": _sha256(p), "size_bytes": p.stat().st_size} for p in files]
        if manifest:
            info["origin"] = {k: manifest[k] for k in ("repo", "revision", "url") if k in manifest}
        return info

    from huggingface_hub import HfApi

    info["type"] = "huggingface"
    info["config_name"] = spec.config_name
    info["split"] = spec.split
    info["streaming"] = spec.streaming
    info["revision_requested"] = spec.revision
    info["revision_resolved"] = HfApi().dataset_info(spec.source, revision=spec.revision).sha
    return info
