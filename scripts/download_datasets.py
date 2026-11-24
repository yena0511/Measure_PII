"""Download configured datasets into datasets/<domain>/<name>/raw/ and write a manifest.

Usage: python scripts/download_datasets.py [--domain healthcare]
"""
import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import py7zr
import pyarrow.parquet as pq
import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent


def load_records(path):
    """Read a JSON array or JSON Lines file."""
    with path.open(encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def table_info(paths):
    """Return (sorted column names, total row count) for parquet, CSV or JSON files."""
    if paths[0].suffix == ".parquet":
        columns = sorted(pq.read_schema(paths[0]).names)
        return columns, sum(pq.read_metadata(p).num_rows for p in paths)
    if paths[0].suffix == ".csv":
        columns, num_rows = None, 0
        for p in paths:
            with p.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                num_rows += sum(1 for _ in reader)
                columns = columns or sorted(reader.fieldnames)
        return columns, num_rows
    records = [record for p in paths for record in load_records(p)]
    return sorted(records[0]), len(records)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(dataset_dir, manifest):
    manifest["rows_match_expected"] = manifest["num_rows"] == manifest["expected_rows"]
    manifest["downloaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def count_posts(path, created_before):
    """Count questions, and answers to those questions, created before a date."""
    questions = set()
    answers = []
    for _, elem in ET.iterparse(path):
        if elem.tag == "row" and elem.attrib["CreationDate"] < created_before:
            if elem.attrib["PostTypeId"] == "1":
                questions.add(elem.attrib["Id"])
            elif elem.attrib["PostTypeId"] == "2":
                answers.append((elem.attrib["ParentId"], int(elem.attrib["Score"])))
        elem.clear()
    answers = [score for parent, score in answers if parent in questions]
    return len(questions), len(answers), sum(score >= 5 for score in answers)


def fetch_stackexchange(domain, name, spec):
    dataset_dir = ROOT / "datasets" / domain / name
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "dump.7z"
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "dataset-downloader"})
    with urllib.request.urlopen(request) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)
    archive_sha256 = sha256(archive)
    with py7zr.SevenZipFile(archive) as z:
        z.extract(path=raw_dir, targets=["Posts.xml"])
    archive.unlink()

    path = raw_dir / "Posts.xml"
    n_questions, n_answers, n_answers_ge5 = count_posts(path, spec["created_before"])
    return write_manifest(dataset_dir, {
        "domain": domain,
        "name": name,
        "url": spec["url"],
        "archive_sha256": archive_sha256,
        "file": "raw/Posts.xml",
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "created_before": spec["created_before"],
        "num_rows": n_questions,
        "num_rows_definition": "questions created before created_before",
        "num_answers": n_answers,
        "num_answers_score_ge5": n_answers_ge5,
        "expected_rows": spec["expected_rows"],
    })


def fetch_hub(domain, name, spec):
    dataset_dir = ROOT / "datasets" / domain / name
    raw_dir = dataset_dir / "raw"
    snapshot_download(
        repo_id=spec["repo"],
        repo_type="dataset",
        revision=spec["revision"],
        allow_patterns=[spec["file"]],
        local_dir=raw_dir,
    )
    shutil.rmtree(raw_dir / ".cache", ignore_errors=True)

    paths = sorted(raw_dir.glob(spec["file"]))
    columns, num_rows = table_info(paths)
    return write_manifest(dataset_dir, {
        "domain": domain,
        "name": name,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "file": f"raw/{spec['file']}",
        "sha256": sha256(paths[0]) if len(paths) == 1 else {p.name: sha256(p) for p in paths},
        "size_bytes": sum(p.stat().st_size for p in paths),
        "columns": columns,
        "num_rows": num_rows,
        "expected_rows": spec["expected_rows"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="Only download this domain")
    parser.add_argument("--name", help="Only download this dataset")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text(encoding="utf-8"))
    for domain, datasets in config.items():
        if args.domain and domain != args.domain:
            continue
        for name, spec in datasets.items():
            if args.name and name != args.name:
                continue
            fetch =fetch_stackexchange if "url" in spec else fetch_hub
            m = fetch(domain, name, spec)
            status = "OK" if m["rows_match_expected"] else "MISMATCH"
            print(f"{status} {domain}/{name}: rows={m['num_rows']} expected={m['expected_rows']}")


if __name__ == "__main__":
    main()
