"""Rebuild the Enron train/test/validation split used by PrivateInvestigator (pii_leakage CustomEnron).

Reproduces src/pii_leakage/extern/enron/enron.py ("undefended" config) verbatim:
  * walk the CMU maildir, sort file paths, read with iso-8859-1
  * drop the header up to the "X-FileName:" line and the blank lines after it
  * shuffle with pii_leakage.utils.random.rnd_idx(N, seed=42)
  * train = [0, 0.45), test = [0.45, 0.55), validation = [0.55, 1.0)

Usage:
  python scripts/build_enron_pi_split.py --maildir datasets/email/enron_cmu/raw/maildir \
      --output-dir datasets/email/enron_cmu/pi_split
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

SPLITS = [("train", 0.0, 0.45), ("test", 0.45, 0.55), ("validation", 0.55, 1.0)]
SEED = 42


def travel_dirs(path):
    file_list = []
    if os.path.isdir(path):
        for name in os.listdir(path):
            file_list.extend(travel_dirs(os.path.join(path, name)))
    else:
        file_list.append(path)
    return file_list


def rnd_idx(N, seed=None):
    lst = list(range(250_000)) if N < 250_000 else list(range(N))
    if seed is not None:
        random.Random(seed).shuffle(lst)
    else:
        random.shuffle(lst)
    return [x for x in lst if x < N]


def load(maildir):
    data, paths = [], []
    index_l = 0
    for file_path in sorted(travel_dirs(str(maildir))):
        with open(file_path, "r", encoding="iso-8859-1") as f:
            try:
                lines = f.readlines()
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError :\t{file_path}")
                continue
        header_ends = False
        # Same loop as the original; index_l deliberately carries over for empty files.
        for index_l, line in enumerate(lines):
            if line[:11] == "X-FileName:":
                header_ends = True
                continue
            if header_ends:
                if line.strip() == "":
                    continue
                break
        data.append("".join(lines[index_l:]))
        paths.append(os.path.relpath(file_path, maildir))
    return data, paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maildir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    data, paths = load(args.maildir)
    order = rnd_idx(N=len(data), seed=SEED)
    data = [data[i] for i in order]
    paths = [paths[i] for i in order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"maildir": str(args.maildir), "n_files": len(data), "seed": SEED, "splits": {}}
    for name, start, end in SPLITS:
        s, e = int(len(data) * start), int(len(data) * end)
        out = args.output_dir / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for i in range(s, e):
                f.write(json.dumps({"id": i + 1, "path": paths[i], "text": data[i]}, ensure_ascii=False) + "\n")
        summary["splits"][name] = {
            "file": out.name, "start": s, "end": e, "num_rows": e - s,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        }
        print(f"{name}: {e - s:,} rows -> {out}")
    (args.output_dir / "split_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
