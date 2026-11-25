"""Extract domain, task, fine-tuning method, training data and I/O format from Hugging Face model cards.

Usage:
  python scripts/extract_model_info.py --model-id <hub id> [<hub id> ...] [--output-dir outputs/model_info_v2]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_profile.extractor import ModelInfoExtractor  # noqa: E402
from model_profile.model_card import fetch_model_card  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", nargs="+", required=True, help="Hugging Face model IDs")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "model_info_v2")
    p.add_argument("--llm", default="Qwen/Qwen3-8B")
    args = p.parse_args(argv)

    extractor = ModelInfoExtractor(args.llm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_id in args.model_id:
        info = extractor.extract(fetch_model_card(model_id))
        path = args.output_dir / f"{model_id.replace('/', '__')}.json"
        path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{model_id} -> {path}")


if __name__ == "__main__":
    main()
