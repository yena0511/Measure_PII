import re

from huggingface_hub import DatasetCard, HfApi, ModelCard, hf_hub_download

_DATASET_LINK = re.compile(r"huggingface\.co/datasets/([\w.\-]+/[\w.\-]+)")


def fetch_model_card(model_id):
    """Return the model card (YAML metadata + README body) pinned to the current commit."""
    revision = HfApi().model_info(model_id).sha
    card = ModelCard.load(hf_hub_download(model_id, "README.md", revision=revision))
    return {"model_id": model_id, "revision": revision, "text": card.content}


def linked_dataset_ids(card_text):
    """Datasets named in the card's YAML `datasets:` field or linked as huggingface.co/datasets/<id>, in order."""
    meta = ModelCard(card_text).data.to_dict().get("datasets") or []
    ids = [meta] if isinstance(meta, str) else list(meta)
    ids += [m.rstrip(".") for m in _DATASET_LINK.findall(card_text)]
    return list(dict.fromkeys(ids))


def fetch_dataset_card(dataset_id):
    """Return the dataset card (README only; no data rows) pinned to the current commit."""
    revision = HfApi().dataset_info(dataset_id).sha
    card = DatasetCard.load(hf_hub_download(dataset_id, "README.md", repo_type="dataset", revision=revision))
    return {"dataset_id": dataset_id, "revision": revision, "text": card.content}
