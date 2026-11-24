import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

UNKNOWN = "unknown"

# Field -> what to extract. Examples are separated by "/" so the LLM does not copy them as one value.
FIELDS = {
    "domain": "the application domain of the model or its training data (Medical / Legal / News / Finance / Code / General)",
    "task": "what the model is trained or meant to do, from its description or usage examples (QA / Summarization / Chat / Classification / Translation)",
    "fine_tuning_method": "how it was trained from its base model: SFT (supervised or instruction fine-tuning) / CPT (continued or domain-adaptive pre-training) / LoRA / QLoRA / Full fine-tuning / DPO / RLHF",
    "training_data": '"name" = training dataset name(s); "description" = what the data is (document types, sources, size)',
    "input_format": "how inputs are given to the model (Chat Template / Instruction / Completion)",
    "output_format": "what the model outputs (Summary / Answer / Free text)",
}

EXTRACT_PROMPT = """You extract facts about a fine-tuned language model from its Hugging Face model card (YAML metadata followed by the README).

Rules:
- Use only what the card states. Do not guess from the model name, base model, or general knowledge.
- For each field, FIRST copy the sentence(s) that state it from the card VERBATIM into "evidence", THEN write the value based only on that evidence.
- If the card does not state a field, use an empty evidence list and the value "unknown".
- Values are short labels; join several with ", ". Never add a label the evidence does not mention.
- Hub tags such as text-generation-inference, transformers, unsloth or trl name software libraries, not the task or method.

Fields (examples are separated by "/"):
""" + "\n".join(f"- {f}: {d}" for f, d in FIELDS.items()) + """

Answer with one JSON object only:
{"domain": {"evidence": [], "value": ""},
 "task": {"evidence": [], "value": ""},
 "fine_tuning_method": {"evidence": [], "value": ""},
 "training_data": {"evidence": [], "name": "", "description": ""},
 "input_format": {"evidence": [], "value": ""},
 "output_format": {"evidence": [], "value": ""}}"""

SUPPORT_PROMPT = """You verify information extracted from a Hugging Face model card.
Answer "yes" only if the card explicitly states the claim, possibly using a synonym or an abbreviation (e.g. "continued pre-training" = CPT, "biomedicine" = Medical, a "legal" tag = Legal, "trained using the full parameters" = Full fine-tuning).
Answer "no" if the card does not state it, or the claim is only implied or guessed. Answer with "yes" or "no" only."""


def _tokens(text):
    return re.findall(r"[a-z0-9]+", re.sub(r"\\[nt]", " ", text.lower()))


def _in_card(quote, card_text):
    """True if the quote's words appear in order in the card, allowing a few skipped words (markdown, YAML lists)."""
    words = _tokens(quote)
    pattern = r"(?:\S+ ){0,10}".join(re.escape(w) + " " for w in words)
    return bool(words) and re.search(r"(?:^| )" + pattern, " ".join(_tokens(card_text)) + " ") is not None


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if value is None or " ".join(_tokens(str(value))) in {"", "unknown", "n a", "none", "not specified"}:
        return UNKNOWN
    return str(value).strip()


class ModelInfoExtractor:
    def __init__(self, llm="Qwen/Qwen3-8B", max_card_tokens=24000, max_new_tokens=1024):
        self.llm, self.max_card_tokens, self.max_new_tokens = llm, max_card_tokens, max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(llm)
        self.model = AutoModelForCausalLM.from_pretrained(llm, dtype=torch.bfloat16, device_map="auto").eval()

    def extract(self, card):
        card_ids = self.tokenizer(card["text"])["input_ids"]
        truncated = len(card_ids) > self.max_card_tokens
        card_text = self.tokenizer.decode(card_ids[: self.max_card_tokens]) if truncated else card["text"]
        user_prompt = f"Model ID: {card['model_id']}\n\n<model_card>\n{card_text}\n</model_card>"

        parsed = self._generate_json(EXTRACT_PROMPT, user_prompt)
        info = {"model_id": card["model_id"]}
        record = {"evidence": {}, "unverified_evidence": [], "rejected_values": {}}
        for field in FIELDS:
            out = parsed.get(field) or {}
            is_data = field == "training_data"
            value = _clean({"name": out.get("name"), "description": out.get("description")} if is_data else out.get("value"))
            quotes = [q for q in out.get("evidence") or [] if isinstance(q, str) and q.strip()]
            record["evidence"][field] = quotes
            record["unverified_evidence"] += [{"field": field, "quote": q} for q in quotes if not _in_card(q, card_text)]
            # A second LLM call checks the value against the whole card; unsupported values become unknown.
            empty = _clean({"name": None, "description": None} if is_data else None)
            if value != empty and not self._supported(field, value, card_text):
                record["rejected_values"][field], value = value, empty
            info[field] = value

        info.update(record)
        info["extraction"] = {
            "card_revision": card["revision"],
            "card_truncated": truncated,
            "llm": self.llm,
            "decoding": "greedy, non-thinking",
        }
        return info

    def _generate_json(self, system_prompt, user_prompt):
        raw = self._generate(system_prompt, user_prompt)
        return json.loads(raw[raw.index("{") : raw.rindex("}") + 1])

    def _supported(self, field, value, card_text):
        claim = f"Claim: the model's {field.replace('_', ' ')} is {json.dumps(value, ensure_ascii=False)}."
        answer = self._generate(SUPPORT_PROMPT, f"<model_card>\n{card_text}\n</model_card>\n\n{claim}")
        return answer.strip().lower().startswith("yes")

    @torch.inference_mode()
    def _generate(self, system_prompt, user_prompt):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
