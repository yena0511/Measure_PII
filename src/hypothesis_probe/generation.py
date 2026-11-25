"""Target-model decoding settings (V3): taken from the model card's own generation code when it states them.

Precedence per setting: command line > model card > DEFAULTS. Sampling is always on (n samples per probe), so the
card's do_sample is not used, and a card temperature of 0 (greedy) is ignored.
"""
import re

DEFAULTS = {"max_new_tokens": 256, "temperature": 0.8, "top_p": 0.95}
CARD_KEYS = {"max_new_tokens": int, "temperature": float, "top_p": float, "top_k": int, "repetition_penalty": float}
CALL = re.compile(r"(?:\.generate|GenerationConfig|\bpipeline|\bpipe|\bgenerator)\s*\(")
NUMBER = r"([0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?)"


def _call_args(text, start):
    """Text inside the parentheses opened just before `start` (balanced)."""
    depth = 1
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i]
    return text[start:]


def card_generation_config(card_text):
    """{setting: value} from the first generate(...) / GenerationConfig(...) / pipeline(...) call in the card's
    code blocks (else anywhere in the card) that sets any of CARD_KEYS as a number."""
    blocks = re.findall(r"```[^\n]*\n(.*?)```", card_text, flags=re.S)
    for text in [*blocks, card_text]:
        for call in CALL.finditer(text):
            args = _call_args(text, call.end())
            found = {k: cast(m.group(1)) for k, cast in CARD_KEYS.items()
                     if (m := re.search(rf"\b{k}\s*=\s*{NUMBER}", args))}
            if found.get("temperature") == 0:
                del found["temperature"]
            if found:
                return found
    return {}


def resolve(cli, card):
    """{setting: {"value", "source"}} for DEFAULTS plus any extra card setting (top_k, repetition_penalty).
    `cli`: {setting: value or None}."""
    out = {}
    for k in [*DEFAULTS, *(k for k in card if k not in DEFAULTS)]:
        if cli.get(k) is not None:
            out[k] = {"value": cli[k], "source": "command line"}
        elif k in card:
            out[k] = {"value": card[k], "source": "model card"}
        else:
            out[k] = {"value": DEFAULTS[k], "source": "default"}
    return out
