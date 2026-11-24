"""Hugging Face chat LMs. Models load lazily and only one is kept on the GPU at a time."""
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_active = []  # the ChatLM whose weights are loaded


class ChatLM:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def _tok(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self.tokenizer

    def _load(self):
        if self.model is not None:
            return
        for other in list(_active):
            other.unload()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="auto"
        ).eval()
        _active.append(self)

    def unload(self):
        self.model = None
        if self in _active:
            _active.remove(self)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def truncate(self, text, max_tokens):
        """Return (text cut to max_tokens, whether it was cut)."""
        ids = self._tok()(text, add_special_tokens=False)["input_ids"]
        return (self._tok().decode(ids[:max_tokens]), True) if len(ids) > max_tokens else (text, False)

    @torch.inference_mode()
    def chat(self, messages, n=1, max_new_tokens=1024, **sampling):
        """Greedy if no sampling kwargs (temperature, top_p) are given; otherwise n sampled replies."""
        self._load()
        tok = self._tok()
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # The chat template already contains BOS where the model needs one.
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        decoding = {"do_sample": True, "num_return_sequences": n, **sampling} if sampling else {"do_sample": False}
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id, **decoding,
        )
        start = inputs["input_ids"].shape[1]
        return [tok.decode(o[start:], skip_special_tokens=True) for o in out]
