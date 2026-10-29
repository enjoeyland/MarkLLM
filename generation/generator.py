"""WatermarkGenerator: the batched counterpart of `generate_watermarked_text`, for any watermark that has a `logits_processor`.

Each algorithm's `generate_watermarked_text` takes one prompt and returns one text. This generator takes a list of prompts,
generates them as ONE batch (left padding), and returns one `GeneratedText` per prompt. Splitting the data into batches is
the caller's job (e.g. a torch DataLoader), so there is one place that decides the batch size. What else to do during generation is
added with hooks (`GenerationHook`), so one generator class serves every use:

    hf_kwargs()                        extra arguments for `model.generate` (e.g. output_scores=True)
    wrap_processor(processor)          wrap the watermark's logits processor (called with None when there is no watermark)
    before_batch(watermark, prompts)   called before each generate call (e.g. reset a stateful processor)
    collect(out, row, new_tokens, ntok) -> dict   per-sentence values stored in GeneratedText.extras

The generation itself is the same as `generate_watermarked_text`: `model.generate(**encoded_prompts,
logits_processor=[watermark.logits_processor], **gen_kwargs)`, with `return_dict_in_generate=True` so hooks can read the output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import LogitsProcessorList


class GenerationHook:
    """Base class of hooks: every method does nothing by default; override only what is needed."""

    def hf_kwargs(self) -> Dict[str, Any]:
        return {}

    def wrap_processor(self, processor):
        return processor

    def before_batch(self, watermark, prompts: List[str]) -> None:
        pass

    def collect(self, out, row: int, new_tokens: torch.Tensor, ntok: int) -> Dict[str, Any]:
        return {}


@dataclass
class GeneratedText:
    prompt: str
    text: str            # prompt + generation, as generate_watermarked_text returns it
    completion: str      # the generation only
    tokens: List[int]    # generated token ids up to and including the first EOS
    ntok: int
    extras: Dict[str, Any] = field(default_factory=dict)


class WatermarkGenerator:
    def __init__(self, model, tokenizer, hooks: Sequence[GenerationHook] = (), device: Optional[str] = None, **gen_kwargs):
        self.model, self.tokenizer = model, tokenizer
        self.hooks = list(hooks)
        self.device = device or str(next(model.parameters()).device)
        self.gen_kwargs = gen_kwargs

    def _encode(self, prompts: List[str]):
        tok = self.tokenizer
        side, pad = tok.padding_side, tok.pad_token
        tok.padding_side = "left"  # decoder-only models generate after the last prompt token
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        try:
            return tok(prompts, return_tensors="pt", add_special_tokens=True, padding=True).to(self.device)
        finally:
            tok.padding_side, tok.pad_token = side, pad

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], watermark=None) -> List[GeneratedText]:
        """The prompts are generated together as one batch."""
        prompts = list(prompts)
        tok = self.tokenizer
        processor = getattr(watermark, "logits_processor", None) if watermark is not None else None
        if watermark is not None and processor is None:
            raise NotImplementedError("%s has no logits_processor" % type(watermark).__name__)
        for h in self.hooks:
            processor = h.wrap_processor(processor)
            h.before_batch(watermark, prompts)
        kwargs = dict(self.gen_kwargs)
        for h in self.hooks:
            kwargs.update(h.hf_kwargs())
        enc = self._encode(prompts)
        out = self.model.generate(**enc, logits_processor=LogitsProcessorList([processor]) if processor is not None else None,
                                  pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                                  return_dict_in_generate=True, **kwargs)
        P = enc["input_ids"].shape[1]
        new = out.sequences[:, P:]
        texts = tok.batch_decode(out.sequences, skip_special_tokens=True)
        res = []
        for i, prompt in enumerate(prompts):
            is_eos = (new[i] == tok.eos_token_id).nonzero()
            n = int(is_eos[0, 0]) + 1 if len(is_eos) else new.shape[1]
            extras = {}
            for h in self.hooks:
                extras.update(h.collect(out, i, new[i, :n], n))
            res.append(GeneratedText(prompt=prompt, text=texts[i], completion=tok.decode(new[i, :n], skip_special_tokens=True),
                                     tokens=new[i, :n].tolist(), ntok=n, extras=extras))
        return res
