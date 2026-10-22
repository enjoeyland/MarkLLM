"""Batched generation for any MarkLLM watermark (HF transformers), extensible with hooks.

    from generation import WatermarkGenerator, GenerationHook
    gen = WatermarkGenerator(model, tokenizer, batch_size=64, max_new_tokens=200, do_sample=True)
    outs = gen.generate(prompts, watermark=AutoWatermark.load("KGW", ...))   # watermark=None: no watermark
    outs[i].text, outs[i].completion, outs[i].tokens, outs[i].extras
"""
from .generator import GeneratedText, GenerationHook, WatermarkGenerator

__all__ = ["WatermarkGenerator", "GenerationHook", "GeneratedText"]
