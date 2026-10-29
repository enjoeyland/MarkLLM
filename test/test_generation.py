"""generation.WatermarkGenerator: batched generation for any watermark, extended with hooks. CPU, tiny random GPT2.
Run from the MarkLLM root: python -m pytest test/test_generation.py"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
transformers = pytest.importorskip("transformers")
from generation import GenerationHook, WatermarkGenerator  # noqa: E402


@pytest.fixture(scope="module")
def model_tok():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("gpt2 tokenizer not available")
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(n_layer=1, n_embd=32, n_head=2)).eval(), tok


class Bias:
    """A stand-in watermark: +5 on token 11 ('.' in GPT2)."""

    def __init__(self):
        self.logits_processor = self

    def __call__(self, input_ids, scores):
        scores[:, 11] += 5.0
        return scores


class Record(GenerationHook):
    def __init__(self):
        self.batches, self.wrapped = 0, 0

    def hf_kwargs(self):
        return dict(output_scores=True)

    def wrap_processor(self, processor):
        self.wrapped += 1
        return processor

    def before_batch(self, watermark, prompts):
        self.batches += 1

    def collect(self, out, row, new_tokens, ntok):
        return dict(n_scores=len(out.scores))


def test_one_result_per_prompt_one_batch_per_call(model_tok):
    model, tok = model_tok
    hook = Record()
    gen = WatermarkGenerator(model, tok, hooks=[hook], max_new_tokens=5, do_sample=True)
    prompts = ["The dog", "A cat sat on the", "Hi"]
    outs = gen.generate(prompts, watermark=Bias())
    assert [o.prompt for o in outs] == prompts
    assert all(o.text.startswith(p) for o, p in zip(outs, prompts))
    assert all(1 <= o.ntok <= 5 and len(o.tokens) == o.ntok for o in outs)
    assert hook.batches == 1 and hook.wrapped == 1 and all(o.extras["n_scores"] >= 1 for o in outs)
    gen.generate(prompts[:1], watermark=Bias())   # the caller batches: each call is one batch
    assert hook.batches == 2


def test_no_watermark(model_tok):
    model, tok = model_tok
    outs = WatermarkGenerator(model, tok, max_new_tokens=3, do_sample=True).generate(["Hello"], watermark=None)
    assert len(outs) == 1 and outs[0].ntok >= 1
