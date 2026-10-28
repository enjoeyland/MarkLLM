"""evaluation.tools.success_rate_calculator. CPU, no model.
Run from the MarkLLM root: python -m pytest test/test_success_rate_calculator.py"""
import os
import random
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.tools.success_rate_calculator import DetectionResult, DynamicThresholdSuccessRateCalculator  # noqa: E402

LABELS = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']


class _LoopReference(DynamicThresholdSuccessRateCalculator):
    """The threshold search as it was before the sorted / cumulative-count version: every candidate re-counts all scores (O(N^2))."""

    def _find_best_threshold(self, inputs):
        best_threshold, best_metrics = 0, None
        for i in range(len(inputs) - 1):
            threshold = (inputs[i].detect_result + inputs[i + 1].detect_result) / 2
            metrics = self._compute_metrics(inputs, threshold)
            if best_metrics is None or metrics['F1'] > best_metrics['F1']:
                best_threshold, best_metrics = threshold, metrics
        return best_threshold

    def _find_threshold_by_fpr(self, inputs):
        threshold = 0
        for i in range(len(inputs) - 1):
            threshold = (inputs[i].detect_result + inputs[i + 1].detect_result) / 2
            if self._compute_metrics(inputs, threshold)['FPR'] <= self.target_fpr:
                break
        return threshold

    def _find_threshold(self, inputs):
        inputs = sorted(inputs, key=lambda x: x.detect_result, reverse=self.reverse)
        return self._find_best_threshold(inputs) if self.rule == 'best' else self._find_threshold_by_fpr(inputs)


def _scores(rng, n, mean, ties):
    if ties:   # z-scores on a grid, as for green-token counts: many equal values
        return [float(rng.randint(0, 12) + mean) / 2 for _ in range(n)]
    return [rng.gauss(mean, 1.0) for _ in range(n)]


def _cases():
    rng = random.Random(0)
    for trial in range(300):
        ties = trial % 2 == 0
        n_w, n_n = rng.randint(0, 40), rng.randint(0, 120)
        yield (_scores(rng, n_w, 2.0, ties), _scores(rng, n_n, 0.0, ties), rng.choice(['best', 'target_fpr']),
               rng.choice([0.0, 0.01, 0.05, 0.3, 1.0]), rng.random() < 0.3)


def test_dynamic_threshold_same_as_loop():
    """The same threshold and the same rates as the per-candidate loop, bit for bit (ties, reverse, empty classes included)."""
    n = 0
    for w, nw, rule, fpr, reverse in _cases():
        kw = dict(labels=LABELS, rule=rule, target_fpr=fpr if rule == 'target_fpr' else None, reverse=reverse)
        new, old = DynamicThresholdSuccessRateCalculator(**kw), _LoopReference(**kw)
        inputs = [DetectionResult(True, x) for x in w] + [DetectionResult(False, x) for x in nw]
        assert new._find_threshold(inputs) == old._find_threshold(inputs), (rule, fpr, reverse)
        assert new.calculate(w, nw) == old.calculate(w, nw)
        n += 1
    assert n == 300


def test_dynamic_threshold_is_fast():
    """44,000 scores (4,000 watermarked, 40,000 not) in about a second; the loop took about 4 minutes."""
    rng = random.Random(1)
    w, nw = [rng.gauss(3, 1) for _ in range(4000)], [rng.gauss(0, 1) for _ in range(40000)]
    t0 = time.time()
    for rule in ('best', 'target_fpr'):
        DynamicThresholdSuccessRateCalculator(rule=rule, target_fpr=0.01).calculate(w, nw)
    assert time.time() - t0 < 10
