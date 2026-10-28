"""evaluation.tools.success_rate_calculator. CPU, no model.
Run from the MarkLLM root: python -m pytest test/test_success_rate_calculator.py"""
import os
import random
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.tools.success_rate_calculator import (DetectionResult, DetectionThreshold, DynamicThresholdSuccessRateCalculator,  # noqa: E402
                                                     FundamentalSuccessRateCalculator)

LABELS = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']


class _LoopReference:
    """DynamicThresholdSuccessRateCalculator as it was before the sorted / cumulative-count version and the split into
    DetectionThreshold + FundamentalSuccessRateCalculator: every candidate re-counts all scores (O(N^2))."""

    def __init__(self, labels, rule='best', target_fpr=None, reverse=False):
        self.labels, self.rule, self.target_fpr, self.reverse = labels, rule, target_fpr, reverse

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

    def _compute_metrics(self, inputs, threshold):
        if not self.reverse:
            TP = sum(1 for x in inputs if x.detect_result >= threshold and x.gold_label)
            FP = sum(1 for x in inputs if x.detect_result >= threshold and not x.gold_label)
            TN = sum(1 for x in inputs if x.detect_result < threshold and not x.gold_label)
            FN = sum(1 for x in inputs if x.detect_result < threshold and x.gold_label)
        else:
            TP = sum(1 for x in inputs if x.detect_result <= threshold and x.gold_label)
            FP = sum(1 for x in inputs if x.detect_result <= threshold and not x.gold_label)
            TN = sum(1 for x in inputs if x.detect_result > threshold and not x.gold_label)
            FN = sum(1 for x in inputs if x.detect_result > threshold and x.gold_label)
        return {
            'TPR': TP / (TP + FN) if TP + FN else 0, 'FPR': FP / (FP + TN) if FP + TN else 0,
            'TNR': TN / (TN + FP) if TN + FP else 0, 'FNR': FN / (FN + TP) if FN + TP else 0,
            'P': TP / (TP + FP) if TP + FP else 0, 'R': TP / (TP + FN) if TP + FN else 0,
            'F1': 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0, 'ACC': (TP + TN) / (len(inputs)) if inputs else 0,
        }

    def calculate(self, watermarked_result, non_watermarked_result):
        inputs = [DetectionResult(True, x) for x in watermarked_result] + [DetectionResult(False, x) for x in non_watermarked_result]
        metrics = self._compute_metrics(inputs, self._find_threshold(inputs))
        return {label: metrics[label] for label in self.labels if label in metrics}


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
    """DetectionThreshold.find_threshold gives the old threshold bit for bit, and DynamicThresholdSuccessRateCalculator the old
    rates (F1 now from FundamentalSuccessRateCalculator, 2PR/(P+R) instead of 2TP/(2TP+FP+FN): equal up to rounding)."""
    n = 0
    for w, nw, rule, fpr, reverse in _cases():
        kw = dict(rule=rule, target_fpr=fpr if rule == 'target_fpr' else None, reverse=reverse)
        old = _LoopReference(LABELS, **kw)
        inputs = [DetectionResult(True, x) for x in w] + [DetectionResult(False, x) for x in nw]
        threshold = DetectionThreshold(**kw).find_threshold(w, nw)
        assert threshold == old._find_threshold(inputs), (rule, fpr, reverse)
        new_rates, old_rates = DynamicThresholdSuccessRateCalculator(LABELS, **kw).calculate(w, nw), old.calculate(w, nw)
        assert new_rates.keys() == old_rates.keys()
        for k in LABELS:
            assert new_rates[k] == pytest.approx(old_rates[k], rel=1e-12, abs=0) if k == 'F1' else new_rates[k] == old_rates[k], k
        n += 1
    assert n == 300


def test_dynamic_is_threshold_then_fundamental():
    """DynamicThresholdSuccessRateCalculator = DetectionThreshold.find_threshold + decide + FundamentalSuccessRateCalculator."""
    w, nw = [3.0, 2.0, 2.5, 0.5], [2.0, 1.0, 0.0, -1.0, 1.5]
    th = DetectionThreshold(rule='target_fpr', target_fpr=0.2)
    t = th.find_threshold(w, nw)
    assert t == 1.75   # midpoint of 1.5 and 2.0: at or above it one of five non-watermarked scores (2.0), FPR 0.2
    assert th.decide(nw, t) == [True, False, False, False, False]
    assert (FundamentalSuccessRateCalculator(LABELS).calculate(th.decide(w, t), th.decide(nw, t))
            == DynamicThresholdSuccessRateCalculator(LABELS, rule='target_fpr', target_fpr=0.2).calculate(w, nw))


def test_dynamic_threshold_is_fast():
    """44,000 scores (4,000 watermarked, 40,000 not) in about a second; the loop took about 4 minutes."""
    rng = random.Random(1)
    w, nw = [rng.gauss(3, 1) for _ in range(4000)], [rng.gauss(0, 1) for _ in range(40000)]
    t0 = time.time()
    for rule in ('best', 'target_fpr'):
        DynamicThresholdSuccessRateCalculator(rule=rule, target_fpr=0.01).calculate(w, nw)
    assert time.time() - t0 < 10
