# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==========================================================
# success_rate_calculator.py
# Description: Calculate success rate of watermark detection
# ==========================================================

import numpy as np
from typing import List, Dict, Union
from exceptions.exceptions import TypeMismatchException, ConfigurationError


class DetectionResult:
    """Detection result."""

    def __init__(self, gold_label: bool, detect_result: Union[bool, float]) -> None:
        """
            Initialize the detection result.

            Parameters:
                gold_label (bool): The expected watermark presence.
                detect_result (Union[bool, float]): The detection result.
        """
        self.gold_label = gold_label
        self.detect_result = detect_result


def _roc_counts(watermarked: np.ndarray, non_watermarked: np.ndarray, thresholds: np.ndarray, strict: bool = False):
    """Number of watermarked (TP) and non-watermarked (FP) scores at or above (strict: above) each threshold.
    One sort per class and a binary search per threshold: O((N + T) log N) for N scores and T thresholds."""
    side = 'right' if strict else 'left'
    tp = len(watermarked) - np.searchsorted(np.sort(watermarked), thresholds, side=side)
    fp = len(non_watermarked) - np.searchsorted(np.sort(non_watermarked), thresholds, side=side)
    return tp, fp


class BaseSuccessRateCalculator:
    """Base class for success rate calculator."""
    def __init__(self, labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']) -> None:
        """
            Initialize the success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
        """
        self.labels = labels
    
    def _check_instance(self, data: List[Union[bool, float]], expected_type: type):
        """Check if the data is an instance of the expected type."""
        for d in data:
            if not isinstance(d, expected_type):
                raise TypeMismatchException(expected_type, type(d))
    
    def _filter_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Filter metrics based on the provided labels."""
        return {label: metrics[label] for label in self.labels if label in metrics}
    
    def calculate(self, watermarked_result: List[Union[bool, float]], non_watermarked_result: List[Union[bool, float]]) -> Dict[str, float]:
        """Calculate success rates based on provided results."""
        pass


class FundamentalSuccessRateCalculator(BaseSuccessRateCalculator):
    """
        Calculator for fundamental success rates of watermark detection.

        This class specifically handles the calculation of success rates for scenarios involving
        watermark detection after fixed thresholding. It provides metrics based on comparisons
        between expected watermarked results and actual detection outputs.

        Use this class when you need to evaluate the effectiveness of watermark detection algorithms
        under fixed thresholding conditions.
    """

    def __init__(self, labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']) -> None:
        """
            Initialize the fundamental success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
        """
        super().__init__(labels)
    
    def _compute_metrics(self, inputs: List[DetectionResult]) -> Dict[str, float]:
        """Compute metrics based on the provided inputs."""
        TP = sum(1 for d in inputs if d.detect_result and d.gold_label)
        TN = sum(1 for d in inputs if not d.detect_result and not d.gold_label)
        FP = sum(1 for d in inputs if d.detect_result and not d.gold_label)
        FN = sum(1 for d in inputs if not d.detect_result and d.gold_label)

        TPR = TP / (TP + FN) if TP + FN else 0.0
        FPR = FP / (FP + TN) if FP + TN else 0.0
        TNR = TN / (TN + FP) if TN + FP else 0.0
        FNR = FN / (FN + TP) if FN + TP else 0.0
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * (P * R) / (P + R) if P + R else 0.0
        ACC = (TP + TN) / (len(inputs)) if inputs else 0.0

        return {
            'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
            'P': P, 'R': R, 'F1': F1, 'ACC': ACC
        }

    def calculate(self, watermarked_result: List[bool], non_watermarked_result: List[bool]) -> Dict[str, float]:
        """calculate success rates of watermark detection based on provided results."""
        self._check_instance(watermarked_result, bool)
        self._check_instance(non_watermarked_result, bool)

        inputs = [DetectionResult(True, x) for x in watermarked_result] + [DetectionResult(False, x) for x in non_watermarked_result]
        metrics = self._compute_metrics(inputs)
        return self._filter_metrics(metrics)


class DetectionThreshold:
    """
        A detection threshold chosen from scores whose labels are known.

        find_threshold(watermarked, non_watermarked) gives the threshold; decide(scores, threshold) gives the decisions
        (True: watermark detected), which FundamentalSuccessRateCalculator turns into rates. Rules:
            'best'        the candidate that maximizes F1
            'target_fpr'  the first candidate whose FPR is at most target_fpr
        The candidates are midpoints of neighbouring sorted scores (both classes), and a score at or above the threshold
        is detected (reverse: at or below).
    """

    RULES = ['best', 'target_fpr']

    def __init__(self, rule='best', target_fpr=None, reverse=False) -> None:
        """
            Parameters:
                rule (str): The rule for determining the threshold. Choose from 'best' or 'target_fpr'.
                target_fpr (float): The target false positive rate to achieve.
                reverse (bool): Whether to reverse the sorting order of the detection results.
                                False: higher values are considered positive.
                                True: lower values are considered positive.
        """
        self.rule = rule
        self.target_fpr = target_fpr
        self.reverse = reverse

        # Validate rule configuration
        if self.rule not in self.RULES:
            raise ConfigurationError(f"Invalid rule specified: {self.rule}. Choose from {', '.join(repr(r) for r in self.RULES)}.")

        # Validate target_fpr configuration based on the rule
        if self.rule != 'best':
            if self.target_fpr is None:
                raise ConfigurationError(f"target_fpr must be set when rule is '{self.rule}'.")
            if not isinstance(self.target_fpr, (float, int)) or not (0 <= self.target_fpr <= 1):
                raise ConfigurationError("target_fpr must be a float or int within the range [0, 1].")

    def find_threshold(self, watermarked_result: List[float], non_watermarked_result: List[float]) -> float:
        """The threshold for these scores under the rule."""
        inputs = [DetectionResult(True, x) for x in watermarked_result] + [DetectionResult(False, x) for x in non_watermarked_result]
        return self._find_threshold(inputs)

    def decide(self, scores: List[float], threshold: float) -> List[bool]:
        """Detection decisions at the threshold: score >= threshold (reverse: score <= threshold)."""
        if self.reverse:
            return [bool(x <= threshold) for x in scores]
        return [bool(x >= threshold) for x in scores]

    def _candidates(self, inputs: List[DetectionResult]):
        """Candidate thresholds (midpoints of neighbouring scores, in the order of the sorted scores) and, at each, the counts
        TP, FP, FN, TN. With reverse, lower scores are positive: the scores are negated, so 'x <= t' becomes '-x >= -t'."""
        sign = -1.0 if self.reverse else 1.0
        scores = np.sort(np.array([sign * x.detect_result for x in inputs], dtype=float))
        thresholds = (scores[:-1] + scores[1:]) / 2
        watermarked = np.array([sign * x.detect_result for x in inputs if x.gold_label], dtype=float)
        non_watermarked = np.array([sign * x.detect_result for x in inputs if not x.gold_label], dtype=float)
        tp, fp = _roc_counts(watermarked, non_watermarked, thresholds)
        return sign * thresholds, tp, fp, len(watermarked) - tp, len(non_watermarked) - fp

    def _find_best_threshold(self, inputs: List[DetectionResult]) -> float:
        """Find the best threshold that maximizes F1 (the first one if several do)."""
        thresholds, tp, fp, fn, tn = self._candidates(inputs)
        if len(thresholds) == 0:
            return 0
        denominator = 2 * tp + fp + fn
        f1 = np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 0)
        return float(thresholds[np.argmax(f1)])

    def _find_threshold_by_fpr(self, inputs: List[DetectionResult]) -> float:
        """Find the threshold that achieves the target FPR: the first candidate whose FPR is at most target_fpr (the last
        candidate if none is)."""
        thresholds, tp, fp, fn, tn = self._candidates(inputs)
        if len(thresholds) == 0:
            return 0
        fpr = np.where(fp + tn > 0, fp / np.maximum(fp + tn, 1), 0)
        ok = fpr <= self.target_fpr
        return float(thresholds[np.argmax(ok)] if ok.any() else thresholds[-1])

    def _find_threshold(self, inputs: List[DetectionResult]) -> float:
        """Find the threshold based on the specified rule."""
        # If the rule is to find the best threshold by maximizing F1
        if self.rule == 'best':
            return self._find_best_threshold(inputs)
        else:
            # If the rule is to find the threshold that achieves the target FPR
            return self._find_threshold_by_fpr(inputs)


class DynamicThresholdSuccessRateCalculator(BaseSuccessRateCalculator):
    """
        Calculator for success rates of watermark detection with dynamic thresholding.

        This class calculates success rates for watermark detection scenarios where the detection
        thresholds can dynamically change based on varying conditions. It supports evaluating the
        effectiveness of watermark detection algorithms that adapt to different signal or noise conditions.

        Use this class to evaluate detection systems where the threshold for detecting a watermark
        is not fixed and can vary. It is DetectionThreshold (the threshold and the decisions) followed by
        FundamentalSuccessRateCalculator (the rates at that threshold).
    """

    def __init__(self, 
                 labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], 
                 rule='best', 
                 target_fpr=None,
                 reverse=False) -> None:
        """
            Initialize the dynamic threshold success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
                rule (str): The rule for determining the threshold (see DetectionThreshold).
                target_fpr (float): The target false positive rate to achieve.
                reverse (bool): Whether to reverse the sorting order of the detection results.
                                False: higher values are considered positive.
                                True: lower values are considered positive.
        """
        super().__init__(labels)
        self.detection_threshold = DetectionThreshold(rule, target_fpr, reverse)
        self.rule = rule
        self.target_fpr = target_fpr
        self.reverse = reverse

    def calculate(self, watermarked_result: List[float], non_watermarked_result: List[float]) -> Dict[str, float]:
        """Calculate success rates based on provided results."""
        self._check_instance(watermarked_result + non_watermarked_result, float)

        threshold = self.detection_threshold.find_threshold(watermarked_result, non_watermarked_result)
        decide = self.detection_threshold.decide
        return FundamentalSuccessRateCalculator(self.labels).calculate(decide(watermarked_result, threshold),
                                                                       decide(non_watermarked_result, threshold))
