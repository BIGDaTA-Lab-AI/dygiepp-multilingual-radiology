from overrides import overrides
from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric

from dygie.training.f1 import compute_f1

# Import necessary modules
from typing import List, Tuple
from allennlp.nn.util import replace_masked_values

# TODO(dwadden) Need to use the decoded predictions so that we catch the gold examples longer than
# the span boundary.

class NERMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold labels.
    """
    def __init__(self, number_of_classes: int, none_label: int=0):
        self.number_of_classes = number_of_classes
        self.none_label = none_label
        self.reset()

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        predictions = predictions.cpu()
        gold_labels = gold_labels.cpu()
        mask = mask.cpu()
        for label in range(self.number_of_classes):
            if label == self.none_label:
                continue
            self._true_positives += ((predictions==label)*(gold_labels==label)*mask.bool()).sum().item()
            self._false_positives += ((predictions==label)*(gold_labels!=label)*mask.bool()).sum().item()
            self._true_negatives += ((predictions!=label)*(gold_labels!=label)*mask.bool()).sum().item()
            self._false_negatives += ((predictions!=label)*(gold_labels==label)*mask.bool()).sum().item()
            
            self._per_class_tp[label] += ((predictions == label) * (gold_labels == label) * mask.bool()).sum().item()
            self._per_class_fp[label] += ((predictions == label) * (gold_labels != label) * mask.bool()).sum().item()
            self._per_class_fn[label] += ((predictions != label) * (gold_labels == label) * mask.bool()).sum().item()
    
    # Helper function for safe division
    def _safe_divide(self, numerator, denominator):
        return numerator / denominator if denominator != 0 else 0.0
    
    @overrides
    def get_metric(self, reset=False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        predicted = self._true_positives + self._false_positives
        gold = self._true_positives + self._false_negatives
        matched = self._true_positives
        precision, recall, f1_measure = compute_f1(predicted, gold, matched)
        
        # Calculate per-class precision, recall, and F1
        per_class_precision = [self._safe_divide(tp, tp + fp) for tp, fp in zip(self._per_class_tp, self._per_class_fp)]
        per_class_recall = [self._safe_divide(tp, tp + fn) for tp, fn in zip(self._per_class_tp, self._per_class_fn)]
        per_class_f1 = [self._safe_divide(2 * p * r, p + r) for p, r in zip(per_class_precision, per_class_recall)]

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return precision, recall, f1_measure, per_class_precision, per_class_recall, per_class_f1

    @overrides
    def reset(self):
        self._true_positives = 0
        self._false_positives = 0
        self._true_negatives = 0
        self._false_negatives = 0
        
        # New variables to store per-class metrics
        self._per_class_tp = [0] * self.number_of_classes
        self._per_class_fp = [0] * self.number_of_classes
        self._per_class_fn = [0] * self.number_of_classes
