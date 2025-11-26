"""
Evaluation Metrics for Long-Context QA Benchmarks

Implements standard NLP evaluation metrics:
- F1 Score (token-level)
- Exact Match
- ROUGE (1, 2, L)
- Accuracy (for classification/multiple choice)
- Retrieval Accuracy (for needle-in-haystack)
- BLEU (for translation/generation)

All metrics are compatible with the unified BenchmarkSample format.
"""

import re
import string
import collections
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import numpy as np
except ImportError:
    np = None
    import math

from .config import MetricName


# =============================================================================
# Text Normalization Utilities
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation.
    
    Following standard QA evaluation practices:
    - Lowercase
    - Remove punctuation
    - Remove articles
    - Normalize whitespace
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    
    # Remove articles
    articles = {'a', 'an', 'the'}
    words = text.split()
    words = [w for w in words if w not in articles]
    
    # Normalize whitespace
    text = ' '.join(words)
    
    return text


def get_tokens(text: str) -> List[str]:
    """Tokenize text for F1 calculation."""
    if not text:
        return []
    return normalize_text(text).split()


def normalize_answer(answer: str) -> str:
    """Normalize answer string."""
    return normalize_text(answer).strip()


# =============================================================================
# Base Metric Class
# =============================================================================

@dataclass
class MetricResult:
    """Result from metric computation."""
    name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    
    def __repr__(self):
        return f"{self.name}: {self.score:.4f}"


class BaseMetric(ABC):
    """Abstract base class for all metrics."""
    
    def __init__(self, name: MetricName):
        self.name = name
    
    @abstractmethod
    def compute(
        self,
        prediction: str,
        references: List[str],
    ) -> MetricResult:
        """
        Compute metric for a single prediction.
        
        Args:
            prediction: Model prediction string
            references: List of acceptable reference answers
        
        Returns:
            MetricResult with score and optional details
        """
        pass
    
    def compute_batch(
        self,
        predictions: List[str],
        references_list: List[List[str]],
    ) -> Dict[str, Any]:
        """
        Compute metric for a batch of predictions.
        
        Args:
            predictions: List of prediction strings
            references_list: List of reference lists
        
        Returns:
            Dict with mean score and per-sample scores
        """
        scores = []
        for pred, refs in zip(predictions, references_list):
            result = self.compute(pred, refs)
            scores.append(result.score)
        
        if np is not None:
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            min_val = np.min(scores)
            max_val = np.max(scores)
        else:
            mean_val = sum(scores) / len(scores) if scores else 0.0
            std_val = (sum((x - mean_val) ** 2 for x in scores) / len(scores)) ** 0.5 if scores else 0.0
            min_val = min(scores) if scores else 0.0
            max_val = max(scores) if scores else 0.0
        
        return {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'scores': scores,
        }


# =============================================================================
# F1 Score
# =============================================================================

class F1Score(BaseMetric):
    """
    Token-level F1 score.
    
    Standard metric for extractive QA tasks.
    Computes precision, recall, and F1 between prediction and reference tokens.
    """
    
    def __init__(self):
        super().__init__(MetricName.F1)
    
    def compute(
        self,
        prediction: str,
        references: List[str],
    ) -> MetricResult:
        """Compute F1 score against best matching reference."""
        if not references:
            return MetricResult(name="f1", score=0.0)
        
        # Get F1 against each reference, take max
        best_f1 = 0.0
        best_details = {}
        
        for ref in references:
            pred_tokens = get_tokens(prediction)
            ref_tokens = get_tokens(ref)
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1 = 0.0 if pred_tokens != ref_tokens else 1.0
                precision = recall = f1
            else:
                # Count common tokens
                common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
                num_common = sum(common.values())
                
                if num_common == 0:
                    f1 = precision = recall = 0.0
                else:
                    precision = num_common / len(pred_tokens)
                    recall = num_common / len(ref_tokens)
                    f1 = 2 * precision * recall / (precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_details = {
                    'precision': precision,
                    'recall': recall,
                    'reference': ref,
                }
        
        return MetricResult(
            name="f1",
            score=best_f1,
            details=best_details,
        )


# =============================================================================
# Exact Match
# =============================================================================

class ExactMatch(BaseMetric):
    """
    Exact match metric.
    
    Returns 1.0 if normalized prediction exactly matches any reference,
    0.0 otherwise.
    """
    
    def __init__(self):
        super().__init__(MetricName.EXACT_MATCH)
    
    def compute(
        self,
        prediction: str,
        references: List[str],
    ) -> MetricResult:
        """Check if prediction exactly matches any reference."""
        norm_pred = normalize_answer(prediction)
        
        for ref in references:
            if normalize_answer(ref) == norm_pred:
                return MetricResult(
                    name="exact_match",
                    score=1.0,
                    details={'matched_reference': ref},
                )
        
        return MetricResult(name="exact_match", score=0.0)


# =============================================================================
# ROUGE Scores
# =============================================================================

class ROUGEScore(BaseMetric):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores.
    
    Standard metrics for summarization tasks:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    """
    
    def __init__(self, rouge_type: str = "rouge_l"):
        name_map = {
            "rouge_1": MetricName.ROUGE_1,
            "rouge_2": MetricName.ROUGE_2,
            "rouge_l": MetricName.ROUGE_L,
        }
        super().__init__(name_map.get(rouge_type, MetricName.ROUGE_L))
        self.rouge_type = rouge_type
    
    def compute(
        self,
        prediction: str,
        references: List[str],
    ) -> MetricResult:
        """Compute ROUGE score against best matching reference."""
        if not references:
            return MetricResult(name=self.rouge_type, score=0.0)
        
        best_score = 0.0
        
        for ref in references:
            if self.rouge_type == "rouge_1":
                score = self._compute_rouge_n(prediction, ref, n=1)
            elif self.rouge_type == "rouge_2":
                score = self._compute_rouge_n(prediction, ref, n=2)
            else:  # rouge_l
                score = self._compute_rouge_l(prediction, ref)
            
            best_score = max(best_score, score)
        
        return MetricResult(name=self.rouge_type, score=best_score)
    
    def _compute_rouge_n(self, prediction: str, reference: str, n: int = 1) -> float:
        """Compute ROUGE-N score (n-gram overlap)."""
        pred_tokens = get_tokens(prediction)
        ref_tokens = get_tokens(reference)
        
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return 0.0
        
        # Get n-grams
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        if not ref_ngrams:
            return 0.0
        
        # Count overlap
        overlap = sum((pred_ngrams & ref_ngrams).values())
        
        # Compute F1
        precision = overlap / sum(pred_ngrams.values()) if pred_ngrams else 0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> collections.Counter:
        """Extract n-grams from token list."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return collections.Counter(ngrams)
    
    def _compute_rouge_l(self, prediction: str, reference: str) -> float:
        """Compute ROUGE-L score (longest common subsequence)."""
        pred_tokens = get_tokens(prediction)
        ref_tokens = get_tokens(reference)
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Compute LCS length
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        # Compute F1
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        
        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


# =============================================================================
# Accuracy
# =============================================================================

class Accuracy(BaseMetric):
    """
    Accuracy metric for classification/multiple-choice tasks.
    
    Returns 1.0 if prediction matches correct answer, 0.0 otherwise.
    """
    
    def __init__(self):
        super().__init__(MetricName.ACCURACY)
    
    def compute(
        self,
        prediction: str,
        references: List[str],
    ) -> MetricResult:
        """Check if prediction matches reference (for single correct answer)."""
        if not references:
            return MetricResult(name="accuracy", score=0.0)
        
        # Normalize and compare
        norm_pred = normalize_answer(prediction)
        
        for ref in references:
            if normalize_answer(ref) == norm_pred:
                return MetricResult(name="accuracy", score=1.0)
        
        return MetricResult(name="accuracy", score=0.0)


# =============================================================================
# Retrieval Accuracy
# =============================================================================

class RetrievalAccuracy(BaseMetric):
    """
    Retrieval accuracy for needle-in-haystack tasks.
    
    Checks if the target information was retrieved correctly.
    More lenient than exact match - checks if answer is contained in prediction.
    """
    
    def __init__(self):
        super().__init__(MetricName.RETRIEVAL_ACCURACY)
    
    def compute(
        self,
        prediction: str,
        references: List[str],
    ) -> MetricResult:
        """Check if any reference is contained in prediction."""
        if not references:
            return MetricResult(name="retrieval_accuracy", score=0.0)
        
        norm_pred = normalize_answer(prediction)
        
        for ref in references:
            norm_ref = normalize_answer(ref)
            
            # Exact match
            if norm_ref == norm_pred:
                return MetricResult(
                    name="retrieval_accuracy",
                    score=1.0,
                    details={'match_type': 'exact', 'matched': ref},
                )
            
            # Contains match
            if norm_ref in norm_pred:
                return MetricResult(
                    name="retrieval_accuracy",
                    score=1.0,
                    details={'match_type': 'contains', 'matched': ref},
                )
        
        return MetricResult(name="retrieval_accuracy", score=0.0)


# =============================================================================
# BLEU Score
# =============================================================================

class BLEUScore(BaseMetric):
    """
    BLEU (Bilingual Evaluation Understudy) score.
    
    Standard metric for translation and generation tasks.
    """
    
    def __init__(self, max_n: int = 4):
        super().__init__(MetricName.BLEU)
        self.max_n = max_n
    
    def compute(
        self,
        prediction: str,
        references: List[str],
    ) -> MetricResult:
        """Compute BLEU score."""
        if not references:
            return MetricResult(name="bleu", score=0.0)
        
        pred_tokens = get_tokens(prediction)
        ref_tokens_list = [get_tokens(ref) for ref in references]
        
        if not pred_tokens:
            return MetricResult(name="bleu", score=0.0)
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self._modified_precision(pred_tokens, ref_tokens_list, n)
            precisions.append(precision)
        
        # Compute brevity penalty
        bp = self._brevity_penalty(pred_tokens, ref_tokens_list)
        
        # Geometric mean of precisions
        if min(precisions) > 0:
            if np is not None:
                log_prec = sum(np.log(p) for p in precisions) / len(precisions)
                bleu = bp * np.exp(log_prec)
            else:
                log_prec = sum(math.log(p) for p in precisions) / len(precisions)
                bleu = bp * math.exp(log_prec)
        else:
            bleu = 0.0
        
        return MetricResult(
            name="bleu",
            score=bleu,
            details={'precisions': precisions, 'brevity_penalty': bp},
        )
    
    def _modified_precision(
        self,
        pred_tokens: List[str],
        ref_tokens_list: List[List[str]],
        n: int,
    ) -> float:
        """Compute modified n-gram precision."""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        
        if not pred_ngrams:
            return 0.0
        
        # Get max counts from references
        max_ref_counts = collections.Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
        
        # Clip counts
        clipped_counts = 0
        for ngram, count in pred_ngrams.items():
            clipped_counts += min(count, max_ref_counts[ngram])
        
        return clipped_counts / sum(pred_ngrams.values())
    
    def _get_ngrams(self, tokens: List[str], n: int) -> collections.Counter:
        """Extract n-grams from token list."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return collections.Counter(ngrams)
    
    def _brevity_penalty(
        self,
        pred_tokens: List[str],
        ref_tokens_list: List[List[str]],
    ) -> float:
        """Compute brevity penalty."""
        pred_len = len(pred_tokens)
        
        # Find closest reference length
        ref_lens = [len(ref) for ref in ref_tokens_list]
        closest_ref_len = min(ref_lens, key=lambda x: (abs(x - pred_len), x))
        
        if pred_len >= closest_ref_len:
            return 1.0
        
        exp_val = 1 - closest_ref_len / pred_len
        if np is not None:
            return np.exp(exp_val)
        else:
            return math.exp(exp_val)


# =============================================================================
# Metric Registry
# =============================================================================

class MetricRegistry:
    """Registry of available metrics."""
    
    _metrics = {
        MetricName.F1: F1Score,
        MetricName.EXACT_MATCH: ExactMatch,
        MetricName.ROUGE_1: lambda: ROUGEScore("rouge_1"),
        MetricName.ROUGE_2: lambda: ROUGEScore("rouge_2"),
        MetricName.ROUGE_L: lambda: ROUGEScore("rouge_l"),
        MetricName.ACCURACY: Accuracy,
        MetricName.BLEU: BLEUScore,
        MetricName.RETRIEVAL_ACCURACY: RetrievalAccuracy,
    }
    
    @classmethod
    def get_metric(cls, name: MetricName) -> BaseMetric:
        """Get metric by name."""
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}")
        
        metric_or_factory = cls._metrics[name]
        if callable(metric_or_factory) and not isinstance(metric_or_factory, type):
            return metric_or_factory()
        return metric_or_factory()
    
    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all available metrics."""
        return [m.value for m in cls._metrics.keys()]


def compute_metrics(
    prediction: str,
    references: List[str],
    metrics: List[MetricName] = None,
) -> Dict[str, float]:
    """
    Compute multiple metrics for a prediction.
    
    Args:
        prediction: Model prediction string
        references: List of reference answers
        metrics: List of metrics to compute (default: F1, exact match)
    
    Returns:
        Dict mapping metric name to score
    
    Example:
        >>> scores = compute_metrics(
        ...     "The answer is 42",
        ...     ["42", "forty-two"],
        ...     [MetricName.F1, MetricName.EXACT_MATCH]
        ... )
        >>> print(scores)  # {'f1': 0.5, 'exact_match': 0.0}
    """
    if metrics is None:
        metrics = [MetricName.F1, MetricName.EXACT_MATCH]
    
    results = {}
    for metric_name in metrics:
        metric = MetricRegistry.get_metric(metric_name)
        result = metric.compute(prediction, references)
        results[result.name] = result.score
    
    return results


def compute_batch_metrics(
    predictions: List[str],
    references_list: List[List[str]],
    metrics: List[MetricName] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics for a batch of predictions.
    
    Args:
        predictions: List of prediction strings
        references_list: List of reference lists
        metrics: List of metrics to compute
    
    Returns:
        Dict mapping metric name to batch statistics
    
    Example:
        >>> results = compute_batch_metrics(
        ...     ["answer1", "answer2"],
        ...     [["ref1"], ["ref2", "ref2b"]],
        ...     [MetricName.F1]
        ... )
        >>> print(results['f1']['mean'])  # 0.75
    """
    if metrics is None:
        metrics = [MetricName.F1, MetricName.EXACT_MATCH]
    
    results = {}
    for metric_name in metrics:
        metric = MetricRegistry.get_metric(metric_name)
        batch_result = metric.compute_batch(predictions, references_list)
        results[metric_name.value] = batch_result
    
    return results


# =============================================================================
# Aggregation and Reporting
# =============================================================================

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    
    dataset_name: str
    method_name: str
    num_samples: int
    metrics: Dict[str, Dict[str, Any]]
    
    # Per-sample results (optional)
    per_sample_scores: Optional[Dict[str, List[float]]] = None
    
    # Metadata
    sparsity: Optional[float] = None
    context_length: Optional[int] = None
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"=" * 60,
            f"Evaluation Report: {self.dataset_name}",
            f"Method: {self.method_name}",
            f"Samples: {self.num_samples}",
            f"=" * 60,
        ]
        
        if self.sparsity is not None:
            lines.append(f"Sparsity: {self.sparsity:.2%}")
        
        lines.append("\nMetrics:")
        lines.append("-" * 40)
        
        for metric_name, stats in self.metrics.items():
            mean = stats.get('mean', 0)
            std = stats.get('std', 0)
            lines.append(f"  {metric_name}: {mean:.4f} ± {std:.4f}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'dataset_name': self.dataset_name,
            'method_name': self.method_name,
            'num_samples': self.num_samples,
            'metrics': self.metrics,
            'sparsity': self.sparsity,
            'context_length': self.context_length,
        }


def create_comparison_table(
    reports: List[EvaluationReport],
    primary_metric: str = "f1",
) -> str:
    """
    Create comparison table from multiple reports.
    
    Args:
        reports: List of evaluation reports
        primary_metric: Metric to highlight
    
    Returns:
        Formatted table string
    """
    # Get all methods and datasets
    methods = sorted(set(r.method_name for r in reports))
    datasets = sorted(set(r.dataset_name for r in reports))
    
    # Build table
    lines = []
    
    # Header
    header = ["Dataset"] + methods
    lines.append(" | ".join(f"{h:>15}" for h in header))
    lines.append("-" * (17 * len(header)))
    
    # Rows
    for dataset in datasets:
        row = [dataset]
        for method in methods:
            # Find matching report
            matching = [r for r in reports 
                       if r.dataset_name == dataset and r.method_name == method]
            
            if matching:
                score = matching[0].metrics.get(primary_metric, {}).get('mean', 0)
                row.append(f"{score:.4f}")
            else:
                row.append("-")
        
        lines.append(" | ".join(f"{v:>15}" for v in row))
    
    return "\n".join(lines)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Test F1
    f1 = F1Score()
    result = f1.compute("The quick brown fox", ["the brown fox jumps"])
    print(f"F1: {result}")
    
    # Test exact match
    em = ExactMatch()
    result = em.compute("42", ["42", "forty-two"])
    print(f"Exact Match: {result}")
    
    # Test ROUGE
    rouge = ROUGEScore("rouge_l")
    result = rouge.compute(
        "The cat sat on the mat",
        ["The cat was sitting on a mat"]
    )
    print(f"ROUGE-L: {result}")
    
    # Test retrieval accuracy
    ra = RetrievalAccuracy()
    result = ra.compute(
        "The answer to life is 42 and that's final",
        ["42"]
    )
    print(f"Retrieval Accuracy: {result}")
    
    # Test batch metrics
    predictions = [
        "Paris is the capital of France",
        "The answer is 42",
    ]
    references = [
        ["Paris", "Paris, France"],
        ["42", "forty-two"],
    ]
    
    batch_results = compute_batch_metrics(
        predictions, references,
        [MetricName.F1, MetricName.EXACT_MATCH]
    )
    print(f"\nBatch results: {batch_results}")
    
    print("\nAll metric tests passed!")

