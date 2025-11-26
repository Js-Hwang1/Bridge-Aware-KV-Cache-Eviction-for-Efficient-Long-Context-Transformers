"""
Perplexity Metrics for Language Model Evaluation

Provides:
- Standard perplexity computation
- Bits-per-character (BPC)
- Cross-entropy loss
- Sliding window perplexity (for long contexts)

ICML Publication-Quality Implementation:
- Numerically stable computation
- Proper handling of variable-length sequences
- Support for masked positions (padding)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PerplexityResult:
    """Result of perplexity computation."""
    
    perplexity: float                      # Main metric: PPL
    cross_entropy: float                   # Cross-entropy loss (nats)
    bits_per_token: float                  # BPT = CE / ln(2)
    bits_per_char: Optional[float] = None  # BPC (if char count available)
    
    # Statistics
    num_tokens: int = 0
    num_samples: int = 0
    
    # Per-sample data (optional)
    per_sample_ppl: Optional[List[float]] = None
    per_sample_ce: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'perplexity': self.perplexity,
            'cross_entropy': self.cross_entropy,
            'bits_per_token': self.bits_per_token,
            'bits_per_char': self.bits_per_char,
            'num_tokens': self.num_tokens,
            'num_samples': self.num_samples,
        }


# =============================================================================
# Core Perplexity Computation
# =============================================================================

def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
) -> Tuple[float, float, int]:
    """
    Compute perplexity from model logits.
    
    Args:
        logits: Model output logits [B, N, V] (V = vocab size)
        labels: Target token IDs [B, N]
        attention_mask: Mask for valid positions [B, N] (1 = valid)
        reduction: 'mean' for average PPL, 'sum' for total loss
    
    Returns:
        perplexity: exp(cross_entropy)
        cross_entropy: Average cross-entropy loss
        num_tokens: Number of tokens used in computation
    """
    B, N, V = logits.shape
    
    # Shift for causal LM: predict token t from tokens 0..t-1
    shift_logits = logits[:, :-1, :].contiguous()  # [B, N-1, V]
    shift_labels = labels[:, 1:].contiguous()       # [B, N-1]
    
    # Flatten for cross-entropy
    flat_logits = shift_logits.view(-1, V)          # [B*(N-1), V]
    flat_labels = shift_labels.view(-1)             # [B*(N-1)]
    
    # Handle attention mask
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous().view(-1)  # [B*(N-1)]
    else:
        shift_mask = torch.ones_like(flat_labels, dtype=torch.bool)
    
    # Compute cross-entropy loss (no reduction)
    loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')  # [B*(N-1)]
    
    # Apply mask
    loss = loss * shift_mask.float()
    
    # Count valid tokens
    num_tokens = shift_mask.sum().item()
    
    if num_tokens == 0:
        return float('nan'), float('nan'), 0
    
    # Compute average cross-entropy
    total_loss = loss.sum()
    avg_loss = total_loss / num_tokens
    
    # Perplexity = exp(cross_entropy)
    perplexity = torch.exp(avg_loss).item()
    cross_entropy = avg_loss.item()
    
    return perplexity, cross_entropy, int(num_tokens)


def compute_sliding_window_perplexity(
    model: Any,
    input_ids: torch.Tensor,
    stride: int,
    max_length: int,
    device: torch.device,
) -> Tuple[float, float, int]:
    """
    Compute perplexity with sliding window for long sequences.
    
    This is the standard method for evaluating LMs on sequences
    longer than the model's max context length.
    
    Args:
        model: HuggingFace model
        input_ids: Full sequence [N] or [1, N]
        stride: Sliding window stride
        max_length: Model's max context length
        device: Compute device
    
    Returns:
        perplexity, cross_entropy, num_tokens
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    seq_len = input_ids.size(1)
    
    nlls = []
    prev_end = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        target_len = end_loc - prev_end  # Tokens to evaluate in this window
        
        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        target_chunk = input_chunk.clone()
        
        # Mask tokens we've already seen (from overlap)
        if begin_loc > 0:
            overlap = max_length - stride
            target_chunk[:, :overlap] = -100  # Ignore these in loss
        
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            nll = outputs.loss * target_len  # Unnormalized loss
        
        nlls.append(nll.item())
        prev_end = end_loc
        
        if end_loc == seq_len:
            break
    
    # Compute overall perplexity
    total_nll = sum(nlls)
    num_tokens = seq_len - 1  # First token has no prediction
    
    if num_tokens == 0:
        return float('nan'), float('nan'), 0
    
    avg_nll = total_nll / num_tokens
    perplexity = math.exp(avg_nll)
    
    return perplexity, avg_nll, num_tokens


# =============================================================================
# Perplexity Evaluator Class
# =============================================================================

class PerplexityEvaluator:
    """
    Comprehensive perplexity evaluation.
    
    Features:
    - Batch processing
    - Sliding window for long contexts
    - Aggregation across samples
    - Memory-efficient computation
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device = None,
        max_length: int = 4096,
        stride: Optional[int] = None,
        use_sliding_window: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.max_length = max_length
        self.stride = stride or max_length
        self.use_sliding_window = use_sliding_window
        
        # Move model to eval mode
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, int]:
        """
        Evaluate perplexity on a batch.
        
        Args:
            input_ids: [B, N] token IDs
            attention_mask: [B, N] mask (optional)
        
        Returns:
            perplexity, cross_entropy, num_tokens
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        
        logits = outputs.logits  # [B, N, V]
        
        # Compute perplexity
        return compute_perplexity(logits, input_ids, attention_mask)
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader: Any,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> PerplexityResult:
        """
        Evaluate perplexity on entire dataset.
        
        Args:
            dataloader: PyTorch DataLoader
            max_samples: Optional limit on samples
            verbose: Print progress
        
        Returns:
            PerplexityResult with aggregated metrics
        """
        total_nll = 0.0
        total_tokens = 0
        num_samples = 0
        per_sample_ppl = []
        per_sample_ce = []
        
        for batch_idx, batch in enumerate(dataloader):
            if max_samples and num_samples >= max_samples:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            batch_size = input_ids.size(0)
            
            # Process each sample in batch
            for i in range(batch_size):
                sample_ids = input_ids[i:i+1]
                sample_mask = attention_mask[i:i+1] if attention_mask is not None else None
                
                if self.use_sliding_window and sample_ids.size(1) > self.max_length:
                    # Use sliding window for long sequences
                    ppl, ce, n_tokens = compute_sliding_window_perplexity(
                        self.model,
                        sample_ids,
                        stride=self.stride,
                        max_length=self.max_length,
                        device=self.device,
                    )
                else:
                    # Direct computation
                    ppl, ce, n_tokens = self.evaluate_batch(sample_ids, sample_mask)
                
                if not math.isnan(ce):
                    total_nll += ce * n_tokens
                    total_tokens += n_tokens
                    per_sample_ppl.append(ppl)
                    per_sample_ce.append(ce)
                
                num_samples += 1
                
                if max_samples and num_samples >= max_samples:
                    break
            
            if verbose and (batch_idx + 1) % 10 == 0:
                current_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('nan')
                logger.info(f"Processed {num_samples} samples, current PPL: {current_ppl:.2f}")
        
        # Compute final metrics
        if total_tokens == 0:
            return PerplexityResult(
                perplexity=float('nan'),
                cross_entropy=float('nan'),
                bits_per_token=float('nan'),
                num_tokens=0,
                num_samples=num_samples,
            )
        
        avg_ce = total_nll / total_tokens
        final_ppl = math.exp(avg_ce)
        bpt = avg_ce / math.log(2)  # Convert nats to bits
        
        return PerplexityResult(
            perplexity=final_ppl,
            cross_entropy=avg_ce,
            bits_per_token=bpt,
            num_tokens=total_tokens,
            num_samples=num_samples,
            per_sample_ppl=per_sample_ppl,
            per_sample_ce=per_sample_ce,
        )


# =============================================================================
# Utility Functions
# =============================================================================

def aggregate_results(
    results: List[PerplexityResult],
    weights: Optional[List[int]] = None,
) -> PerplexityResult:
    """
    Aggregate multiple perplexity results (e.g., from different datasets).
    
    Args:
        results: List of PerplexityResult
        weights: Optional weights (default: num_tokens)
    
    Returns:
        Aggregated PerplexityResult
    """
    if not results:
        return PerplexityResult(
            perplexity=float('nan'),
            cross_entropy=float('nan'),
            bits_per_token=float('nan'),
        )
    
    if weights is None:
        weights = [r.num_tokens for r in results]
    
    total_weight = sum(weights)
    if total_weight == 0:
        return PerplexityResult(
            perplexity=float('nan'),
            cross_entropy=float('nan'),
            bits_per_token=float('nan'),
        )
    
    # Weighted average of cross-entropy (then exp for PPL)
    weighted_ce = sum(r.cross_entropy * w for r, w in zip(results, weights)) / total_weight
    
    return PerplexityResult(
        perplexity=math.exp(weighted_ce),
        cross_entropy=weighted_ce,
        bits_per_token=weighted_ce / math.log(2),
        num_tokens=sum(r.num_tokens for r in results),
        num_samples=sum(r.num_samples for r in results),
    )


def format_perplexity_result(result: PerplexityResult) -> str:
    """Format result for display."""
    return (
        f"Perplexity: {result.perplexity:.2f} | "
        f"Cross-Entropy: {result.cross_entropy:.4f} | "
        f"BPT: {result.bits_per_token:.4f} | "
        f"Tokens: {result.num_tokens:,} | "
        f"Samples: {result.num_samples}"
    )


def compute_perplexity_statistics(
    per_sample_ppl: List[float],
) -> Dict[str, float]:
    """Compute statistics over per-sample perplexities."""
    import numpy as np
    
    arr = np.array([p for p in per_sample_ppl if not math.isnan(p)])
    
    if len(arr) == 0:
        return {'mean': float('nan'), 'std': float('nan')}
    
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
    }

