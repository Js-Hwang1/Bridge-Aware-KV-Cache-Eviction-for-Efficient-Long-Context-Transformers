"""
H2O Cache: Heavy-Hitter Oracle Baseline
========================================

Reference implementation of H2O for comparison.

Reference: Zhang et al., 2023 - "H2O: Heavy-Hitter Oracle for Efficient
           Generative Inference of Large Language Models"
           arXiv:2306.14048
"""

import torch
from typing import Optional, Tuple, Dict, Any

from ..scoring import ImportanceTracker


class H2OCache:
    """
    H2O (Heavy-Hitter Oracle) KV cache with cumulative attention tracking.

    Keeps tokens with highest cumulative attention across generation.
    Baseline for comparison with CAB.
    """

    def __init__(
        self,
        max_cache_size: int = 4096,
        sparsity: float = 0.9,
        eviction_interval: int = 10,
        device: str = 'cuda',
    ):
        """
        Initialize H2O cache.

        Args:
            max_cache_size: Maximum number of tokens to cache
            sparsity: Target sparsity (0.9 = keep 10% of tokens)
            eviction_interval: Evict every K tokens (amortization)
            device: Device for cache
        """
        self.max_cache_size = max_cache_size
        self.sparsity = sparsity
        self.eviction_interval = eviction_interval
        self.device = device

        # Cache storage (per layer)
        self.key_cache = []
        self.value_cache = []

        # Importance tracking
        self.importance_tracker = ImportanceTracker(device=device)

        # State
        self.tokens_since_last_eviction = 0
        self.total_evictions = 0

        # Statistics
        self.stats = {
            'total_tokens_processed': 0,
            'total_evictions': 0,
            'total_tokens_evicted': 0,
        }

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states.

        Args:
            key_states: [B, H, 1, D] new key states
            value_states: [B, H, 1, D] new value states
            layer_idx: Layer index
            attention_weights: [B, H, 1, N] attention weights

        Returns:
            keys: [B, H, N, D] updated key cache
            values: [B, H, N, D] updated value cache
        """
        # Ensure layer exists
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        # Append new states
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        # Update importance tracker (only for first layer)
        if layer_idx == 0 and attention_weights is not None:
            self.importance_tracker.update(attention_weights)

        # Update stats
        self.stats['total_tokens_processed'] += 1
        self.tokens_since_last_eviction += 1

        # Check if eviction needed
        cache_len = self.key_cache[layer_idx].shape[2]
        eviction_threshold = int(self.max_cache_size * 1.1)

        should_evict = (
            cache_len > eviction_threshold or
            self.tokens_since_last_eviction >= self.eviction_interval
        )

        if should_evict:
            self._evict()

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _evict(self):
        """Evict tokens using H2O policy (keep highest cumulative attention)."""
        if len(self.key_cache) == 0 or self.key_cache[0] is None:
            return

        # Get minimum cache length across all layers (handles async updates)
        cache_len = min(
            self.key_cache[i].shape[2]
            for i in range(len(self.key_cache))
            if self.key_cache[i] is not None
        )

        # Compute target size
        keep_ratio = 1.0 - self.sparsity
        keep_size = int(self.max_cache_size * keep_ratio * 0.9)

        if cache_len <= keep_size:
            return

        # Select indices with highest cumulative attention
        keep_indices = self.importance_tracker.get_top_k_indices(k=keep_size)

        # Sort indices to maintain order
        keep_indices = keep_indices.sort().values

        # Prune all layers
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_indices, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_indices, :]

        # Prune tracker
        self.importance_tracker.prune(keep_indices)

        # Update state
        tokens_evicted = cache_len - len(keep_indices)
        self.tokens_since_last_eviction = 0
        self.total_evictions += 1
        self.stats['total_evictions'] += 1
        self.stats['total_tokens_evicted'] += tokens_evicted

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length."""
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def get_max_length(self) -> int:
        """Get maximum cache length."""
        return self.max_cache_size

    def reset(self):
        """Reset cache to empty state."""
        self.key_cache = []
        self.value_cache = []
        self.importance_tracker.reset()
        self.tokens_since_last_eviction = 0
        self.total_evictions = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats,
            'current_cache_size': self.get_seq_length(0),
            'max_cache_size': self.max_cache_size,
        }

    def __len__(self) -> int:
        """Return current cache length."""
        return self.get_seq_length(0)

    def __repr__(self) -> str:
        return (
            f"H2OCache("
            f"size={self.get_seq_length(0)}/{self.max_cache_size}, "
            f"sparsity={self.sparsity:.0%}, "
            f"evictions={self.total_evictions})"
        )
