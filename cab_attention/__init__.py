"""
CAB: Curvature-Aware Block-Sparse Attention
=============================================

Efficient KV cache eviction using three components:
- Local context (recent tokens)
- Bridge tokens (low Forman-Ricci curvature connectors)
- Important tokens (high cumulative attention, H2O-style)

Usage:
    from cab_attention import CABCache

    cache = CABCache(
        max_cache_size=4096,
        sparsity=0.9,
        local_ratio=0.3,
        bridge_ratio=0.2,
        importance_ratio=0.5,
    )

    # Use with HuggingFace models
    outputs = model.generate(
        input_ids=input_ids,
        past_key_values=cache,
        max_new_tokens=512,
    )
"""

__version__ = "5.0.0"

from .cache.cab_cache import CABCache
from .cache.h2o_cache import H2OCache

__all__ = [
    "CABCache",
    "H2OCache",
]
